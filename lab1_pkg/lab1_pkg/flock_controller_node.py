'''
Flock controller node.

Follows the centroid path with a proportional waypoint controller and publishes the holonomic
velocity of the flock. Each reynolds_node uses it as the navigation rule.

    sub: flock/path   (nav_msgs/Path, latched)  centroid path from flock_planning_node
         robot_i/odom (nav_msgs/Odometry)       positions of every robot -> flock centroid
    pub: flock/cmd_vel  (geometry_msgs/Twist)        desired flock velocity, world frame (linear.x, linear.y)
         flock/centroid (geometry_msgs/PointStamped) current flock centroid (RViz / plotting)
         flock/velocity (geometry_msgs/Vector3Stamped) actual centroid velocity = mean actual velocity of the robots
         flock/speed    (std_msgs/Float64)           norm of flock/velocity (plotting)
         flock/cmd_speed (std_msgs/Float64)          norm of flock/cmd_vel, the commanded flock speed (plotting)
The actual velocity of each robot is the derivative of its odom position over the last vel_window seconds:
Stage reports the commanded velocity in the odom twist, even while the robot is blocked by an obstacle.
         flock/goal_distance (std_msgs/Float64)      straight line distance centroid -> goal (plotting)
         flock/path_distance (std_msgs/Float64)      distance centroid -> goal along the remaining path (plotting)
The centroid and its velocity use the same robots: all of them except the ones farther than max_dist.
The distances are only published while there is a path.

Control law (arrival), with c the centroid, w the active waypoint and D the distance c -> goal along the path:
    v = min(v_max, K D) (w - c) / |w - c|
The flock moves at v_max and slows down linearly over the last v_max / K meters of the path, until it stops
at the goal. D >= |w - c|, so the intermediate waypoints never slow it down.
w switches to the next waypoint when |w - c| < waypoint_tol. The goal is reached when |goal - c| < goal_tol.
Nothing is published until the first path arrives, so the flock behaves as plain Reynolds without a goal.
After the goal is reached, a zero velocity is published to keep the flock stopped there.
'''
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64
from std_srvs.srv import SetBool

LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)

class FlockController(Node):

    def __init__(self):
        super().__init__('flock_controller_node')

        # Declare parameters
        self.declare_parameter('number_robots', 6)
        self.declare_parameter('f', 10.0)  # control frequency [Hz]
        self.declare_parameter('k', 1.0)   # arrival gain [1/s]: the flock slows down over the last v_max / k meters
        self.declare_parameter('v_max', 0.3)  # flock speed limit [m/s]
        self.declare_parameter('waypoint_tol', 0.3)  # [m] distance to switch to the next waypoint
        self.declare_parameter('goal_tol', 0.1)  # [m] distance to consider the goal reached
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('max_dist', 2.0)             # [m] robots farther from the centroid stop following flock/cmd_vel
        self.declare_parameter('max_dist_hysteresis', 0.3)  # [m] navigation is enabled again below max_dist - hysteresis
        self.declare_parameter('vel_window', 0.5)  # [s] time window to differentiate the robot positions

        # Get parameter values
        self.number_robots = self.get_parameter('number_robots').get_parameter_value().integer_value
        self.dt = 1 / self.get_parameter('f').get_parameter_value().double_value
        self.k = self.get_parameter('k').get_parameter_value().double_value
        self.v_max = abs(self.get_parameter('v_max').get_parameter_value().double_value)
        self.waypoint_tol = self.get_parameter('waypoint_tol').get_parameter_value().double_value
        self.goal_tol = self.get_parameter('goal_tol').get_parameter_value().double_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.max_dist = self.get_parameter('max_dist').get_parameter_value().double_value
        self.max_dist_hysteresis = self.get_parameter('max_dist_hysteresis').get_parameter_value().double_value
        self.vel_window = self.get_parameter('vel_window').get_parameter_value().double_value

        # Initialize variables
        self.robot_poses = [None for _ in range(self.number_robots)]
        self.robot_history = [deque() for _ in range(self.number_robots)]  # (t, [x, y]) of the last vel_window seconds
        self.path = None  # (N, 2) array of waypoints
        self.wp_idx = 0  # active waypoint
        self.goal_reached = False

        # Clients of the navigation rule switch of every reynolds_node
        self.nav_clients = [self.create_client(SetBool, f'robot_{i}/reynolds/enable')
                            for i in range(self.number_robots)]
        self.nav_enabled = [True] * self.number_robots  # last state confirmed by each reynolds_node
        self.nav_pending = [False] * self.number_robots  # request waiting for its response


        # Subscribers
        self.create_subscription(Path, 'flock/path', self.path_callback, LATCHED_QOS)
        for i in range(self.number_robots):
            self.create_subscription(Odometry, f'robot_{i}/odom', lambda msg, id=i: self.odom_callback(msg, id), 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'flock/cmd_vel', 10)
        self.centroid_pub = self.create_publisher(PointStamped, 'flock/centroid', 10)
        self.velocity_pub = self.create_publisher(Vector3Stamped, 'flock/velocity', 10)
        self.speed_pub = self.create_publisher(Float64, 'flock/speed', 10)
        self.cmd_speed_pub = self.create_publisher(Float64, 'flock/cmd_speed', 10)
        self.goal_dist_pub = self.create_publisher(Float64, 'flock/goal_distance', 10)
        self.path_dist_pub = self.create_publisher(Float64, 'flock/path_distance', 10)

        self.timer = self.create_timer(self.dt, self.timer_callback)

    def odom_callback(self, msg: Odometry, robot_id: int):
        p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.robot_poses[robot_id] = p

        # Position history for the actual velocity: keep just enough samples to span vel_window
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        history = self.robot_history[robot_id]
        if history and t <= history[-1][0]: return  # repeated or out of order stamp
        history.append((t, p))
        while len(history) > 2 and t - history[1][0] >= self.vel_window:
            history.popleft()

    def robot_velocity(self, robot_id):
        '''Actual velocity of a robot: position difference over the history window (0 with < 2 samples)'''
        history = self.robot_history[robot_id]
        if len(history) < 2: return np.zeros(2)
        (t0, p0), (t1, p1) = history[0], history[-1]
        return (p1 - p0) / (t1 - t0)

    def path_callback(self, msg: Path):
        if len(msg.poses) == 0:
            self.get_logger().info('Empty path received: navigation stopped')
            self.path = None
            return
        self.path = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        self.wp_idx = 0
        self.goal_reached = False
        self.get_logger().info(f'New path: {len(self.path)} waypoints to ({self.path[-1, 0]:.2f}, {self.path[-1, 1]:.2f})')

    def flock_members(self):
        '''Boolean mask of the robots that form the flock: the ones within max_dist of the full mean.
        None until every robot has been seen'''
        if any(p is None for p in self.robot_poses): return None
        poses = np.array(self.robot_poses)
        close = np.linalg.norm(poses - poses.mean(axis=0), axis=1) <= self.max_dist
        if not close.any(): close[:] = True  # every robot is "far" (e.g. the flock split in two): keep them all
        return close

    def centroid(self, members):
        '''Mean position of the flock members'''
        return np.array(self.robot_poses)[members].mean(axis=0)

    def centroid_velocity(self, members):
        '''Mean actual velocity of the flock members (= velocity of their centroid)'''
        return np.mean([self.robot_velocity(i) for i in np.flatnonzero(members)], axis=0)

    def remaining_path_length(self, c):
        '''Distance from c to the goal along the path: to the active waypoint, then waypoint to waypoint'''
        segments = np.linalg.norm(np.diff(self.path[self.wp_idx:], axis=0), axis=1)
        return float(np.linalg.norm(self.path[self.wp_idx] - c) + segments.sum())

    def disable_far_robots(self, centroid, max_dist):
        '''Disable the navigation rule of robots farther than max_dist from the centroid, and enable it again
        when they come back closer than max_dist - max_dist_hysteresis. A request is only sent when the
        state of a robot has to change, and asynchronously: a blocking call() inside a timer callback
        deadlocks the single-threaded executor that has to process the response.'''
        for i, pose in enumerate(self.robot_poses):
            dist = np.linalg.norm(pose - centroid)
            if self.nav_enabled[i] and dist > max_dist:
                enable = False
            elif not self.nav_enabled[i] and dist < max_dist - self.max_dist_hysteresis:
                enable = True
            else:
                continue  # already in the right state

            # One request at a time per robot, and only if its reynolds_node is up
            if self.nav_pending[i] or not self.nav_clients[i].service_is_ready(): continue
            self.nav_pending[i] = True
            future = self.nav_clients[i].call_async(SetBool.Request(data=enable))
            future.add_done_callback(lambda fut, i=i, enable=enable: self.enable_navigation_done(fut, i, enable))

    def enable_navigation_done(self, future, robot_id, enable):
        '''Response of robot_i/reynolds/enable_navigation: store the confirmed state'''
        self.nav_pending[robot_id] = False
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warn(f'robot_{robot_id}: enable_navigation call failed: {e}')
            return
        if response.success:
            self.nav_enabled[robot_id] = enable
            self.get_logger().info(f'robot_{robot_id}: {response.message}')
        else:
            self.get_logger().warn(f'robot_{robot_id}: {response.message}')



    def timer_callback(self):
        members = self.flock_members()
        if members is None: return
        c = self.centroid(members)
        self.publish_centroid(c)
        self.publish_velocity(self.centroid_velocity(members))

        if self.path is None: return  # no goal yet: do not interfere with the Reynolds rules
        self.disable_far_robots(c, self.max_dist)

        # Switch to the next waypoint once the current one is close enough (never skip the goal)
        last = len(self.path) - 1
        while self.wp_idx < last and np.linalg.norm(self.path[self.wp_idx] - c) < self.waypoint_tol:
            self.wp_idx += 1
        self.publish_distances(c)

        if self.goal_reached:
            self.publish_cmd(np.zeros(2))
            return

        error = self.path[self.wp_idx] - c
        if self.wp_idx == last and np.linalg.norm(error) < self.goal_tol:
            self.goal_reached = True
            self.get_logger().info(f'Goal reached: centroid at ({c[0]:.2f}, {c[1]:.2f})')
            self.publish_cmd(np.zeros(2))
            return

        # Arrival: speed proportional to the remaining path length, saturated to v_max, towards the waypoint
        dist = np.linalg.norm(error)
        if dist < 1e-6:
            self.publish_cmd(np.zeros(2))
            return
        speed = min(self.v_max, self.k * self.remaining_path_length(c))
        self.publish_cmd(speed * error / dist)

    def publish_cmd(self, v: np.ndarray):
        msg = Twist()
        msg.linear.x = float(v[0])
        msg.linear.y = float(v[1])
        self.cmd_pub.publish(msg)
        self.cmd_speed_pub.publish(Float64(data=float(np.linalg.norm(v))))

    def publish_centroid(self, c: np.ndarray):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.point.x = float(c[0])
        msg.point.y = float(c[1])
        self.centroid_pub.publish(msg)

    def publish_velocity(self, v: np.ndarray):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.vector.x = float(v[0])
        msg.vector.y = float(v[1])
        self.velocity_pub.publish(msg)
        self.speed_pub.publish(Float64(data=float(np.linalg.norm(v))))

    def publish_distances(self, c: np.ndarray):
        self.goal_dist_pub.publish(Float64(data=float(np.linalg.norm(self.path[-1] - c))))
        self.path_dist_pub.publish(Float64(data=self.remaining_path_length(c)))


def main(args=None):
    rclpy.init(args=args)
    node = FlockController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
