'''
Flock controller node.

Follows the centroid path with a proportional waypoint controller and publishes the holonomic
velocity of the flock. Each reynolds_node uses it as the navigation rule.

    sub: flock/path   (nav_msgs/Path, latched)  centroid path from flock_planning_node
         robot_i/odom (nav_msgs/Odometry)       positions of every robot -> flock centroid
    pub: flock/cmd_vel  (geometry_msgs/Twist)        desired flock velocity, world frame (linear.x, linear.y)
         flock/centroid (geometry_msgs/PointStamped) current flock centroid (RViz / plotting)

Control law, with c the centroid and w the active waypoint:
    v = K (w - c), saturated to v_max
w switches to the next waypoint when |w - c| < waypoint_tol. The goal is reached when |goal - c| < goal_tol.
Nothing is published until the first path arrives, so the flock behaves as plain Reynolds without a goal.
After the goal is reached, a zero velocity is published to keep the flock stopped there.
'''
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry, Path

LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


class FlockController(Node):

    def __init__(self):
        super().__init__('flock_controller_node')

        # Declare parameters
        self.declare_parameter('number_robots', 6)
        self.declare_parameter('f', 10.0)  # control frequency [Hz]
        self.declare_parameter('k', 1.0)  # proportional gain [1/s]
        self.declare_parameter('v_max', 0.3)  # flock speed limit [m/s]
        self.declare_parameter('waypoint_tol', 0.3)  # [m] distance to switch to the next waypoint
        self.declare_parameter('goal_tol', 0.1)  # [m] distance to consider the goal reached
        self.declare_parameter('map_frame', 'map')

        # Get parameter values
        self.number_robots = self.get_parameter('number_robots').get_parameter_value().integer_value
        self.dt = 1 / self.get_parameter('f').get_parameter_value().double_value
        self.k = self.get_parameter('k').get_parameter_value().double_value
        self.v_max = abs(self.get_parameter('v_max').get_parameter_value().double_value)
        self.waypoint_tol = self.get_parameter('waypoint_tol').get_parameter_value().double_value
        self.goal_tol = self.get_parameter('goal_tol').get_parameter_value().double_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value

        # Initialize variables
        self.robot_poses = [None for _ in range(self.number_robots)]
        self.path = None  # (N, 2) array of waypoints
        self.wp_idx = 0  # active waypoint
        self.goal_reached = False

        # Subscribers
        self.create_subscription(Path, 'flock/path', self.path_callback, LATCHED_QOS)
        for i in range(self.number_robots):
            self.create_subscription(Odometry, f'robot_{i}/odom', lambda msg, id=i: self.odom_callback(msg, id), 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'flock/cmd_vel', 10)
        self.centroid_pub = self.create_publisher(PointStamped, 'flock/centroid', 10)

        self.timer = self.create_timer(self.dt, self.timer_callback)

    def odom_callback(self, msg: Odometry, robot_id: int):
        self.robot_poses[robot_id] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

    def path_callback(self, msg: Path):
        if len(msg.poses) == 0:
            self.get_logger().info('Empty path received: navigation stopped')
            self.path = None
            return
        self.path = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        self.wp_idx = 0
        self.goal_reached = False
        self.get_logger().info(f'New path: {len(self.path)} waypoints to ({self.path[-1, 0]:.2f}, {self.path[-1, 1]:.2f})')

    def centroid(self):
        '''Mean position of the flock, None until every robot has been seen'''
        if any(p is None for p in self.robot_poses): return None
        return np.mean(self.robot_poses, axis=0)

    # def disable_far_robots(self, max_dist: float):
    #     '''Disable robots that are too far from the centroid using service calls to the reynolds_node'''

    def timer_callback(self):
        c = self.centroid()
        if c is None: return
        self.publish_centroid(c)

        if self.path is None: return  # no goal yet: do not interfere with the Reynolds rules
        if self.goal_reached:
            self.publish_cmd(np.zeros(2))
            return

        # Switch to the next waypoint once the current one is close enough (never skip the goal)
        last = len(self.path) - 1
        while self.wp_idx < last and np.linalg.norm(self.path[self.wp_idx] - c) < self.waypoint_tol:
            self.wp_idx += 1

        error = self.path[self.wp_idx] - c
        if self.wp_idx == last and np.linalg.norm(error) < self.goal_tol:
            self.goal_reached = True
            self.get_logger().info(f'Goal reached: centroid at ({c[0]:.2f}, {c[1]:.2f})')
            self.publish_cmd(np.zeros(2))
            return

        # Proportional control with saturation
        v = self.k * error
        speed = np.linalg.norm(v)
        if speed > self.v_max: v = v * self.v_max / speed
        self.publish_cmd(v)

    def publish_cmd(self, v: np.ndarray):
        msg = Twist()
        msg.linear.x = float(v[0])
        msg.linear.y = float(v[1])
        self.cmd_pub.publish(msg)

    def publish_centroid(self, c: np.ndarray):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.point.x = float(c[0])
        msg.point.y = float(c[1])
        self.centroid_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FlockController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
