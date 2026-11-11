import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import SetBool

import numpy as np
from .utils.ReynoldsBack import ReynoldsRule
from .utils.ObstacleAvoidance import ObstacleAvoidance

# The map is published once and latched: a volatile subscriber would never receive it
LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)

def allocate(accels, max_acc):
    '''Prioritized acceleration allocation (Reynolds 1987): add the rules in priority order
    until the sum of their norms reaches max_acc. The rule that overflows is trimmed,
    the ones after it get nothing.'''
    total = np.zeros(2)
    used = {}
    remaining = max_acc
    for name, a in accels:  # highest priority first
        norm = np.linalg.norm(a)
        if norm > remaining:
            a = a * (remaining / norm)  # norm > remaining >= 0, so no division by zero
            norm = remaining
        used[name] = a
        total += a
        remaining -= norm
    return total, used


class BoidController(Node):

    def __init__(self):
        super().__init__('reynolds_node')

        # Declare parameters
        self.declare_parameter('robot_id', 1)
        self.declare_parameter('number_robots', 6)
        self.declare_parameter("ca", 1.0)  # alignment weight
        self.declare_parameter("cc", 1.0)  # cohesion weight
        self.declare_parameter("cs", 1.0)  # separation weight
        self.declare_parameter("f", 10.0)  # frequency for integration
        self.declare_parameter("max_range", 10.0)  # Neighbors maximum range
        self.declare_parameter("fov", 135.0)  # Neighbors maximum field of view
        self.declare_parameter("brake_acc", 0.1)  # braking deceleration [m/s^2] when no neighbors are visible (magnitude)
        self.declare_parameter("max_acc", 1.0)  # acceleration limit [m/s^2], in any direction (magnitude)
        self.declare_parameter("cn", 1.0)  # navigation weight (match the flock velocity from flock/cmd_vel)
        self.declare_parameter("nav_timeout", 1.0)  # [s] flock/cmd_vel older than this is ignored
        self.declare_parameter("co", 0.02)  # obstacle avoidance weight
        self.declare_parameter("obs_range", 0.5)  # [m] obstacles farther than this are ignored (potential influence distance)

        # Get parameter values
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        self.number_robots = self.get_parameter('number_robots').get_parameter_value().integer_value

        # Reynolds parameters
        ca = self.get_parameter('ca').get_parameter_value().double_value
        cc = self.get_parameter('cc').get_parameter_value().double_value
        cs = self.get_parameter('cs').get_parameter_value().double_value
        max_range = self.get_parameter('max_range').get_parameter_value().double_value
        fov = np.deg2rad(self.get_parameter('fov').get_parameter_value().double_value)/2

        # Reynolds Rules object
        self.ReynoldsAccs = ReynoldsRule(ca, cc, cs, max_range, fov, self.robot_id)

        self.dt = 1/self.get_parameter('f').get_parameter_value().double_value
        self.brake_acc = abs(self.get_parameter('brake_acc').get_parameter_value().double_value)  # the sign is applied in the control law
        self.max_acc = abs(self.get_parameter('max_acc').get_parameter_value().double_value)
        self.cn = self.get_parameter('cn').get_parameter_value().double_value
        self.nav_timeout = self.get_parameter('nav_timeout').get_parameter_value().double_value

        # Obstacle avoidance parameters
        self.co = self.get_parameter('co').get_parameter_value().double_value
        self.obs_range = self.get_parameter('obs_range').get_parameter_value().double_value

        # Initialize variables 
        self.robot_pose = None    # x, y, theta
        self.robot_vel = None     # linear_x, linear_y, angular_z
        self.neigh_poses = [None for _ in range(self.number_robots)]  # List of neighbor poses
        self.neigh_vels = [None for _ in range(self.number_robots)]  # List of neighbor velocities
        self.nav_vel = None    # flock velocity [vx, vy] from the flock controller
        self.nav_stamp = None  # arrival time of nav_vel
        self.avoid_obstacles = None        # ObstacleAvoidance, created when the map arrives
        self.enabled_navigation = True # Flag to enable/disable navigation rule (when robot is too far from centroid)

        # Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, LATCHED_QOS)  # Occupancy grid map, received once
        self.create_subscription(Odometry, f'robot_{self.robot_id}/odom', self.odom_callback, 10) # Odometry of robot i 
        for i in range(self.number_robots):
            if i == self.robot_id: continue  # Skip subscribing to its own odometry
            self.create_subscription(Odometry, f'robot_{i}/odom', lambda msg, id=i: self.neigh_odom_callback(msg, id), 10)
        
        self.create_subscription(Twist, 'flock/cmd_vel', self.nav_callback, 10)  # navigation velocity of the flock

        # Publisher for the velocity cmd
        self.publisher_ = self.create_publisher(Twist, f'robot_{self.robot_id}/cmd_vel', 10)

        # Publishers for the acceleration of each rule and the total (for plotting)
        self.accel_pubs = {
            name: self.create_publisher(Vector3Stamped, f'robot_{self.robot_id}/reynolds/{name}', 10)
            for name in ('alignment', 'cohesion', 'separation', 'navigation', 'obstacle', 'total')}

        # Service to enable/disable the navigation rule
        self.create_service(SetBool, f'robot_{self.robot_id}/reynolds/enable_navigation', self.enable_navigation_callback)

        # Main timer only if there are more than 1 robot
        if self.number_robots > 1: 
            self.timer = self.create_timer(self.dt, self.timer_callback)
        else: 
            self.get_logger().warn('No neighbors detected. Skipping Reynolds calculation.')

    def enable_navigation_callback(self, request, response):
        '''Service to enable/disable the navigation rule'''
        self.enabled_navigation = request.data
        response.success = True
        response.message = f'Navigation rule {"enabled" if self.enabled_navigation else "disabled"}'
        return response
    
    def map_callback(self, msg: OccupancyGrid):
        '''Build the obstacle avoidance from the map, then stop listening: the map is static'''
        self.avoid_obstacles = ObstacleAvoidance(msg, self.co, self.obs_range)
        self.destroy_subscription(self.map_sub)

    def nav_callback(self, msg: Twist):
        '''Store the flock velocity commanded by the flock controller'''
        self.nav_vel = np.array([msg.linear.x, msg.linear.y])
        self.nav_stamp = self.get_clock().now()

    def navigation_acc(self):
        '''Navigation rule: acceleration to match the flock velocity. None if there is no recent flock/cmd_vel'''
        if self.nav_vel is None: return np.zeros(2)
        if (self.get_clock().now() - self.nav_stamp).nanoseconds * 1e-9 > self.nav_timeout: return np.zeros(2)
        return self.cn * (self.nav_vel - self.robot_vel)

    
    def neigh_odom_callback(self, msg: Odometry, robot_id: int):
        # Extract the linear velocity from the odometry message
        pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])

        # Store the neighbor's pose and velocity
        self.neigh_poses[robot_id] = pose
        self.neigh_vels[robot_id] = vel
    
    def odom_callback(self, msg: Odometry):
        '''Store the robot's pose and velocity'''
        self.robot_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.robot_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])

    def timer_callback(self):
        '''Control loop that calculates the desired acceleration for robot i based on its neighbors following Reynolds Rules'''
        # No odometry from robot i arrived yet
        if self.robot_pose is None or self.robot_vel is None: return 

        # Reynolds acceleration
        reynolds_accs = self.ReynoldsAccs.solve_reynolds(self.robot_pose, self.robot_vel,
                                                         self.neigh_poses, self.neigh_vels)

        # Extract reynolds accelerations
        a_sep = reynolds_accs['separation']
        a_ali = reynolds_accs['alignment']
        a_coh = reynolds_accs['cohesion']
        a_nav = self.navigation_acc() if self.enabled_navigation else np.zeros(2)
        a_obs = self.avoid_obstacles.solve_obstacle(self.robot_pose) if self.avoid_obstacles is not None else np.zeros(2)

        # Acceleration calculaltion
        order = [('obstacle', a_obs), ('separation', a_sep), ('alignment', a_ali), ('cohesion', a_coh), ('navigation', a_nav)]
        x_dotdot, used = allocate(order, self.max_acc)
        self.publish_accels({**used, 'total': x_dotdot})


        # Apply the acceleration to update the robot's velocity in the world frame
        vd = self.robot_vel + x_dotdot * self.dt

        # Update the velocity
        msg_vel = Twist()
        msg_vel.linear.x = vd[0]
        msg_vel.linear.y = vd[1]

        # Publish the final local velocity 
        self.publisher_.publish(msg_vel)

    def publish_accels(self, accels):
        '''Publish each acceleration [ax, ay] on its debug topic'''
        stamp = self.get_clock().now().to_msg()
        for name, accel in accels.items():
            msg = Vector3Stamped()
            msg.header.stamp = stamp
            msg.header.frame_id = 'map'
            msg.vector.x = float(accel[0])
            msg.vector.y = float(accel[1])
            self.accel_pubs[name].publish(msg)

def main(args = None):
    rclpy.init(args=args)
    node = BoidController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()