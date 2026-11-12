import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import ColorRGBA, Float64
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker

import numpy as np
from .utils.ReynoldsBack import ReynoldsRule
from .utils.ObstacleAvoidance import ObstacleMap, PotentialField, SteerToAvoid

# The map is published once and latched: a volatile subscriber would never receive it
LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)

# Colors of the steer-to-avoid feelers in RViz
FEELER_CLEAR = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.6)    # look-ahead ray, nothing ahead
FEELER_BLOCKED = ColorRGBA(r=0.9, g=0.1, b=0.1, a=1.0)  # look-ahead ray up to the obstacle
FEELER_STEER = ColorRGBA(r=0.1, g=0.75, b=0.2, a=1.0)   # chosen free direction

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
        self.declare_parameter("brake_acc", 0.1)  # braking deceleration [m/s^2] of a robot disabled by the flock controller (too far from the centroid)
        self.declare_parameter("max_acc", 1.0)  # acceleration limit [m/s^2], in any direction (magnitude)
        
        self.declare_parameter("cn", 1.0)  # navigation weight (match the flock velocity from flock/cmd_vel)
        self.declare_parameter("nav_timeout", 1.0)  # [s] flock/cmd_vel older than this is ignored

        self.declare_parameter("max_speed", 0.5)  # [m/s] speed limit of the commanded velocity (velocity truncation)

        self.declare_parameter("co", 0.02)  # potential field (repulsion) weight, 0 disables it
        self.declare_parameter("obs_range", 0.5)  # [m] obstacles farther than this are ignored (potential influence distance)

        self.declare_parameter("csa", 2.0)  # steer-to-avoid weight, 0 disables it
        self.declare_parameter("steer_time", 1.5)  # [s] look-ahead time: look-ahead distance = steer_min_dist + steer_time * speed
        self.declare_parameter("steer_min_dist", 0.3)  # [m] look-ahead distance at zero speed
        self.declare_parameter("steer_radius", 0.12)  # [m] half width of the corridor that must be free (robot radius 0.0375 + margin)
        self.declare_parameter("steer_angle_step", 15.0)  # [deg] angle between the feelers of the fan
        self.declare_parameter("steer_max_angle", 90.0)  # [deg] largest deviation from the heading that is tried
        self.declare_parameter("steer_debug", True)  # publish the feelers on flock/feelers (RViz)

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

        self.max_speed = abs(self.get_parameter('max_speed').get_parameter_value().double_value)

        # Obstacle avoidance parameters
        self.co = self.get_parameter('co').get_parameter_value().double_value
        self.obs_range = self.get_parameter('obs_range').get_parameter_value().double_value
        self.csa = self.get_parameter('csa').get_parameter_value().double_value
        self.steer_time = self.get_parameter('steer_time').get_parameter_value().double_value
        self.steer_min_dist = self.get_parameter('steer_min_dist').get_parameter_value().double_value
        self.steer_radius = self.get_parameter('steer_radius').get_parameter_value().double_value
        self.steer_angle_step = np.deg2rad(self.get_parameter('steer_angle_step').get_parameter_value().double_value)
        self.steer_max_angle = np.deg2rad(self.get_parameter('steer_max_angle').get_parameter_value().double_value)
        self.steer_debug = self.get_parameter('steer_debug').get_parameter_value().bool_value

        # Initialize variables 
        self.robot_pose = None    # x, y, theta
        self.robot_vel = None     # linear_x, linear_y, angular_z
        self.neigh_poses = [None for _ in range(self.number_robots)]  # List of neighbor poses
        self.neigh_vels = [None for _ in range(self.number_robots)]  # List of neighbor velocities
        self.nav_vel = None    # flock velocity [vx, vy] from the flock controller
        self.nav_stamp = None  # arrival time of nav_vel
        self.potential_field = None        # PotentialField, created when the map arrives
        self.steer_to_avoid = None         # SteerToAvoid, created when the map arrives
        self.enabled = True # False while the flock controller disables this robot (too far from the centroid): it brakes to a stop

        # Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, LATCHED_QOS)  # Occupancy grid map, received once
        self.create_subscription(Odometry, f'robot_{self.robot_id}/odom', self.odom_callback, 10) # Odometry of robot i 
        for i in range(self.number_robots):
            if i == self.robot_id: continue  # Skip subscribing to its own odometry
            self.create_subscription(Odometry, f'robot_{i}/odom', lambda msg, id=i: self.neigh_odom_callback(msg, id), 10)
        
        self.create_subscription(Twist, 'flock/cmd_vel', self.nav_callback, 10)  # navigation velocity of the flock

        # Publisher for the velocity cmd
        self.publisher_ = self.create_publisher(Twist, f'robot_{self.robot_id}/cmd_vel', 10)

        # Publishers for the acceleration of each rule and the total, as a vector and as its magnitude (for plotting)
        rules = ('alignment', 'cohesion', 'separation', 'navigation', 'obstacle', 'steer', 'total')
        self.accel_pubs = {
            name: self.create_publisher(Vector3Stamped, f'robot_{self.robot_id}/reynolds/{name}', 10)
            for name in rules}
        self.accel_norm_pubs = {
            name: self.create_publisher(Float64, f'robot_{self.robot_id}/reynolds/{name}/magnitude', 10)
            for name in rules}

        # Steer-to-avoid feelers of every robot, one namespace per robot (RViz)
        self.feeler_pub = self.create_publisher(Marker, 'flock/feelers', 10)

        # Service to enable/disable the navigation rule
        self.create_service(SetBool, f'robot_{self.robot_id}/reynolds/enable', self.enable_callback)

        # Main timer only if there are more than 1 robot
        if self.number_robots > 1: 
            self.timer = self.create_timer(self.dt, self.timer_callback)
        else: 
            self.get_logger().warn('No neighbors detected. Skipping Reynolds calculation.')

    def enable_callback(self, request, response):
        '''Service to enable/disable the navigation rule'''
        self.enabled = request.data
        response.success = True
        response.message = f'Navigation rule {"enabled" if self.enabled else "disabled"}'
        return response
    
    def map_callback(self, msg: OccupancyGrid):
        '''Build both obstacle rules from the map, then stop listening: the map is static'''
        obstacle_map = ObstacleMap.from_msg(msg)
        self.potential_field = PotentialField(obstacle_map, self.co, self.obs_range)
        self.steer_to_avoid = SteerToAvoid(obstacle_map, self.csa, self.steer_time, self.steer_min_dist,
                                           self.steer_radius, self.steer_angle_step, self.steer_max_angle)
        self.destroy_subscription(self.map_sub)

    def nav_callback(self, msg: Twist):
        '''Store the flock velocity commanded by the flock controller'''
        self.nav_vel = np.array([msg.linear.x, msg.linear.y])
        self.nav_stamp = self.get_clock().now()

    def recent_nav_vel(self):
        '''The flock velocity from flock/cmd_vel, None if there is none or it is older than nav_timeout'''
        if self.nav_vel is None: return None
        if (self.get_clock().now() - self.nav_stamp).nanoseconds * 1e-9 > self.nav_timeout: return None
        return self.nav_vel

    def navigation_acc(self):
        '''Navigation rule: acceleration to match the flock velocity. 0 if there is no recent flock/cmd_vel'''
        nav_vel = self.recent_nav_vel()
        if nav_vel is None: return np.zeros(2)
        return self.cn * (nav_vel - self.robot_vel)

    
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
        if not self.enabled: 
            # Brake until stop if disabled
            if np.linalg.norm(self.robot_vel) > 5e-2:
                brake_acc = -self.brake_acc * self.robot_vel / np.linalg.norm(self.robot_vel)
                vd = self.robot_vel + brake_acc * self.dt
                msg_vel = Twist()
                msg_vel.linear.x = vd[0]
                msg_vel.linear.y = vd[1]
                self.publisher_.publish(msg_vel)
            return
        
        # Reynolds acceleration
        reynolds_accs = self.ReynoldsAccs.solve_reynolds(self.robot_pose, self.robot_vel,
                                                         self.neigh_poses, self.neigh_vels)

        # Extract reynolds accelerations
        a_sep = reynolds_accs['separation']
        a_ali = reynolds_accs['alignment']
        a_coh = reynolds_accs['cohesion']
        a_nav = self.navigation_acc() if self.enabled else np.zeros(2)
        a_obs, a_steer = np.zeros(2), np.zeros(2)
        if self.potential_field is not None:  # the map has arrived
            a_obs = self.potential_field.solve(self.robot_pose)
            a_steer, feelers = self.steer_to_avoid.solve(self.robot_pose, self.robot_vel, self.recent_nav_vel())
            if self.steer_debug and self.csa != 0.0: self.publish_feelers(feelers)

        # Prioritized allocation: obstacles first (repulsion is the close range emergency rule), then the flock
        order = [('obstacle', a_obs), ('steer', a_steer), ('separation', a_sep), ('cohesion', a_coh),
                 ('navigation', a_nav), ('alignment', a_ali)]
        x_dotdot, used = allocate(order, self.max_acc)
        self.publish_accels({**used, 'total': x_dotdot})


        # Apply the acceleration to update the robot's velocity in the world frame, truncated to max_speed
        vd = self.robot_vel + x_dotdot * self.dt
        speed = np.linalg.norm(vd)
        if speed > self.max_speed: vd = vd * self.max_speed / speed

        # Update the velocity
        msg_vel = Twist()
        msg_vel.linear.x = vd[0]
        msg_vel.linear.y = vd[1]

        # Publish the final local velocity 
        self.publisher_.publish(msg_vel)

    def publish_accels(self, accels):
        '''Publish each acceleration [ax, ay] and its magnitude on its debug topics'''
        stamp = self.get_clock().now().to_msg()
        for name, accel in accels.items():
            msg = Vector3Stamped()
            msg.header.stamp = stamp
            msg.header.frame_id = 'map'
            msg.vector.x = float(accel[0])
            msg.vector.y = float(accel[1])
            self.accel_pubs[name].publish(msg)
            self.accel_norm_pubs[name].publish(Float64(data=float(np.linalg.norm(accel))))

    def publish_feelers(self, feelers):
        '''Steer-to-avoid feelers in RViz: the look-ahead ray (gray if clear, red up to the obstacle)
        and the chosen free direction (green). Nothing while the robot is too slow to have a heading'''
        if feelers is None: return
        p = self.robot_pose
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns = f'robot_{self.robot_id}'
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.01
        m.lifetime = Duration(sec=0, nanosec=300_000_000)  # disappears when the robot stops publishing

        def segment(a, b, color):
            m.points += [Point(x=float(a[0]), y=float(a[1]), z=0.02), Point(x=float(b[0]), y=float(b[1]), z=0.02)]
            m.colors += [color, color]

        blocked = feelers['hit'] < feelers['length']
        segment(p, p + feelers['hit'] * feelers['heading'], FEELER_BLOCKED if blocked else FEELER_CLEAR)
        if feelers['direction'] is not None:
            segment(p, p + feelers['length'] * feelers['direction'], FEELER_STEER)
        self.feeler_pub.publish(m)

def main(args = None):
    rclpy.init(args=args)
    node = BoidController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()