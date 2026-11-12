'''
Flock markers node: RViz view of the robots, their trajectories and the obstacle avoidance rule.

    sub: robot_i/odom               (nav_msgs/Odometry)         robot positions
         robot_i/reynolds/obstacle  (geometry_msgs/Vector3Stamped) obstacle avoidance acceleration of robot i
         flock/centroid             (geometry_msgs/PointStamped)   flock centroid (only with flock navigation)
         flock/path                 (nav_msgs/Path, latched)       a new path clears the trails (clear_on_new_path)
    pub: flock/markers              (visualization_msgs/MarkerArray)

Markers:
    robots      sphere on every robot: blue, orange while its obstacle avoidance rule is active,
                gray while its reynolds_node publishes nothing (disabled far robot, or not running)
    trail_i     trajectory of robot i, drawn behind it. Orange where the robot was avoiding an obstacle
    centroid    trajectory of the flock centroid, magenta

The trails are split in chunks of chunk_size points. A full chunk is published once and kept by RViz,
only the chunk being drawn is sent again at every update, so the message size does not grow with the run.
'''
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import Point, PointStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)

ROBOT_COLOR = ColorRGBA(r=0.15, g=0.45, b=1.0, a=1.0)     # blue
AVOIDING_COLOR = ColorRGBA(r=1.0, g=0.45, b=0.0, a=1.0)   # orange
INACTIVE_COLOR = ColorRGBA(r=0.55, g=0.55, b=0.55, a=1.0)  # gray
TRAIL_COLOR = ColorRGBA(r=0.15, g=0.45, b=1.0, a=0.6)     # translucent blue
CENTROID_COLOR = ColorRGBA(r=0.9, g=0.0, b=0.9, a=1.0)    # magenta


class Trail():
    '''Line strip split in chunks: only the last chunk changes'''
    def __init__(self, ns: str, width: float, z: float, min_step: float, chunk_size: int):
        self.ns, self.width, self.z = ns, width, z
        self.min_step, self.chunk_size = min_step, chunk_size
        self.reset()

    def reset(self):
        self.chunk_id = 0
        self.points, self.colors = [], []
        self.finished = []  # (id, points, colors) of full chunks not published yet
        self.last = None

    def add(self, p: np.ndarray, color: ColorRGBA):
        '''Add a point if it is at least min_step from the previous one'''
        if self.last is not None and np.linalg.norm(p - self.last) < self.min_step: return
        self.last = p
        self.points.append(Point(x=float(p[0]), y=float(p[1]), z=self.z))
        self.colors.append(color)
        if len(self.points) >= self.chunk_size:
            # Close the chunk; the next one starts at its last point so the line has no gap
            self.finished.append((self.chunk_id, self.points, self.colors))
            self.chunk_id += 1
            self.points, self.colors = [self.points[-1]], [self.colors[-1]]

    def markers(self, header) -> list:
        '''Full chunks not published yet + the current chunk'''
        chunks = self.finished + [(self.chunk_id, self.points, self.colors)]
        self.finished = []
        out = []
        for chunk_id, points, colors in chunks:
            if len(points) < 2: continue
            m = Marker(header=header, ns=self.ns, id=chunk_id, type=Marker.LINE_STRIP, action=Marker.ADD)
            m.pose.orientation.w = 1.0
            m.scale.x = self.width
            m.points, m.colors = points, colors
            m.color = colors[-1]
            out.append(m)
        return out


class FlockMarkers(Node):

    def __init__(self):
        super().__init__('flock_markers_node')

        # Declare parameters
        self.declare_parameter('number_robots', 6)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('rate', 10.0)            # [Hz] marker updates
        self.declare_parameter('robot_diameter', 0.09)  # [m] sphere drawn on each robot (the robot is 0.072 m)
        self.declare_parameter('trail_width', 0.015)    # [m]
        self.declare_parameter('trail_step', 0.02)      # [m] minimum distance between trail points
        self.declare_parameter('chunk_size', 100)       # trail points per marker
        self.declare_parameter('avoid_threshold', 1e-3)  # [m/s^2] obstacle acceleration above which a robot is avoiding
        self.declare_parameter('inactive_timeout', 0.5)  # [s] no acceleration message for this long: robot inactive
        self.declare_parameter('clear_on_new_path', True)  # a new flock/path (new goal) clears the trails

        # Get parameter values
        self.number_robots = self.get_parameter('number_robots').get_parameter_value().integer_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        rate = self.get_parameter('rate').get_parameter_value().double_value
        self.robot_diameter = self.get_parameter('robot_diameter').get_parameter_value().double_value
        width = self.get_parameter('trail_width').get_parameter_value().double_value
        step = self.get_parameter('trail_step').get_parameter_value().double_value
        chunk_size = max(2, self.get_parameter('chunk_size').get_parameter_value().integer_value)
        self.avoid_threshold = self.get_parameter('avoid_threshold').get_parameter_value().double_value
        self.inactive_timeout = self.get_parameter('inactive_timeout').get_parameter_value().double_value
        self.clear_on_new_path = self.get_parameter('clear_on_new_path').get_parameter_value().bool_value

        # Initialize variables
        n = self.number_robots
        self.poses = [None] * n
        self.avoiding = [False] * n
        self.last_accel = [None] * n  # time of the last obstacle acceleration message of each robot
        self.trails = [Trail(f'trail_{i}', width, 0.005, step, chunk_size) for i in range(n)]
        self.centroid_trail = Trail('centroid', 2 * width, 0.01, step, chunk_size)
        self.clear = False  # send a DELETEALL with the next update

        # Subscribers
        for i in range(n):
            self.create_subscription(Odometry, f'robot_{i}/odom', lambda msg, id=i: self.odom_callback(msg, id), 10)
            self.create_subscription(Vector3Stamped, f'robot_{i}/reynolds/obstacle',
                                     lambda msg, id=i: self.obstacle_callback(msg, id), 10)
        self.create_subscription(PointStamped, 'flock/centroid', self.centroid_callback, 10)
        self.create_subscription(Path, 'flock/path', self.path_callback, LATCHED_QOS)

        # Publisher
        self.marker_pub = self.create_publisher(MarkerArray, 'flock/markers', 10)
        self.timer = self.create_timer(1 / rate, self.timer_callback)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def active(self, i: int) -> bool:
        '''True while robot i's reynolds_node keeps publishing its accelerations'''
        return self.last_accel[i] is not None and self.now_sec() - self.last_accel[i] < self.inactive_timeout

    def robot_color(self, i: int) -> ColorRGBA:
        if not self.active(i): return INACTIVE_COLOR
        return AVOIDING_COLOR if self.avoiding[i] else ROBOT_COLOR

    def odom_callback(self, msg: Odometry, robot_id: int):
        p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.poses[robot_id] = p
        avoiding = self.avoiding[robot_id] and self.active(robot_id)
        self.trails[robot_id].add(p, AVOIDING_COLOR if avoiding else TRAIL_COLOR)

    def obstacle_callback(self, msg: Vector3Stamped, robot_id: int):
        self.avoiding[robot_id] = np.hypot(msg.vector.x, msg.vector.y) > self.avoid_threshold
        self.last_accel[robot_id] = self.now_sec()

    def centroid_callback(self, msg: PointStamped):
        self.centroid_trail.add(np.array([msg.point.x, msg.point.y]), CENTROID_COLOR)

    def path_callback(self, msg: Path):
        if not self.clear_on_new_path: return
        for trail in self.trails + [self.centroid_trail]:
            trail.reset()
        self.clear = True

    def timer_callback(self):
        header = Marker().header
        header.frame_id = self.map_frame
        header.stamp = self.get_clock().now().to_msg()

        markers = []
        if self.clear:
            markers.append(Marker(header=header, action=Marker.DELETEALL))
            self.clear = False

        for trail in self.trails + [self.centroid_trail]:
            markers += trail.markers(header)

        for i, p in enumerate(self.poses):
            if p is None: continue
            m = Marker(header=header, ns='robots', id=i, type=Marker.SPHERE, action=Marker.ADD)
            m.pose.position.x, m.pose.position.y = float(p[0]), float(p[1])
            m.pose.position.z = self.robot_diameter / 2
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = self.robot_diameter
            m.color = self.robot_color(i)
            markers.append(m)

        if markers:
            self.marker_pub.publish(MarkerArray(markers=markers))


def main(args=None):
    rclpy.init(args=args)
    node = FlockMarkers()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
