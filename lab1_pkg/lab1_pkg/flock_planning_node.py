'''
Flock planning node.

Loads the simulation map, publishes it as an OccupancyGrid and plans a path for the flock
centroid with RRT-Connect, published as a nav_msgs/Path.

    sub: robot_i/odom (nav_msgs/Odometry)        positions of every robot -> flock centroid
         goal_pose    (geometry_msgs/PoseStamped) new goal (RViz '2D Goal Pose'), replans from the current centroid
    pub: map                (nav_msgs/OccupancyGrid, latched)  raw map
         flock/map_inflated (nav_msgs/OccupancyGrid, latched)  map inflated by inflation_radius (planner C-space)
         flock/path         (nav_msgs/Path, latched)            centroid path, start -> goal
'''
import os
import time
from math import atan2, cos, sin

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path

from lab1_pkg.grid_map import GridMap
from lab1_pkg.rrt_connect import RRT_Connect

LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


class FlockPlanner(Node):

    def __init__(self):
        super().__init__('flock_planning_node')

        # Declare parameters
        self.declare_parameter('map_yaml', '')  # map_server yaml, set by the launch file
        self.declare_parameter('map_name', 'empty_10x10')  # used to find the sphero_stage map if map_yaml is empty
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('number_robots', 6)
        self.declare_parameter('inflation_radius', 0.2)  # [m] obstacle inflation for the centroid
        self.declare_parameter('rrt_step', 0.25)  # [m] RRT-Connect extension step
        self.declare_parameter('rrt_max_iter', 5000)
        self.declare_parameter('rrt_attempts', 3)  # RRT-Connect restarts before giving up
        self.declare_parameter('rrt_seed', -1)  # -1: random
        self.declare_parameter('waypoint_spacing', 0.5)  # [m] max distance between path waypoints
        self.declare_parameter('plan_on_start', True)  # plan to (goal_x, goal_y) once every robot is seen
        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 0.0)

        # Get parameter values
        map_yaml = self.get_parameter('map_yaml').get_parameter_value().string_value
        map_name = self.get_parameter('map_name').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.number_robots = self.get_parameter('number_robots').get_parameter_value().integer_value
        inflation_radius = self.get_parameter('inflation_radius').get_parameter_value().double_value
        rrt_step = self.get_parameter('rrt_step').get_parameter_value().double_value
        rrt_max_iter = self.get_parameter('rrt_max_iter').get_parameter_value().integer_value
        self.rrt_attempts = max(1, self.get_parameter('rrt_attempts').get_parameter_value().integer_value)
        rrt_seed = self.get_parameter('rrt_seed').get_parameter_value().integer_value
        self.waypoint_spacing = self.get_parameter('waypoint_spacing').get_parameter_value().double_value
        plan_on_start = self.get_parameter('plan_on_start').get_parameter_value().bool_value
        goal = np.array([self.get_parameter('goal_x').get_parameter_value().double_value,
                         self.get_parameter('goal_y').get_parameter_value().double_value])

        # Build the grid map
        if not map_yaml:
            from ament_index_python.packages import get_package_share_directory
            map_yaml = os.path.join(get_package_share_directory('sphero_stage'), 'config', 'maps', map_name, f'{map_name}.yaml')
        self.grid = GridMap.from_yaml(map_yaml, inflation_radius)
        self.rrt = RRT_Connect(self.grid, rrt_step, rrt_max_iter, seed=None if rrt_seed < 0 else rrt_seed)
        self.get_logger().info(f'Loaded map {map_yaml}: {self.grid.width}x{self.grid.height} cells, '
                               f'{self.grid.resolution} m/cell, inflation {inflation_radius} m')

        # Initialize variables
        self.robot_poses = [None for _ in range(self.number_robots)]
        self.pending_goal = goal if plan_on_start else None  # goal waiting for the centroid to be available

        # Publishers (latched: late subscribers and RViz get the last message)
        self.map_pub = self.create_publisher(OccupancyGrid, 'map', LATCHED_QOS)
        self.inflated_pub = self.create_publisher(OccupancyGrid, 'flock/map_inflated', LATCHED_QOS)
        self.path_pub = self.create_publisher(Path, 'flock/path', LATCHED_QOS)
        self.map_pub.publish(self.grid_msg(self.grid.occupancy_data()))
        self.inflated_pub.publish(self.grid_msg(self.grid.inflated_data()))

        # Subscribers
        for i in range(self.number_robots):
            self.create_subscription(Odometry, f'robot_{i}/odom', lambda msg, id=i: self.odom_callback(msg, id), 10)
        self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 10)

        # Plans as soon as there is a goal and the centroid is known
        self.timer = self.create_timer(0.5, self.timer_callback)

    def odom_callback(self, msg: Odometry, robot_id: int):
        self.robot_poses[robot_id] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

    def goal_callback(self, msg: PoseStamped):
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            self.get_logger().warn(f'Goal in frame "{msg.header.frame_id}", assuming "{self.map_frame}"')
        self.pending_goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.timer_callback()

    def centroid(self):
        '''Mean position of the flock, None until every robot has been seen'''
        if any(p is None for p in self.robot_poses): return None
        return np.mean(self.robot_poses, axis=0)

    def timer_callback(self):
        if self.pending_goal is None: return
        start = self.centroid()
        if start is None:
            self.get_logger().info('Waiting for the odometry of every robot to plan', throttle_duration_sec=5.0)
            return
        goal, self.pending_goal = self.pending_goal, None
        self.plan(start, goal)

    def plan(self, centroid: np.ndarray, goal: np.ndarray):
        '''Plan the centroid path with RRT-Connect and publish it'''
        # The centroid of the flock (or the clicked goal) can be inside an inflated obstacle
        start = self.grid.nearest_free(centroid)
        free_goal = self.grid.nearest_free(goal)
        if start is None or free_goal is None:
            self.get_logger().error('The map has no free cells')
            return
        if not self.grid.in_bounds(goal) or np.linalg.norm(free_goal - goal) > 1e-9:
            self.get_logger().warn(f'Goal ({goal[0]:.2f}, {goal[1]:.2f}) is not free, '
                                   f'using the closest free point ({free_goal[0]:.2f}, {free_goal[1]:.2f})')

        t0 = time.perf_counter()
        for attempt in range(1, self.rrt_attempts + 1):
            path = self.rrt.plan(start, free_goal)
            if path is not None: break
        dt = time.perf_counter() - t0
        if path is None:
            self.get_logger().error(f'RRT-Connect found no path to ({free_goal[0]:.2f}, {free_goal[1]:.2f}) '
                                    f'after {self.rrt_attempts} attempts ({dt:.2f} s)')
            return

        n_raw = len(path)
        path = RRT_Connect.densify(self.rrt.shortcut(path), self.waypoint_spacing)
        if np.linalg.norm(start - centroid) > 1e-9: path.insert(0, centroid)  # path starts at the real centroid

        length = sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:]))
        self.get_logger().info(f'Path to ({free_goal[0]:.2f}, {free_goal[1]:.2f}): {length:.2f} m, {len(path)} waypoints '
                               f'({n_raw} RRT nodes, attempt {attempt}, {1e3 * dt:.0f} ms)')
        self.path_pub.publish(self.path_msg(path))

    def grid_msg(self, data: list) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info.map_load_time = msg.header.stamp
        msg.info.resolution = self.grid.resolution
        msg.info.width = self.grid.width
        msg.info.height = self.grid.height
        msg.info.origin.position.x = float(self.grid.origin[0])
        msg.info.origin.position.y = float(self.grid.origin[1])
        msg.info.origin.orientation.w = 1.0
        msg.data = data
        return msg

    def path_msg(self, path: list) -> Path:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        yaw = 0.0
        for k, p in enumerate(path):
            if k + 1 < len(path): yaw = atan2(path[k + 1][1] - p[1], path[k + 1][0] - p[0])  # heading of the next segment
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(p[0])
            pose.pose.position.y = float(p[1])
            pose.pose.orientation.z = sin(yaw / 2)
            pose.pose.orientation.w = cos(yaw / 2)
            msg.poses.append(pose)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = FlockPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
