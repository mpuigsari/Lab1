import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry
import numpy as np
from math import cos, sin, atan2

class ReynoldsNode(Node):

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

        # Get parameter values
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        self.number_robots = self.get_parameter('number_robots').get_parameter_value().integer_value
        self.ca = self.get_parameter('ca').get_parameter_value().double_value
        self.cc = self.get_parameter('cc').get_parameter_value().double_value
        self.cs = self.get_parameter('cs').get_parameter_value().double_value
        self.dt = 1/self.get_parameter('f').get_parameter_value().double_value
        self.max_range = self.get_parameter('max_range').get_parameter_value().double_value
        self.fov = np.deg2rad(self.get_parameter('fov').get_parameter_value().double_value)/2
        self.brake_acc = abs(self.get_parameter('brake_acc').get_parameter_value().double_value)  # the sign is applied in the control law
        self.max_acc = abs(self.get_parameter('max_acc').get_parameter_value().double_value)

        # Initialize variables 
        self.robot_pose = None  # x, y, theta
        self.robot_vel = None  # linear_x, linear_y, angular_z
        self.neigh_poses = [None for _ in range(self.number_robots)]  # List of neighbor poses
        self.neigh_vels = [None for _ in range(self.number_robots)]  # List of neighbor velocities

        # Subscribers
        self.create_subscription(Odometry, f'robot_{self.robot_id}/odom', self.odom_callback, 10)
        for i in range(self.number_robots):
            if i == self.robot_id: continue  # Skip subscribing to its own odometry
            self.create_subscription(Odometry, f'robot_{i}/odom', lambda msg, id=i: self.neigh_odom_callback(msg, id), 10)

        # Publisher for the velocity cmd
        self.publisher_ = self.create_publisher(Twist, f'robot_{self.robot_id}/cmd_vel', 10)

        # Publishers for the acceleration of each rule and the total (for plotting)
        self.accel_pubs = {
            name: self.create_publisher(Vector3Stamped, f'robot_{self.robot_id}/reynolds/{name}', 10)
            for name in ('alignment', 'cohesion', 'separation', 'total')}

        # Main timer only if there are more than 1 robot
        if self.number_robots > 1: 
            self.timer = self.create_timer(self.dt, self.timer_callback)
        else: 
            self.get_logger().warn('No neighbors detected. Skipping Reynolds calculation.')

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
        # self.robot_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])

    def timer_callback(self):
        '''Control loop that calculates the desired acceleration for robot i based on its neighbors following Reynolds Rules'''
        # No odometry from robot i arrived yet
        if self.robot_pose is None or self.robot_vel is None: return 

        sum_a = 0 # alignment term
        sum_c = 0 # cohesion term
        sum_s = 0 # separation term
        
        visible_robots = 0
        for i in range(self.number_robots):
            if i == self.robot_id: continue  # Skip its own data

            # No valid odometry for the neighbor i
            if self.neigh_poses[i] is None: continue 
            if self.neigh_vels[i] is None: continue

            # Calculate reynolds components
            diff_x = self.neigh_poses[i] - self.robot_pose
            norm_x = np.linalg.norm(diff_x)
            theta = self.wrap(atan2(diff_x[1], diff_x[0]) - atan2(self.robot_vel[1], self.robot_vel[0]))

            if norm_x > self.max_range or abs(theta) > self.fov: continue
            visible_robots +=1

            sum_a += self.neigh_vels[i] - self.robot_vel
            sum_c += diff_x
            sum_s -= diff_x / (norm_x + 1e-6)**2  # Avoid division by zero

        # no robots visible: break until the robot stops
        if visible_robots == 0:
            speed = np.linalg.norm(self.robot_vel)
            if speed < 1e-4: x_dotdot = np.zeros(2)  # already stopped
            # Opposite to the motion, at most the speed left: the last step lands on zero, never reverses
            else: x_dotdot = -min(self.brake_acc, speed / self.dt) * self.robot_vel / speed
            rules = {'alignment': np.zeros(2), 'cohesion': np.zeros(2), 'separation': np.zeros(2)}

        # at least one visible robot
        else: 
            # Calculate the acceleration of each Reynolds rule, normalized by the number of neighbors
            a_align = self.ca * sum_a / visible_robots 
            a_coh = self.cc * sum_c / visible_robots   
            a_sep = self.cs * sum_s / visible_robots   
            x_dotdot = a_align + a_coh + a_sep
            rules = {'alignment': a_align, 'cohesion': a_coh, 'separation': a_sep}

        # Acceleration limit
        acc = np.linalg.norm(x_dotdot)
        if acc > self.max_acc: x_dotdot = x_dotdot * self.max_acc / acc

        # total: the limited acceleration actually applied
        self.publish_accels({**rules, 'total': x_dotdot})  

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

    @staticmethod
    def quaternion_to_yaw(q):
        '''Convert quaternion to yaw angle'''
        return 2.0 * atan2(q.z, q.w)
    
    @staticmethod
    def wrap(theta): 
        return atan2(sin(theta), cos(theta))

def main(args = None):
    rclpy.init(args=args)
    node = ReynoldsNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()