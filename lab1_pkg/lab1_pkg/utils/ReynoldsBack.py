import numpy as np
from math import atan2, sin, cos

def wrap(theta): 
    return atan2(sin(theta), cos(theta))

class ReynoldsRule(): 
    def __init__(self, ca: float, cc: float, cs: float, max_range: float, fov: float, id: int): 

        # Store the parameters
        self.id = id
        self.ca = ca
        self.cc = cc
        self.cs = cs
        self.max_range = max_range
        self.fov = fov

    def solve_reynolds(self, robot_x: np.ndarray, robot_vel: np.ndarray, 
                             robot_xs: list[np.ndarray], robot_vels: list[np.ndarray]
                             ) -> dict:
        '''
        Input: robot i position and velocities as a numpy array
               neighbors positions and velocities as a list of numpy arrays
        
        Calculate the reynolds rules acceleration for each one of the three components
        
        Returns a dictionary of three elements [alignment, cohesion, separation]
        Accelerations are zero if no robots are visible
        '''
        N = len(robot_xs)   # Number of robots in the simulation 
        visible_robots = 0  # Number of visible robots
        sum_a = 0
        sum_c = 0
        sum_s = 0
        
        # Calculate the component of each reynolds rule
        for i in range(N):
            if i == self.id: continue  # Skip its own data, its None

            # No valid odometry for the neighbor i
            if robot_xs[i] is None: continue 
            if robot_vels[i] is None: continue

            # Calculate reynolds components
            diff_x = robot_xs[i] - robot_x
            norm_x = np.linalg.norm(diff_x)
            theta = wrap(atan2(diff_x[1], diff_x[0]) - atan2(robot_vel[1], robot_vel[0]))

            # Check visibility conditions: rage and foc
            if norm_x > self.max_range or abs(theta) > self.fov: continue
            visible_robots += 1

            sum_a += robot_vels[i] - robot_vel
            sum_c += diff_x
            sum_s -= diff_x / (norm_x + 1e-6)**2  # Avoid division by zero

        # no robots visible, publish 0 acc
        if visible_robots == 0: 
            return {'alignment': np.zeros(2), 'cohesion': np.zeros(2), 'separation': np.zeros(2)}

        # At least one visible robot
        # Calculate the acceleration of each Reynolds rule, normalized by the number of neighbors
        
        # Alignment
        a_align = self.ca * sum_a / visible_robots 
        # Cohesion
        a_coh = self.cc * sum_c / visible_robots   
        # Separation
        a_sep = self.cs * sum_s / visible_robots   
        # Return each one of the components
        return {'alignment': a_align, 'cohesion': a_coh, 'separation': a_sep}
