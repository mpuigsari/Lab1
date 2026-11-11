import numpy as np
from scipy import ndimage
from nav_msgs.msg import OccupancyGrid

FREE = 0  # nav_msgs/OccupancyGrid: 0 free, 100 occupied, -1 unknown

class ObstacleAvoidance():
    '''
    Potential field obstacle avoidance on an occupancy grid.

    Repulsive potential (Khatib):  U(r) = 0.5 * co * (1/r - 1/r0)^2   if r < r0,  0 otherwise
    Acceleration = -grad U         a(r) = co * (1/r - 1/r0) / r^2 * n   (n: unit vector obstacle -> robot)

    r is the distance from the robot to the closest obstacle point and r0 the influence distance.
    The acceleration is 0 at r0 (continuous) and grows as 1/r^3 near the obstacle.
    '''
    def __init__(self, msg: OccupancyGrid, co: float, r0: float):
        self.co = co     # Scaling constant, tuning
        self.r0 = r0     # Influence distance in which the potential field starts to affect

        # OccupancyGrid -> matrix grid[row, col]: data is row-major, row 0 at y = origin_y, col 0 at x = origin_x
        self.resolution = msg.info.resolution
        grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)

        # Unknown cells are obstacles. 
        # Obstacles are inflated by one cell to prevent being too close to them
        self.blocked = np.pad(grid != FREE, 1, constant_values=True)
        self.origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y]) - self.resolution
        self.height, self.width = self.blocked.shape

        # For every cell, the (row, col) of the closest obstacle cell. Computed once, then lookups are O(1)
        _, self.closest_idx = ndimage.distance_transform_edt(~self.blocked, return_indices=True)

    def world_to_cell(self, p: np.ndarray) -> tuple:
        '''World [x, y] -> (row, col), clamped to the map'''
        col, row = np.floor((p - self.origin) / self.resolution).astype(int)
        return min(max(row, 0), self.height - 1), min(max(col, 0), self.width - 1)

    def closest_obstacle(self, p: np.ndarray) -> np.ndarray:
        '''World [x, y] of the closest obstacle point to p'''
        row, col = self.world_to_cell(p)
        o_row, o_col = self.closest_idx[0, row, col], self.closest_idx[1, row, col]

        # The distance transform is measured between cell centers: the closest obstacle surface can belong
        # to a neighbor of that cell. Take the closest point of the obstacle cells (squares) in its 3x3 block
        rows, cols = np.mgrid[max(o_row - 1, 0):min(o_row + 2, self.height),
                              max(o_col - 1, 0):min(o_col + 2, self.width)]
        mask = self.blocked[rows, cols]
        lo = self.origin + np.stack([cols[mask], rows[mask]], axis=1) * self.resolution
        candidates = np.clip(p, lo, lo + self.resolution)
        return candidates[np.argmin(np.linalg.norm(candidates - p, axis=1))]

    def solve_obstacle(self, robot_x: np.ndarray) -> np.ndarray:
        '''Repulsive acceleration [ax, ay] away from the closest obstacle, 0 if it is farther than r0'''
        diff = robot_x - self.closest_obstacle(robot_x)  # obstacle -> robot
        r = np.linalg.norm(diff)

        if r >= self.r0: return np.zeros(2)
        if r < 1e-6: return np.zeros(2)  # robot center inside an obstacle cell: direction undefined

        return self.co * (1/r - 1/self.r0) / r**2 * diff / r