'''
Obstacle avoidance rules on an occupancy grid.

    ObstacleMap     the grid, preprocessed once (distance transform) and shared by both rules
    PotentialField  repulsion from the closest obstacle point (Khatib), short range safety rule
    SteerToAvoid    look ahead along the heading and steer towards the closest free direction
                    (Reynolds steer-to-avoid), long range rule that also works head-on

Both rules are 0 outside their range and grow as the robot gets closer to the obstacle.
'''
import numpy as np
from scipy import ndimage

FREE = 0  # nav_msgs/OccupancyGrid: 0 free, 100 occupied, -1 unknown


def rotate(u: np.ndarray, angle: float) -> np.ndarray:
    '''Rotate the 2D vector u by angle [rad] (counterclockwise)'''
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])


class ObstacleMap():
    '''
    Occupancy grid with, for every cell, the distance to the closest obstacle cell and its index.

    grid[row, col]: row 0 at y = origin_y, col 0 at x = origin_x (nav_msgs/OccupancyGrid convention).
    Unknown cells are obstacles. A one cell wall is added around the map border, so the border acts as
    a wall (Stage floorplans use 'boundary 1').
    '''
    def __init__(self, grid: np.ndarray, resolution: float, origin):
        self.resolution = float(resolution)
        self.blocked = np.pad(np.asarray(grid) != FREE, 1, constant_values=True)
        self.origin = np.asarray(origin, dtype=float)[:2] - self.resolution  # the pad moves cell (0, 0) one cell down-left
        self.height, self.width = self.blocked.shape

        # For every cell, the distance [m] to the closest obstacle cell (between cell centers) and the
        # (row, col) of that cell. Computed once, then lookups are O(1)
        dist, self.closest_idx = ndimage.distance_transform_edt(~self.blocked, return_indices=True)
        self.dist = dist * self.resolution

    @classmethod
    def from_msg(cls, msg):
        '''Build from a nav_msgs/OccupancyGrid: data is row-major, row 0 at y = origin_y'''
        grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        return cls(grid, msg.info.resolution, [msg.info.origin.position.x, msg.info.origin.position.y])

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

    def clearance(self, p: np.ndarray) -> float:
        '''Approximate distance [m] from p to the closest obstacle surface (cell resolution)'''
        row, col = self.world_to_cell(p)
        return self.dist[row, col] - 0.5 * self.resolution

    def free_distance(self, p: np.ndarray, u: np.ndarray, length: float, radius: float) -> float:
        '''
        Distance a disc of the given radius can travel from p along the unit vector u before touching an
        obstacle, up to length. Sphere tracing: at every point the clearance is a step that cannot cross
        an obstacle, so open space is crossed in a few long steps.
        '''
        s = 0.0
        min_step = 0.5 * self.resolution
        while s < length:
            free = self.clearance(p + s * u) - radius
            if free <= 0.0: return s
            s += max(free, min_step)
        return length


class PotentialField():
    '''
    Repulsive potential (Khatib):  U(r) = 0.5 * co * (1/r - 1/r0)^2   if r < r0,  0 otherwise
    Acceleration = -grad U         a(r) = co * (1/r - 1/r0) / r^2 * n   (n: unit vector obstacle -> robot)

    r is the distance from the robot to the closest obstacle point and r0 the influence distance.
    The acceleration is 0 at r0 (continuous) and grows as 1/r^3 near the obstacle.
    '''
    def __init__(self, obstacle_map: ObstacleMap, co: float, r0: float):
        self.map = obstacle_map
        self.co = co     # Scaling constant, tuning
        self.r0 = r0     # Influence distance in which the potential field starts to affect

    def solve(self, robot_x: np.ndarray) -> np.ndarray:
        '''Repulsive acceleration [ax, ay] away from the closest obstacle, 0 if it is farther than r0'''
        if self.co == 0.0: return np.zeros(2)
        diff = robot_x - self.map.closest_obstacle(robot_x)  # obstacle -> robot
        r = np.linalg.norm(diff)

        if r >= self.r0: return np.zeros(2)
        if r < 1e-6: return np.zeros(2)  # robot center inside an obstacle cell: direction undefined

        return self.co * (1/r - 1/self.r0) / r**2 * diff / r


class SteerToAvoid():
    '''
    Steer-to-avoid (Reynolds 1987, 1999): look ahead along the heading and, if the way is blocked, steer
    towards the free direction closest to the heading (the closest silhouette edge of the obstacle).

        L        = look_min + look_time * |v|           look-ahead distance (longer when faster)
        d        = free distance along the heading      (disc of the given radius, sphere tracing)
        rule off if d >= L                              nothing ahead within the look-ahead
        u        = free direction closest to the heading, searched in a fan of +-angle_step ... +-max_angle
        urgency  = 1 - d / L                            0 at the look-ahead limit, 1 at contact
        a        = csa * urgency * (|v| u - v)          steering = desired velocity - current velocity

    Unlike a repulsive field, the acceleration is mostly lateral: a robot heading straight at a wall turns
    instead of stopping in front of it. At equal deviation, the side of the navigation velocity v_ref is
    preferred, so the robots of a flock tend to go around an obstacle on the same side.
    '''
    MIN_SPEED = 0.02  # [m/s] below this the heading is undefined: the rule is off

    def __init__(self, obstacle_map: ObstacleMap, csa: float, look_time: float, look_min: float,
                 radius: float, angle_step: float, max_angle: float):
        self.map = obstacle_map
        self.csa = csa                  # weight
        self.look_time = look_time      # [s] look-ahead time
        self.look_min = look_min        # [m] look-ahead distance at zero speed
        self.radius = radius            # [m] half width of the corridor that must be free (robot radius + margin)
        self.angle_step = angle_step    # [rad] angular resolution of the fan of feelers
        self.n_steps = max(1, int(round(max_angle / angle_step)))  # feelers on each side

    def solve(self, robot_x: np.ndarray, robot_vel: np.ndarray, v_ref=None) -> tuple:
        '''
        Steering acceleration [ax, ay] and a debug dict for visualization, None when the rule is off:
            heading, length (look-ahead L), hit (free distance d), direction (chosen free direction or None)
        '''
        speed = np.linalg.norm(robot_vel)
        if self.csa == 0.0 or speed < self.MIN_SPEED: return np.zeros(2), None

        h = robot_vel / speed
        length = self.look_min + self.look_time * speed
        # A robot already closer to an obstacle than the radius still has to find the directions that
        # follow the wall or leave it: shrink the corridor to half its current clearance
        radius = max(0.0, min(self.radius, 0.5 * self.map.clearance(robot_x)))

        d = self.map.free_distance(robot_x, h, length, radius)
        debug = {'heading': h, 'length': length, 'hit': d, 'direction': None}
        if d >= length: return np.zeros(2), debug

        # Preferred side: the one of the navigation velocity, else the one with more room
        side = 0.0
        if v_ref is not None and np.linalg.norm(v_ref) > self.MIN_SPEED:
            cross = h[0] * v_ref[1] - h[1] * v_ref[0]
            if abs(cross) > 1e-3 * np.linalg.norm(v_ref): side = np.sign(cross)
        if side == 0.0:
            left = self.map.free_distance(robot_x, rotate(h, self.angle_step), length, radius)
            right = self.map.free_distance(robot_x, rotate(h, -self.angle_step), length, radius)
            side = 1.0 if left >= right else -1.0

        # Fan of feelers, closest to the heading first: the first one free for the whole look-ahead wins.
        # If none is, the most open one; if no feeler does better than the heading, brake
        best, best_d = None, d
        for k in range(1, self.n_steps + 1):
            for s in (side, -side):
                u = rotate(h, s * k * self.angle_step)
                free = self.map.free_distance(robot_x, u, length, radius)
                if free >= length:
                    best, best_d = u, free
                    break
                if free > best_d: best, best_d = u, free
            if best_d >= length: break

        urgency = 1.0 - d / length
        v_des = speed * best if best is not None else np.zeros(2)
        debug['direction'] = best
        return self.csa * urgency * (v_des - robot_vel), debug
