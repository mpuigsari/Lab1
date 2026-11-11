'''
2D occupancy grid map built from a map_server-style yaml + image (no ROS dependency).

Cell convention (same as nav_msgs/OccupancyGrid):
    grid[row, col], row 0 is the bottom of the map (y = origin_y), col 0 is x = origin_x
    values: 0 free, 100 occupied, -1 unknown
'''
import os

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

FREE, OCCUPIED, UNKNOWN = 0, 100, -1


class GridMap():
    def __init__(self, occupancy: np.ndarray, resolution: float, origin, inflation_radius: float = 0.0):
        self.occupancy = np.asarray(occupancy, dtype=np.int8)  # raw map (published)
        self.resolution = float(resolution)
        self.origin = np.array(origin[:2], dtype=float)  # world position of the bottom-left corner of cell (0, 0)
        self.height, self.width = self.occupancy.shape
        self.inflation_radius = float(inflation_radius)

        # Unknown cells are treated as obstacles
        blocked = self.occupancy != FREE

        # Distance [m] from every cell to the closest blocked cell -> inflated (configuration space) map
        clearance = ndimage.distance_transform_edt(~blocked) * self.resolution
        self.inflated = blocked | (clearance <= self.inflation_radius)

        # For every inflated cell, the indices of the closest free cell (used by nearest_free)
        if self.inflated.all():
            self._nearest_free_idx = None
        else:
            _, self._nearest_free_idx = ndimage.distance_transform_edt(self.inflated, return_indices=True)

    @classmethod
    def from_yaml(cls, yaml_path: str, inflation_radius: float = 0.0):
        '''Load a map_server map (yaml + image), trinary mode'''
        with open(yaml_path, 'r') as f:
            meta = yaml.safe_load(f)

        image_path = meta['image']
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image_path)

        # Occupancy probability of each pixel: dark = occupied (unless negate)
        pixels = np.asarray(Image.open(image_path).convert('L'), dtype=float) / 255.0
        occ = pixels if meta.get('negate', 0) else 1.0 - pixels

        grid = np.full(pixels.shape, UNKNOWN, dtype=np.int8)
        grid[occ > meta.get('occupied_thresh', 0.65)] = OCCUPIED
        grid[occ < meta.get('free_thresh', 0.196)] = FREE

        # Image row 0 is the top of the map, grid row 0 is the bottom
        grid = np.flipud(grid).copy()

        # Stage floorplans use 'boundary 1': the map border is a wall
        grid[0, :] = grid[-1, :] = OCCUPIED
        grid[:, 0] = grid[:, -1] = OCCUPIED

        return cls(grid, meta['resolution'], meta.get('origin', [0.0, 0.0, 0.0]), inflation_radius)

    # ---------------------------------------------------------------- coordinates
    def world_to_cell(self, p) -> tuple:
        '''World [x, y] -> (row, col). Can be out of bounds'''
        col, row = np.floor((np.asarray(p, dtype=float)[:2] - self.origin) / self.resolution).astype(int)
        return int(row), int(col)

    def cell_to_world(self, row: int, col: int) -> np.ndarray:
        '''(row, col) -> world [x, y] of the cell center'''
        return self.origin + (np.array([col, row], dtype=float) + 0.5) * self.resolution

    def in_bounds(self, p) -> bool:
        row, col = self.world_to_cell(p)
        return 0 <= row < self.height and 0 <= col < self.width

    def bounds(self) -> tuple:
        '''(lower [x, y], upper [x, y]) world limits of the map'''
        lo = self.origin.copy()
        hi = self.origin + np.array([self.width, self.height]) * self.resolution
        return lo, hi

    # ---------------------------------------------------------------- collision checks (inflated map)
    def is_free(self, p) -> bool:
        row, col = self.world_to_cell(p)
        if not (0 <= row < self.height and 0 <= col < self.width): return False
        return not self.inflated[row, col]

    def segment_free(self, a, b) -> bool:
        '''True if the straight segment a -> b only crosses free cells of the inflated map.
        Exact traversal: every cell the segment passes through is checked, so any sub-segment
        of a free segment is free too (needed to resample paths safely).'''
        a = np.asarray(a, dtype=float)[:2]
        b = np.asarray(b, dtype=float)[:2]
        ga = (a - self.origin) / self.resolution  # endpoints in cell units
        gb = (b - self.origin) / self.resolution

        # Parameters t in [0, 1] where the segment crosses a vertical or horizontal grid line
        ts = [np.array([0.0, 1.0])]
        for axis in (0, 1):
            if abs(gb[axis] - ga[axis]) > 1e-12:
                lines = np.arange(np.ceil(min(ga[axis], gb[axis])), np.floor(max(ga[axis], gb[axis])) + 1)
                ts.append((lines - ga[axis]) / (gb[axis] - ga[axis]))
        ts = np.unique(np.clip(np.concatenate(ts), 0.0, 1.0))

        # The midpoint between two consecutive crossings lies inside exactly one traversed cell
        t_mid = np.concatenate(([0.0], 0.5 * (ts[:-1] + ts[1:]), [1.0]))
        cols, rows = np.floor(ga + np.outer(t_mid, gb - ga)).astype(int).T
        if (rows < 0).any() or (cols < 0).any() or (rows >= self.height).any() or (cols >= self.width).any():
            return False
        return not self.inflated[rows, cols].any()

    def nearest_free(self, p):
        '''p itself if free, otherwise the center of the closest free cell (None if the map has no free cell)'''
        p = np.asarray(p, dtype=float)[:2]
        if self.is_free(p): return p.copy()
        if self._nearest_free_idx is None: return None
        row, col = self.world_to_cell(p)
        row = min(max(row, 0), self.height - 1)
        col = min(max(col, 0), self.width - 1)
        return self.cell_to_world(self._nearest_free_idx[0, row, col], self._nearest_free_idx[1, row, col])

    # ---------------------------------------------------------------- ROS helpers
    def occupancy_data(self) -> list:
        '''Raw map as a row-major list, ready for OccupancyGrid.data'''
        return self.occupancy.ravel().tolist()

    def inflated_data(self) -> list:
        '''Inflated map (0 / 100) as a row-major list, ready for OccupancyGrid.data'''
        return np.where(self.inflated, OCCUPIED, FREE).astype(np.int8).ravel().tolist()
