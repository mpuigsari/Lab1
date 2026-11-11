'''
RRT-Connect path planner on a GridMap (no ROS dependency).

Kuffner & LaValle, "RRT-Connect: An Efficient Approach to Single-Query Path Planning", ICRA 2000.
Two trees grow from start and goal. Each iteration one tree EXTENDs one step towards a random sample,
then the other tree greedily CONNECTs towards the new node. The trees swap roles every iteration.
'''
import numpy as np

TRAPPED, ADVANCED, REACHED = 0, 1, 2


class _Tree():
    '''Tree stored as a growable array of nodes plus the parent index of each node'''
    def __init__(self, root: np.ndarray, capacity: int = 1024):
        self.nodes = np.empty((capacity, 2))
        self.parents = np.empty(capacity, dtype=int)
        self.nodes[0] = root
        self.parents[0] = -1
        self.n = 1

    def nearest(self, q: np.ndarray) -> int:
        return int(np.argmin(np.sum((self.nodes[:self.n] - q) ** 2, axis=1)))

    def add(self, q: np.ndarray, parent: int) -> int:
        if self.n == len(self.nodes):  # double the capacity
            self.nodes = np.vstack((self.nodes, np.empty_like(self.nodes)))
            self.parents = np.concatenate((self.parents, np.empty_like(self.parents)))
        self.nodes[self.n] = q
        self.parents[self.n] = parent
        self.n += 1
        return self.n - 1

    def branch(self, i: int) -> list:
        '''Nodes from node i back to the root'''
        path = []
        while i != -1:
            path.append(self.nodes[i].copy())
            i = self.parents[i]
        return path

    def edges(self) -> list:
        '''List of (parent, child) node pairs, for visualization'''
        return [(self.nodes[self.parents[i]], self.nodes[i]) for i in range(1, self.n)]


class RRT_Connect():
    def __init__(self, grid_map, step_size: float = 0.25, max_iter: int = 5000, seed=None):
        self.map = grid_map
        self.step_size = step_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = grid_map.bounds()
        self.trees = None  # (start tree, goal tree) of the last query, for visualization
        self.iterations = 0  # iterations used by the last query

    def plan(self, start, goal):
        '''Collision-free path [start, ..., goal] as a list of np.array([x, y]), or None if not found'''
        start = np.asarray(start, dtype=float)[:2]
        goal = np.asarray(goal, dtype=float)[:2]
        self.trees, self.iterations = None, 0
        if not self.map.is_free(start) or not self.map.is_free(goal): return None
        if self.map.segment_free(start, goal): return [start, goal]

        t_start, t_goal = _Tree(start), _Tree(goal)
        self.trees = (t_start, t_goal)
        ta, tb = t_start, t_goal

        for it in range(self.max_iter):
            self.iterations = it + 1
            q_rand = self.rng.uniform(self.lo, self.hi)

            status, i_new = self._extend(ta, q_rand)
            if status != TRAPPED:
                status, i_b = self._connect(tb, ta.nodes[i_new])
                if status == REACHED:
                    # ta: root -> q_new, tb: q_new (same point) -> root
                    path = ta.branch(i_new)[::-1] + tb.branch(i_b)[1:]
                    return path if ta is t_start else path[::-1]

            ta, tb = tb, ta  # swap the trees

        return None

    def _extend(self, tree: _Tree, q: np.ndarray) -> tuple:
        '''Grow the tree one step from its nearest node towards q -> (status, index of the new node)'''
        i_near = tree.nearest(q)
        q_near = tree.nodes[i_near]
        diff = q - q_near
        dist = np.linalg.norm(diff)
        if dist < 1e-9: return REACHED, i_near  # q is already in the tree

        if dist <= self.step_size: q_new, status = q.copy(), REACHED
        else: q_new, status = q_near + diff / dist * self.step_size, ADVANCED

        if not self.map.segment_free(q_near, q_new): return TRAPPED, -1
        return status, tree.add(q_new, i_near)

    def _connect(self, tree: _Tree, q: np.ndarray) -> tuple:
        '''Extend the tree towards q until it is reached or an obstacle is hit'''
        while True:
            status, i = self._extend(tree, q)
            if status != ADVANCED: return status, i

    def shortcut(self, path: list) -> list:
        '''Greedy line-of-sight pruning: from each kept node, jump to the farthest visible node'''
        if path is None or len(path) < 3: return path
        pruned, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.map.segment_free(path[i], path[j]): j -= 1
            pruned.append(path[j])
            i = j
        return pruned

    @staticmethod
    def densify(path: list, spacing: float) -> list:
        '''Resample every segment so consecutive waypoints are at most `spacing` apart'''
        if path is None or len(path) < 2 or spacing <= 0: return path
        out = [np.asarray(path[0], dtype=float)]
        for a, b in zip(path[:-1], path[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
            out.extend(a + (b - a) * k / n for k in range(1, n + 1))
        return out
