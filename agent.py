import random
import heapq
import math
from collections import deque


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('wall_ahead', False):
            return random.choice(['Left', 'Right', 'Down', 'Up'])
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """
    A Simple Reflex Agent that acts based purely on immediate percepts using IF-THEN rules.
    It does not store history or maintain internal state across time steps.
    """

    def sense_and_act(self, percept: dict) -> str:
        # Condition-Action (IF-THEN) Rules
        if percept.get('food_here', False):
            return 'Up'  # Collect food / advance
        elif percept.get('wall_ahead', False):
            return 'Left'  # Turn/change direction when facing a wall
        else:
            return 'Up'  # Default move forward


class ModelBasedAgent:
    """
    A Model-Based Agent that maintains internal memory state (visited cells, position tracker,
    and action/percept history) to update its world model and avoid repeating failed actions.
    """

    def __init__(self):
        # 1. Internal Memory State
        self.visited_cells = set([(0, 0)])  # Set of visited relative coordinate tuples
        self.current_pos = [0, 0]           # Estimated internal position tracker (x, y)
        self.last_action = None             # Last action taken
        self.percept_history = []           # History of past percepts

    def sense_and_act(self, percept: dict) -> str:
        # 2. Update Internal State (Transition & Sensor Model)
        # Update estimated position based on last action executed
        if self.last_action == 'Up':
            self.current_pos[1] += 1
        elif self.last_action == 'Down':
            self.current_pos[1] -= 1
        elif self.last_action == 'Left':
            self.current_pos[0] -= 1
        elif self.last_action == 'Right':
            self.current_pos[0] += 1

        # Record position and percept into internal memory
        self.visited_cells.add(tuple(self.current_pos))
        self.percept_history.append(percept)

        # 3. Condition-Action Rules Querying Internal Memory
        wall_ahead = percept.get('wall_ahead', False)
        food_here = percept.get('food_here', False)

        if food_here:
            action = 'Up'
        elif wall_ahead:
            # Query memory: IF last action facing a wall was 'Left', choose 'Right' to break loop
            if self.last_action == 'Left':
                action = 'Right'
            else:
                action = 'Left'
        else:
            action = 'Up'

        # Record action in state for next transition step
        self.last_action = action
        return action


class SearchAgent:
    """
    A problem-solving agent that uses uninformed search strategies
    (BFS, DFS, UCS) and informed search (A*) to find a path from
    a start position to a goal.
    All algorithms maintain a 'reached' (visited) set to convert
    Tree Search into Graph Search, preventing infinite loops.
    """

    def __init__(self, algorithm='BFS'):
        self.plan = []                  # Stored sequence of actions to execute
        self.active_algo = algorithm    # Which search to use: 'BFS', 'DFS', 'UCS', or 'AStar'

    def sense_and_act(self, percept: dict) -> str:
        # If the current plan is empty, compute a new one
        if not self.plan:
            agent_pos = tuple(percept['agent_pos'])
            walls = [tuple(w) for w in percept['walls']]
            grid_size = percept['grid_size']
            all_food = [tuple(f) for f in percept['all_food']]

            if not all_food:
                return random.choice(['Up', 'Down', 'Left', 'Right'])

            # Find the closest food pellet (Manhattan distance)
            closest_food = min(
                all_food,
                key=lambda f: abs(f[0] - agent_pos[0]) + abs(f[1] - agent_pos[1])
            )

            # Execute the search algorithm matching self.active_algo
            if self.active_algo == 'BFS':
                path = self.bfs_search(agent_pos, closest_food, walls, grid_size)
            elif self.active_algo == 'DFS':
                path = self.dfs_search(agent_pos, closest_food, walls, grid_size)
            elif self.active_algo == 'UCS':
                path = self.ucs_search(agent_pos, closest_food, walls, grid_size)
            elif self.active_algo == 'AStar':
                path = self.astar_search(agent_pos, closest_food, walls, grid_size)
            else:
                path = self.bfs_search(agent_pos, closest_food, walls, grid_size)

            # Store the resulting action sequence in self.plan
            if path:
                self.plan = path
            else:
                return random.choice(['Up', 'Down', 'Left', 'Right'])

        # Return the first action from the plan
        return self.plan.pop(0)


    # ------------------------------------------------------------------
    # BFS – Breadth-First Search (FIFO Queue)
    # Explores the shallowest (closest) nodes first.
    # ------------------------------------------------------------------
    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        width, height = grid_size
        wall_set = set(walls)

        # FIFO queue: each entry is (current_position, path_of_actions)
        queue = deque([(start_pos, [])])

        # Reached set – tracks explored states (Graph Search)
        visited = {start_pos}

        moves = [('Up', (0, 1)), ('Down', (0, -1)),
                 ('Left', (-1, 0)), ('Right', (1, 0))]

        while queue:
            (curr_x, curr_y), path = queue.popleft()  # FIFO: popleft()

            # Goal test
            if (curr_x, curr_y) == goal_pos:
                return path

            # Expand current node
            for action, (dx, dy) in moves:
                nx, ny = curr_x + dx, curr_y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in wall_set:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [action]))

        return None  # No path found

    # ------------------------------------------------------------------
    # DFS – Depth-First Search (LIFO Stack)
    # Explores the deepest nodes first.
    # ------------------------------------------------------------------
    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        width, height = grid_size
        wall_set = set(walls)

        # LIFO stack: each entry is (current_position, path_of_actions)
        stack = [(start_pos, [])]

        # Reached set – tracks explored states (Graph Search)
        visited = {start_pos}

        moves = [('Up', (0, 1)), ('Down', (0, -1)),
                 ('Left', (-1, 0)), ('Right', (1, 0))]

        while stack:
            (curr_x, curr_y), path = stack.pop()  # LIFO: pop() from end

            # Goal test
            if (curr_x, curr_y) == goal_pos:
                return path

            # Expand current node
            for action, (dx, dy) in moves:
                nx, ny = curr_x + dx, curr_y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in wall_set:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        stack.append(((nx, ny), path + [action]))

        return None  # No path found

    # ------------------------------------------------------------------
    # UCS – Uniform-Cost Search (Priority Queue ordered by g(n))
    # Explores the node with the lowest total path cost first.
    # ------------------------------------------------------------------
    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        width, height = grid_size
        wall_set = set(walls)

        # Priority queue: each entry is (cumulative_cost, current_position, path_of_actions)
        # heapq is a min-heap, so the lowest-cost node is popped first
        frontier = [(0, start_pos, [])]

        # Reached set – tracks explored states (Graph Search)
        visited = set()

        moves = [('Up', (0, 1)), ('Down', (0, -1)),
                 ('Left', (-1, 0)), ('Right', (1, 0))]

        while frontier:
            cost, (curr_x, curr_y), path = heapq.heappop(frontier)  # Lowest g(n)

            # Skip if already visited (a cheaper path was already processed)
            if (curr_x, curr_y) in visited:
                continue
            visited.add((curr_x, curr_y))

            # Goal test
            if (curr_x, curr_y) == goal_pos:
                return path

            # Expand current node
            for action, (dx, dy) in moves:
                nx, ny = curr_x + dx, curr_y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in wall_set:
                    if (nx, ny) not in visited:
                        step_cost = 1  # Each move costs 1 (uniform)
                        heapq.heappush(frontier, (cost + step_cost, (nx, ny), path + [action]))

        return None  # No path found

    # ------------------------------------------------------------------
    # Heuristic: Manhattan Distance
    # h(n) = |x1 - x2| + |y1 - y2|
    # ------------------------------------------------------------------
    def manhattan_distance(self, pos, goal):
        """Calculate Manhattan distance between two positions."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # ------------------------------------------------------------------
    # Heuristic: Euclidean Distance
    # h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)
    # ------------------------------------------------------------------
    def euclidean_distance(self, pos, goal):
        """Calculate Euclidean (straight-line) distance between two positions."""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    # ------------------------------------------------------------------
    # A* Search (Priority Queue ordered by f(n) = g(n) + h(n))
    # Combines path cost g(n) with heuristic estimate h(n) for
    # optimal informed expansion. Supports switchable heuristics.
    # ------------------------------------------------------------------
    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        width, height = grid_size
        wall_set = set(walls)

        # Select the heuristic function based on heuristic_type
        if heuristic_type == 'euclidean':
            heuristic = self.euclidean_distance
        else:
            heuristic = self.manhattan_distance

        # Priority queue: (f_cost, g_cost, current_pos, path_taken)
        # For the starting node: g(n) = 0, h(n) = heuristic(start, goal)
        h_start = heuristic(start_pos, goal_pos)
        frontier = [(h_start, 0, start_pos, [])]

        # Reached set – tracks explored states (Graph Search)
        reached_states = set()

        moves = [('Up', (0, 1)), ('Down', (0, -1)),
                 ('Left', (-1, 0)), ('Right', (1, 0))]

        while frontier:
            f_cost, g_cost, (curr_x, curr_y), path = heapq.heappop(frontier)

            # Goal test – return path if we've reached the goal
            if (curr_x, curr_y) == goal_pos:
                return path

            # Skip if already reached (a cheaper path was already processed)
            if (curr_x, curr_y) in reached_states:
                continue
            reached_states.add((curr_x, curr_y))

            # Expand current node – check four adjacent cells
            for action, (dx, dy) in moves:
                nx, ny = curr_x + dx, curr_y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in wall_set:
                    if (nx, ny) not in reached_states:
                        g_new = g_cost + 1              # g_new = g_current + 1
                        h_new = heuristic((nx, ny), goal_pos)  # h_new using chosen heuristic
                        f_new = g_new + h_new            # f_new = g_new + h_new
                        heapq.heappush(frontier, (f_new, g_new, (nx, ny), path + [action]))

        return None  # No path found


# ======================================================================
# Testing Checkpoint: Verify heuristic outputs
# ======================================================================
if __name__ == "__main__":
    agent = SearchAgent()
    start = (0, 0)
    goal = (3, 4)

    manhattan = agent.manhattan_distance(start, goal)
    euclidean = agent.euclidean_distance(start, goal)

    print(f"Manhattan Distance from {start} to {goal}: {manhattan}")   # Expected: 7
    print(f"Euclidean Distance from {start} to {goal}: {euclidean}")   # Expected: 5.0