import random
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
    """A problem-solving agent using Breadth-First Search (BFS)."""

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        width, height = grid_size
        wall_set = set(walls)
        queue = deque([(start_pos, [])])
        visited = {start_pos}

        moves = [('Up', (0, 1)), ('Down', (0, -1)), ('Left', (-1, 0)), ('Right', (1, 0))]

        while queue:
            (curr_x, curr_y), path = queue.popleft()

            if (curr_x, curr_y) == goal_pos:
                return path

            for action, (dx, dy) in moves:
                nx, ny = curr_x + dx, curr_y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in wall_set:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [action]))

        return None