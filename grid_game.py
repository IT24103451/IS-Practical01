# grid_game.py
import random


class GridHuntGame:
    """A small Pacman-style grid environment (4x4) where an agent collects food."""

    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]  # Starting position (x, y)
        self.agent_dir = 'Up'  # Facing direction ('Up', 'Down', 'Left', 'Right')

        # Place a few random food pellets, obstacles (walls), and toxic traps
        self.food_positions = {(1, 2), (2, 3), (3, 0), (2, 1)}
        self.walls = {(1, 1), (2, 2)}
        self.toxic_traps = {(0, 3), (3, 2)}

        self.score = 0
        self.steps = 0

    def get_percept(self, agent=None) -> dict:
        x, y = self.agent_pos
        if self.agent_dir == 'Up':
            front_pos = (x, y + 1)
        elif self.agent_dir == 'Down':
            front_pos = (x, y - 1)
        elif self.agent_dir == 'Left':
            front_pos = (x - 1, y)
        elif self.agent_dir == 'Right':
            front_pos = (x + 1, y)
        else:
            front_pos = (x, y + 1)

        out_of_bounds = (
            front_pos[0] < 0 or front_pos[0] >= self.width or
            front_pos[1] < 0 or front_pos[1] >= self.height
        )
        wall_ahead = out_of_bounds or (front_pos in self.walls)
        food_here = tuple(self.agent_pos) in self.food_positions

        return {
            'wall_ahead': wall_ahead,
            'food_here': food_here,
            'smells_food': food_here,
            'smells_toxin': tuple(self.agent_pos) in self.toxic_traps,
            'hit_wall': tuple(self.agent_pos) in self.walls,
            'score': self.score,
            'remaining_food': len(self.food_positions)
        }

    def execute_action(self, agent, action: str):
        self.steps += 1
        if action in ['Up', 'Down', 'Left', 'Right']:
            self.agent_dir = action
        new_pos = list(self.agent_pos)

        if action == 'Up':
            new_pos[1] = min(self.height - 1, new_pos[1] + 1)
        elif action == 'Down':
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 'Left':
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 'Right':
            new_pos[0] = min(self.width - 1, new_pos[0] + 1)

        # Check collision with walls
        if tuple(new_pos) in self.walls:
            self.score -= 5  # Penalty for hitting a wall
        else:
            self.agent_pos = new_pos

        # Check if eating food
        tuple_pos = tuple(self.agent_pos)
        if tuple_pos in self.food_positions:
            self.food_positions.remove(tuple_pos)
            self.score += 20  # Reward for eating food pellet

        # Check if stepped on toxic trap
        if tuple_pos in self.toxic_traps:
            self.score -= 15  # Severe penalty for stepping on toxic trap

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= 20