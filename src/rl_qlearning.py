"""Small, genuine tabular Q-learning baseline for one fixed drone scenario.

The existing 100 generated maps benchmark trajectory optimization. This agent
learns a policy by interacting with *one* discretized map over many episodes;
its Q-table does not claim generalization to unseen obstacle layouts.
"""

from collections import deque

import numpy as np

from src.environment import Environment2D


class GridDroneEnvironment:
    """Discretized 2D map with exact buffered segment collision checking."""

    ACTIONS = np.array([
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ], dtype=int)

    def __init__(self, scenario, grid_step=0.5, max_steps=160):
        self.original = Environment2D(scenario)
        self.grid_step = float(grid_step)
        self.max_steps = int(max_steps)
        extent = self.original.x_max - self.original.x_min
        cells = round(extent / self.grid_step)
        if cells <= 0 or not np.isclose(cells * self.grid_step, extent):
            raise ValueError("grid_step must divide the 10 m map extent")
        self.size = cells + 1
        self.coordinates = np.linspace(self.original.x_min, self.original.x_max, self.size)
        self.free = np.zeros((self.size, self.size), dtype=bool)
        self.neighbors = np.full((self.size, self.size, len(self.ACTIONS), 2), -1, dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                self.free[i, j] = self.point_safe(self.point((i, j)))
        for i in range(self.size):
            for j in range(self.size):
                if not self.free[i, j]:
                    continue
                for a, (di, dj) in enumerate(self.ACTIONS):
                    ni, nj = i + di, j + dj
                    if (0 <= ni < self.size and 0 <= nj < self.size
                            and self.free[ni, nj]
                            and self.segment_safe(self.point((i, j)), self.point((ni, nj)))):
                        self.neighbors[i, j, a] = (ni, nj)
        self.start = self._nearest_connected_node(self.original.get_start())
        self.goal = self._nearest_connected_node(self.original.get_goal())
        if not self._reachable(self.start, self.goal):
            raise ValueError("No route exists on this grid; reduce grid_step")
        self.goal_xy = self.point(self.goal)

    def point(self, state):
        return np.array([self.coordinates[state[0]], self.coordinates[state[1]]], dtype=float)

    def point_safe(self, point):
        return all(
            np.linalg.norm(point - np.asarray(obstacle["center"], dtype=float))
            >= self.original.effective_obstacle_radius(obstacle)
            for obstacle in self.original.get_obstacles()
        )

    def segment_safe(self, start, end):
        direction = end - start
        squared = float(np.dot(direction, direction))
        for obstacle in self.original.get_obstacles():
            center = np.asarray(obstacle["center"], dtype=float)
            t = np.clip(np.dot(center - start, direction) / squared, 0, 1) if squared else 0.0
            nearest = start + t * direction
            if np.linalg.norm(center - nearest) < self.original.effective_obstacle_radius(obstacle):
                return False
        return True

    def _nearest_connected_node(self, original_point):
        nodes = np.argwhere(self.free)
        nodes = sorted(nodes, key=lambda cell: np.linalg.norm(self.point(cell) - original_point))
        for cell in nodes:
            if self.segment_safe(original_point, self.point(cell)):
                return tuple(int(v) for v in cell)
        raise ValueError("No grid node can connect safely to an endpoint")

    def _reachable(self, start, goal):
        seen = {start}
        queue = deque([start])
        while queue:
            position = queue.popleft()
            if position == goal:
                return True
            for next_point in self.neighbors[position[0], position[1]]:
                node = tuple(int(v) for v in next_point)
                if node[0] >= 0 and node not in seen:
                    seen.add(node)
                    queue.append(node)
        return False

    def transition(self, state, action):
        neighbor = self.neighbors[state[0], state[1], action]
        if neighbor[0] < 0:
            return state, -4.0, False, True
        next_state = tuple(int(v) for v in neighbor)
        before = np.linalg.norm(self.point(state) - self.goal_xy)
        after = np.linalg.norm(self.point(next_state) - self.goal_xy)
        reward = 2.0 * (before - after) - 0.10
        success = next_state == self.goal
        if success:
            reward += 60.0
        return next_state, float(reward), success, False

    def continuous_path(self, states):
        points = [np.asarray(self.original.get_start(), dtype=float)]
        points.extend(self.point(state) for state in states)
        points.append(np.asarray(self.original.get_goal(), dtype=float))
        path = [points[0]]
        for p in points[1:]:
            if not np.allclose(p, path[-1]):
                path.append(p)
        return np.asarray(path)


class QLearningAgent:
    def __init__(self, environment, learning_rate=0.25, discount=0.97, seed=2026):
        self.environment = environment
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.rng = np.random.default_rng(seed)
        self.q_values = np.zeros((environment.size, environment.size, len(environment.ACTIONS)))

    def action(self, state, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(len(self.environment.ACTIONS)))
        values = self.q_values[state[0], state[1]]
        candidates = np.flatnonzero(np.isclose(values, np.max(values)))
        return int(self.rng.choice(candidates))

    def train(self, episodes=3000):
        history = []
        for episode in range(int(episodes)):
            state = self.environment.start
            total_reward = 0.0
            success = False
            epsilon = max(0.03, 0.95 * (0.997 ** episode))
            for step in range(self.environment.max_steps):
                action = self.action(state, epsilon)
                next_state, reward, success, _ = self.environment.transition(state, action)
                future = 0.0 if success else np.max(self.q_values[next_state[0], next_state[1]])
                target = reward + self.discount * future
                index = (state[0], state[1], action)
                self.q_values[index] += self.learning_rate * (target - self.q_values[index])
                state = next_state
                total_reward += reward
                if success:
                    break
            history.append({
                "episode": episode + 1,
                "reward": float(total_reward),
                "success": bool(success),
                "steps": step + 1,
                "epsilon": float(epsilon),
            })
        return history

    def greedy_rollout(self):
        state = self.environment.start
        states = [state]
        for _ in range(self.environment.max_steps):
            action = int(np.argmax(self.q_values[state[0], state[1]]))
            next_state, _, reached_goal, invalid = self.environment.transition(state, action)
            if invalid:
                return states, False
            states.append(next_state)
            state = next_state
            if reached_goal:
                return states, True
        return states, False
