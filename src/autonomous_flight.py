"""Q-learning flight controller guided by an independently optimized route.

This is a discrete, fixed-altitude *simulation*, not a physical flight stack.
The optimizer produces an expert reference route first; the Q-learner then
learns actions from reward signals, without copying that reference as a policy.
"""

import numpy as np

from src.rl_qlearning import GridDroneEnvironment


def distance_to_polyline(point, polyline):
    """Shortest Euclidean distance from a 2D point to a reference route."""
    point = np.asarray(point, dtype=float)
    route = np.asarray(polyline, dtype=float)
    best = float("inf")
    for start, end in zip(route[:-1], route[1:]):
        direction = end - start
        denominator = float(np.dot(direction, direction))
        fraction = (np.clip(np.dot(point - start, direction) / denominator, 0, 1)
                    if denominator > 0 else 0)
        best = min(best, float(np.linalg.norm(point - (start + fraction * direction))))
    return best


class GuidedGridDroneEnvironment(GridDroneEnvironment):
    """Learn goal-reaching actions while favoring a feasible optimized route.

    Actions are checked against *buffered* obstacles before they are accepted.
    Step penalties approximate distance, time and energy under the project's
    fixed-speed power model; these costs are proportional in that model.
    """

    def __init__(self, scenario, reference_path, grid_step=0.5, max_steps=160,
                 guidance_weight=0.32):
        super().__init__(scenario, grid_step=grid_step, max_steps=max_steps)
        self.reference_path = np.asarray(reference_path, dtype=float)
        if self.reference_path.ndim != 2 or self.reference_path.shape[1] != 2 or len(self.reference_path) < 2:
            raise ValueError("reference_path must have at least two XY points")
        if not np.all(np.isfinite(self.reference_path)):
            raise ValueError("reference_path contains nonfinite points")
        if not np.allclose(self.reference_path[0], self.original.get_start(), atol=1e-6):
            raise ValueError("reference_path must start at the drone's start point")
        if not np.allclose(self.reference_path[-1], self.original.get_goal(), atol=1e-6):
            raise ValueError("reference_path must end at the drone's goal point")
        if not all(self.segment_safe(a, b) for a, b in zip(self.reference_path[:-1], self.reference_path[1:])):
            raise ValueError("reference_path intersects buffered obstacles")
        self.guidance_weight = float(guidance_weight)
        self.reference_distances = np.empty((self.size, self.size), dtype=float)
        for i in range(self.size):
            for j in range(self.size):
                self.reference_distances[i, j] = distance_to_polyline(self.point((i, j)), self.reference_path)

    def transition(self, state, action):
        next_state, reward, success, invalid = super().transition(state, action)
        if invalid:
            return next_state, reward - 4.0, success, invalid
        # The learned agent still decides each action; the optimization route
        # provides only a reward signal, not a hard-coded motion plan.
        before = self.reference_distances[state[0], state[1]]
        after = self.reference_distances[next_state[0], next_state[1]]
        distance = float(np.linalg.norm(self.point(next_state) - self.point(state)))
        reward += self.guidance_weight * (before - after - 0.18 * after)
        reward -= 0.18 * distance  # shorter distance => lower time and energy
        return next_state, float(reward), success, False


def flight_telemetry(scenario, route, action_ids, grid, objective, checker):
    """Independently validate the policy's step-by-step flight and cost budget."""
    route = np.asarray(route, dtype=float)
    if not len(route) or not np.all(np.isfinite(route)):
        raise ValueError("No valid route to fly")
    if not all(grid.segment_safe(a, b) for a, b in zip(route[:-1], route[1:])):
        raise ValueError("Autonomous route violates buffered no-fly zone")
    z = route[1:-1].flatten()
    feasible, message = checker.verify_all(z, objective.energy_cost, objective.time_cost)
    total_distance = float(objective.total_distance(z))
    total_time = float(objective.time_cost(z))
    total_energy = float(objective.energy_cost(z))
    cap = grid.original.drone_params["battery_capacity_Wh"]
    initial_soc = grid.original.drone_params.get("initial_soc", 1.0)
    reserve_soc = grid.original.drone_params.get("reserve_soc", 0.0)
    log = []
    elapsed_distance = 0.0
    for i, position in enumerate(route):
        if i:
            elapsed_distance += float(np.linalg.norm(route[i] - route[i - 1]))
        elapsed_time = total_time * elapsed_distance / total_distance if total_distance else 0.0
        energy = total_energy * elapsed_distance / total_distance if total_distance else 0.0
        remaining_soc = initial_soc - energy / cap
        log.append({
            "step": i,
            "x_m": float(position[0]), "y_m": float(position[1]),
            "distance_flown_m": float(elapsed_distance),
            "elapsed_time_s": float(elapsed_time),
            "energy_used_Wh": float(energy),
            "battery_soc_pct": float(100 * remaining_soc),
            "battery_reserve_ok": bool(remaining_soc >= reserve_soc),
            "action": (int(action_ids[i - 1]) if 0 < i <= len(action_ids)
                       and action_ids[i - 1] is not None else None),
        })
    return {
        "success": bool(feasible), "feasibility_message": message,
        "total_distance_m": total_distance, "total_time_s": total_time,
        "total_energy_Wh": total_energy,
        "minimum_buffered_clearance_m": float(checker.minimum_clearance(z)),
        "time_limit_s": float(grid.original.max_time_s),
        "usable_battery_Wh": float(grid.original.usable_battery_Wh()),
        "initial_battery_soc_pct": 100 * initial_soc,
        "remaining_battery_soc_pct": log[-1]["battery_soc_pct"],
        "state_log": log,
    }
