"""Small regression suite for the reused optimizer and Q-learning demo."""

import json
import unittest
from pathlib import Path

import numpy as np

from src.environment import Environment2D
from src.experiments import build_problem, run_method
from src.rl_qlearning import GridDroneEnvironment, QLearningAgent
from src.autonomous_flight import GuidedGridDroneEnvironment, distance_to_polyline, flight_telemetry
from src.objective import Objective
from src.feasibility import FeasibilityChecker


ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "data" / "drone_optimization_dataset_100.json").read_text())


class DroneProjectTests(unittest.TestCase):
    def test_dataset_has_100_distinct_geometries(self):
        self.assertEqual(len(DATA), 100)
        self.assertEqual(len({row["id"] for row in DATA}), 100)
        self.assertEqual(len({(tuple(row["start"]), tuple(row["goal"])) for row in DATA}), 100)
        self.assertTrue(all(2 <= len(row["obstacles"]) <= 3 for row in DATA))
        self.assertEqual(len({json.dumps(row["drone_params"], sort_keys=True) for row in DATA}), 1)
        self.assertTrue(all(not Environment2D(row).validate_fixed_points() for row in DATA))

    def test_energy_is_in_watt_hours(self):
        env, obj, *_rest = build_problem(DATA[0], 5)
        z = _rest[-1]
        power_w = obj.c0 + obj.c1 * obj.v_max + obj.c2 * obj.v_max**2
        self.assertAlmostEqual(obj.energy_cost(z), power_w * obj.time_cost(z) / 3600.0)
        self.assertGreater(env.usable_battery_Wh(), obj.energy_cost(z))

    def test_penalized_gradient_matches_finite_difference(self):
        env, obj, cons, derivatives, feas, z0 = build_problem(DATA[0], 5)
        z = z0 + np.random.default_rng(9).normal(0, .05, len(z0))
        params = (.5, .5, 1000., .1)
        analytic = derivatives.gradient(z, *params)
        h = 1e-6
        numerical = np.array([
            (derivatives.penalized_objective(z + np.eye(len(z))[i] * h, *params)
             - derivatives.penalized_objective(z - np.eye(len(z))[i] * h, *params)) / (2 * h)
            for i in range(len(z))
        ])
        self.assertLess(np.linalg.norm(analytic - numerical) / max(np.linalg.norm(numerical), 1), 1e-5)

    def test_metric_objective_uses_actual_optimizer_weights(self):
        env, obj, cons, derivatives, feas, z0 = build_problem(DATA[0], 5)
        for method in ("Gradient Descent", "Heavy-Ball"):
            solution, optimizer, metrics = run_method(
                env, obj, cons, derivatives, feas, z0.copy(), method,
                rho=1000., max_iter=80, mode="balanced"
            )
            self.assertEqual((optimizer.w_time, optimizer.w_energy), (.5, .5))
            self.assertAlmostEqual(metrics["objective"], obj.combined_objective(solution, .5, .5))

    def test_qlearning_trains_and_finds_spatially_safe_path(self):
        env = GridDroneEnvironment(DATA[0], grid_step=.5)
        agent = QLearningAgent(env, seed=2026)
        history = agent.train(episodes=1200)
        states, goal_reached = agent.greedy_rollout()
        self.assertTrue(goal_reached)
        self.assertGreater(np.mean([item["success"] for item in history[-100:]]), .9)
        path = env.continuous_path(states)
        self.assertTrue(all(env.segment_safe(a, b) for a, b in zip(path[:-1], path[1:])))

    def test_guided_autonomous_policy_respects_battery_time_and_obstacles(self):
        hybrid = json.loads((ROOT / "results" / "hybrid" / "hybrid_comparison.json").read_text())
        reference = hybrid["paths"]["Gradient Descent"]
        grid = GuidedGridDroneEnvironment(DATA[0], reference)
        agent = QLearningAgent(grid, seed=2026)
        history = agent.train(episodes=1500)
        states, success = agent.greedy_rollout()
        self.assertTrue(success)
        self.assertGreater(np.mean([row["success"] for row in history[-100:]]), .95)
        route = grid.continuous_path(states)
        result = flight_telemetry(DATA[0], route, [None] * (len(route) - 1),
                                  grid, Objective(grid.original), FeasibilityChecker(grid.original))
        self.assertTrue(result["success"])
        self.assertLess(result["total_time_s"], result["time_limit_s"])
        self.assertLess(result["total_energy_Wh"], result["usable_battery_Wh"])
        self.assertGreaterEqual(result["minimum_buffered_clearance_m"], -1e-9)

    def test_guidance_rejects_unsafe_optimized_reference(self):
        straight_line = [DATA[0]["start"], DATA[0]["goal"]]
        with self.assertRaisesRegex(ValueError, "reference_path intersects"):
            GuidedGridDroneEnvironment(DATA[0], straight_line)
        self.assertAlmostEqual(distance_to_polyline([3, 1], [[0, 0], [4, 0]]), 1.0)


if __name__ == "__main__":
    unittest.main()
