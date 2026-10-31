"""Train an autonomous Q-learning flight policy using an optimized route as guidance.

Run after `scripts/refine_learned_route.py`. No pretrained neural network,
physical flight dynamics, or generalization to unseen maps is claimed.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.autonomous_flight import GuidedGridDroneEnvironment, flight_telemetry
from src.feasibility import FeasibilityChecker
from src.objective import Objective
from src.rl_qlearning import QLearningAgent


def policy_action_ids(grid, states, route):
    points = [grid.point(s) for s in states]
    indexed = []
    for start, end in zip(route[:-1], route[1:]):
        found = None
        for (s0, s1), p0, p1 in zip(zip(states[:-1], states[1:]), points[:-1], points[1:]):
            if np.allclose(start, p0) and np.allclose(end, p1):
                delta = np.asarray(s1) - np.asarray(s0)
                matches = np.flatnonzero(np.all(grid.ACTIONS == delta, axis=1))
                if len(matches):
                    found = int(matches[0])
                    break
        indexed.append(found)
    return indexed


def save_plots(grid, reference, route, history, destination):
    fig, ax = plt.subplots(figsize=(8, 7))
    env = grid.original
    for obstacle in env.get_obstacles():
        ax.add_patch(plt.Circle(obstacle["center"], obstacle["radius"], alpha=.55, color="#64748b"))
        ax.add_patch(plt.Circle(obstacle["center"], env.effective_obstacle_radius(obstacle),
                                fill=False, linestyle="--", color="#f97316", lw=1.5))
    ax.plot(reference[:, 0], reference[:, 1], "--", color="#16a34a", lw=2.5,
            label="GD optimized guidance (reference)")
    ax.plot(route[:, 0], route[:, 1], "-o", color="#2563eb", lw=2,
            markersize=3, label="Autonomous Q-learning flight")
    ax.scatter(*env.get_start(), s=90, color="#15803d", label="Start", zorder=5)
    ax.scatter(*env.get_goal(), s=90, color="#dc2626", label="Goal", zorder=5)
    ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="X (m)", ylabel="Y (m)",
           title="Optimization-guided autonomous navigation")
    ax.grid(alpha=.2)
    ax.set_aspect("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(destination / "autonomous_route.png", dpi=180)
    plt.close(fig)

    successes = np.asarray([h["success"] for h in history], dtype=float)
    rewards = np.asarray([h["reward"] for h in history])
    window = min(100, len(history))
    kernel = np.ones(window) / window
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, series, title in zip(axes, (successes, rewards),
                                 ("Training success (rolling 100)", "Training reward (rolling 100)")):
        value = np.convolve(series, kernel, mode="valid")
        ax.plot(np.arange(window, window + len(value)), value, color="#0891b2")
        ax.set_ylabel(title)
        ax.grid(alpha=.2)
    axes[1].set_xlabel("Episode")
    fig.tight_layout()
    fig.savefig(destination / "autonomous_training.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "drone_optimization_dataset_100.json")
    parser.add_argument("--hybrid-results", type=Path, default=ROOT / "results" / "hybrid" / "hybrid_comparison.json")
    parser.add_argument("--reference", default="Gradient Descent", choices=["Gradient Descent", "Heavy-Ball"])
    parser.add_argument("--scenario-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "autonomous" / "scenario_0")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    scenario = next((row for row in dataset if int(row["id"]) == args.scenario_id), None)
    if scenario is None:
        parser.error(f"Scenario {args.scenario_id} is not in the dataset")
    hybrid = json.loads(args.hybrid_results.read_text(encoding="utf-8"))
    if int(hybrid["scenario_id"]) != args.scenario_id:
        parser.error("The hybrid optimized reference does not match the requested scenario")
    if not hybrid["methods"][args.reference]["feasible"]:
        parser.error("Selected optimized reference path is not feasible")
    reference = np.asarray(hybrid["paths"][args.reference], dtype=float)
    grid = GuidedGridDroneEnvironment(scenario, reference, grid_step=args.grid_step)
    agent = QLearningAgent(grid, seed=args.seed)
    history = agent.train(args.episodes)
    states, reached_goal = agent.greedy_rollout()
    if not reached_goal:
        raise SystemExit("The trained greedy policy did not reach the goal. Try more episodes.")
    route = grid.continuous_path(states)
    original = grid.original
    metrics = flight_telemetry(scenario, route, policy_action_ids(grid, states, route),
                               grid, Objective(original), FeasibilityChecker(original))
    if not metrics["success"]:
        raise SystemExit(f"Learned flight violates constraints: {metrics['feasibility_message']}")
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    save_plots(grid, reference, route, history, destination)
    np.savez_compressed(destination / "autopilot_q_table.npz", q_values=agent.q_values)
    summary = {
        "algorithm": "optimization-guided tabular Q-learning (not DQN)",
        "scenario_id": args.scenario_id, "seed": args.seed,
        "episodes": args.episodes, "grid_step_m": args.grid_step,
        "trained_on_fixed_map": True,
        "optimized_guidance_method": args.reference,
        "reference_distance_m": float(hybrid["methods"][args.reference]["distance_m"]),
        "success_rate_last_100_training_episodes": float(np.mean([h["success"] for h in history[-100:]])),
        "policy_reached_goal": bool(reached_goal),
        "note": ("The optimizer supplies a reference used in the reward. The Q-table decides "
                 "each move. This is fixed-map, fixed-altitude simulated navigation; "
                 "no unseen-map generalization, physical dynamics, or real flight is claimed."),
        **{key: value for key, value in metrics.items() if key != "state_log"},
    }
    (destination / "flight_log.json").write_text(json.dumps({
        "scenario": scenario, "reference": reference.tolist(),
        "autonomous_path": route.tolist(), "telemetry": metrics["state_log"],
    }, indent=2), encoding="utf-8")
    (destination / "autopilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (destination / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
