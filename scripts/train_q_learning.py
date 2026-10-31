"""Train and evaluate a real Q-learning agent on one fixed drone map."""

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

from src.environment import Environment2D
from src.feasibility import FeasibilityChecker
from src.objective import Objective
from src.rl_qlearning import GridDroneEnvironment, QLearningAgent


def moving_average(values, width=100):
    values = np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(width) / width, mode="valid") if len(values) >= width else values


def draw_route(grid, route, output_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    original = grid.original
    for obstacle in original.get_obstacles():
        center = obstacle["center"]
        ax.add_patch(plt.Circle(center, obstacle["radius"], color="#9ca3af", alpha=0.65))
        ax.add_patch(plt.Circle(center, original.effective_obstacle_radius(obstacle),
                                fill=False, color="#f97316", linestyle="--", linewidth=1.7))
    straight = np.vstack([original.get_start(), original.get_goal()])
    ax.plot(straight[:, 0], straight[:, 1], "k--", alpha=0.35, label="Unsafe direct route")
    ax.plot(route[:, 0], route[:, 1], color="#2563eb", linewidth=2.3, marker="o",
            markersize=3, label="Q-learning learned route")
    ax.scatter(*original.get_start(), color="green", s=100, zorder=5, label="Start")
    ax.scatter(*original.get_goal(), color="red", s=100, zorder=5, label="Goal")
    ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="X (m)", ylabel="Y (m)", title="Learned drone route and safety buffers")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def draw_training(history, output_path):
    rewards = [row["reward"] for row in history]
    successes = [float(row["success"]) for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, values, color, label in (
        (axes[0], rewards, "#2563eb", "100-episode average reward"),
        (axes[1], successes, "#16a34a", "100-episode training success rate"),
    ):
        averaged = moving_average(values)
        x = np.arange(len(averaged)) + (100 if len(values) >= 100 else 1)
        ax.plot(x, averaged, color=color, linewidth=2)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylim(-0.03, 1.03)
    fig.suptitle("Tabular Q-learning training progress")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "drone_optimization_dataset_100.json")
    parser.add_argument("--scenario-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "rl" / "scenario_0")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    scenario = next((row for row in dataset if int(row["id"]) == args.scenario_id), None)
    if scenario is None:
        parser.error(f"Scenario {args.scenario_id} not found")
    grid = GridDroneEnvironment(scenario, grid_step=args.grid_step)
    agent = QLearningAgent(grid, seed=args.seed)
    history = agent.train(episodes=args.episodes)
    states, success = agent.greedy_rollout()
    route = grid.continuous_path(states) if success else np.empty((0, 2))

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "algorithm": "tabular Q-learning (not DQN)",
        "scenario_id": args.scenario_id,
        "training_episodes": args.episodes,
        "seed": args.seed,
        "grid_step_m": args.grid_step,
        "training_success_last_100": float(np.mean([row["success"] for row in history[-100:]])),
        "greedy_policy_reached_goal": bool(success),
        "greedy_path_steps": len(states) - 1,
        "note": "This Q-table was trained and evaluated on one fixed obstacle map; no cross-map generalization is claimed.",
    }
    if success:
        original = Environment2D(scenario)
        objective = Objective(original)
        checker = FeasibilityChecker(original)
        waypoints = route[1:-1].flatten()
        fully_feasible, message = checker.verify_all(waypoints, objective.energy_cost, objective.time_cost)
        spatially_safe = all(grid.segment_safe(a, b) for a, b in zip(route[:-1], route[1:]))
        summary.update({
            "route": route.tolist(),
            "distance_m": float(objective.total_distance(waypoints)),
            "estimated_time_s": float(objective.time_cost(waypoints)),
            "estimated_energy_Wh": float(objective.energy_cost(waypoints)),
            "minimum_buffered_clearance_m": float(checker.minimum_clearance(waypoints)),
            "spatially_safe": bool(spatially_safe),
            "fully_feasible": bool(fully_feasible),
            "feasibility_message": message,
        })
        draw_route(grid, route, output / "learned_route.png")
    draw_training(history, output / "training_curve.png")
    np.savez_compressed(output / "q_table.npz", q_values=agent.q_values)
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "route"}, indent=2))
    if not success:
        raise SystemExit("The greedy policy did not reach the goal; adjust training and retry")


if __name__ == "__main__":
    main()
