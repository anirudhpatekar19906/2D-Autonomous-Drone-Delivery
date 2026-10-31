"""Train independent Q-tables on several maps; measure per-map outcomes.

This is NOT evaluation of one policy on held-out maps. Each scenario gets a
fresh agent, so results describe feasibility and learnability per scenario.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.environment import Environment2D
from src.feasibility import FeasibilityChecker
from src.objective import Objective
from src.rl_qlearning import GridDroneEnvironment, QLearningAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "drone_optimization_dataset_100.json")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "rl" / "multi_scenario_evaluation.json")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))[:args.count]
    records = []
    for i, scenario in enumerate(dataset):
        row = {"scenario_id": scenario["id"], "training_episodes": args.episodes}
        try:
            grid = GridDroneEnvironment(scenario, grid_step=.5)
            agent = QLearningAgent(grid, seed=args.seed + i)
            history = agent.train(args.episodes)
            states, success = agent.greedy_rollout()
            row["train_success_last_100"] = float(np.mean([x["success"] for x in history[-100:]]))
            row["greedy_reached_goal"] = bool(success)
            row["steps"] = len(states) - 1
            if success:
                points = grid.continuous_path(states)
                z = points[1:-1].flatten()
                env = Environment2D(scenario)
                objective = Objective(env)
                checker = FeasibilityChecker(env)
                feasible, message = checker.verify_all(z, objective.energy_cost, objective.time_cost)
                row.update({
                    "fully_feasible": bool(feasible),
                    "feasibility_message": message,
                    "path_length_m": float(objective.total_distance(z)),
                    "min_buffered_clearance_m": float(checker.minimum_clearance(z)),
                })
        except (ValueError, RuntimeError) as error:
            row["error"] = str(error)
            row["greedy_reached_goal"] = False
        records.append(row)
        print(f"Scenario {scenario['id']}: goal={row['greedy_reached_goal']}, feasible={row.get('fully_feasible', False)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "method": "Independent tabular Q-learning training per scenario",
        "warning": "Not one-policy generalization to unseen maps",
        "seed": args.seed,
        "episodes_per_scenario": args.episodes,
        "scenario_count": len(records),
        "greedy_success_rate": sum(x["greedy_reached_goal"] for x in records) / len(records),
        "full_feasibility_rate": sum(x.get("fully_feasible", False) for x in records) / len(records),
        "records": records,
    }
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    columns = ["scenario_id", "training_episodes", "train_success_last_100", "greedy_reached_goal",
               "steps", "fully_feasible", "path_length_m", "min_buffered_clearance_m", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({key: value for key, value in data.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
