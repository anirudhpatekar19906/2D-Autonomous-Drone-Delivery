import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis import hessian_analysis
from experiments import build_problem, run_method
from visualization import Visualizer


ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "data" / "drone_optimization_dataset.json"
RESULTS = ROOT / "results"
WAYPOINTS = 5
MAX_ITER = 3000
RHO = 1000.0
TOL = 1e-8
MODE = "balanced"


def stopping_reason(optimizer):
    return optimizer.stopping_reason or "no_iterations"


def run_scenario(scenario, scenario_index):
    environment, objective, constraints, derivatives, feasibility, z0 = build_problem(
        scenario, WAYPOINTS
    )
    initial = z0.copy()
    methods = {}
    solutions = {}
    optimizers = {}
    method_initials = []
    for method in ("Gradient Descent", "Heavy-Ball"):
        method_initial = initial.copy()
        method_initials.append(method_initial)
        solution, optimizer, metrics = run_method(
            environment,
            objective,
            constraints,
            derivatives,
            feasibility,
            method_initial,
            method,
            rho=RHO,
            max_iter=MAX_ITER,
            tol=TOL,
            mode=MODE,
        )
        alpha, beta = objective.mode_weights(MODE)
        hessian = hessian_analysis(
            derivatives, solution, w_time=alpha, w_energy=beta, rho=RHO
        )
        methods[method] = {
            **metrics,
            "stopping_reason": stopping_reason(optimizer),
            "optimizer_tolerance": TOL,
            "maximum_iterations": MAX_ITER,
            "penalty_coefficient": RHO,
            "hessian_classification": hessian["classification"],
            "condition_number": hessian["condition_number"],
            "minimum_eigenvalue": hessian["minimum_eigenvalue"],
            "maximum_eigenvalue": hessian["maximum_eigenvalue"],
            "penalized_objective_history": [float(value) for value in optimizer.history],
            "gradient_norm_history": [float(value) for value in optimizer.grad_norm_history],
            "constraint_violation_history": [
                float(value) for value in optimizer.constraint_violation_history
            ],
        }
        solutions[method] = solution
        optimizers[method] = optimizer

    if not all(np.array_equal(initial, method_initial) for method_initial in method_initials):
        raise AssertionError("GD and Heavy-Ball did not receive the same initial vector")

    return {
        "scenario_id": scenario["id"],
        "scenario_index": scenario_index,
        "waypoints": WAYPOINTS,
        "shared_initial_waypoint_vector": initial.tolist(),
        "same_initialization": bool(
            np.array_equal(method_initials[0], method_initials[1])
        ),
        "methods": methods,
        "environment": environment,
        "solutions": solutions,
        "optimizers": optimizers,
    }


def aggregate(rows):
    summary = {}
    for method in ("Gradient Descent", "Heavy-Ball"):
        method_rows = [row for row in rows if row["method"] == method]

        def values(key):
            return [row[key] for row in method_rows if row[key] is not None]

        def mean(key):
            data = values(key)
            return float(np.mean(data)) if data else None

        def median(key):
            data = values(key)
            return float(np.median(data)) if data else None

        objectives = values("final_objective")
        summary[method] = {
            "scenario_count": len(method_rows),
            "mean_final_objective": mean("final_objective"),
            "median_final_objective": median("final_objective"),
            "std_final_objective": float(np.std(objectives)) if objectives else None,
            "mean_iterations": mean("iterations"),
            "median_iterations": median("iterations"),
            "mean_runtime_s": mean("runtime_s"),
            "median_runtime_s": median("runtime_s"),
            "mean_final_penalized_gradient_norm": mean("final_penalized_gradient_norm"),
            "median_final_penalized_gradient_norm": median("final_penalized_gradient_norm"),
            "mean_sampled_constraint_violation": mean("maximum_sampled_constraint_violation"),
            "median_sampled_constraint_violation": median("maximum_sampled_constraint_violation"),
            "max_sampled_constraint_violation": max(
                values("maximum_sampled_constraint_violation"), default=None
            ),
            "mean_minimum_clearance_m": mean("minimum_clearance_m"),
            "median_minimum_clearance_m": median("minimum_clearance_m"),
            "exact_feasibility_rate": float(
                np.mean([row["exact_feasible"] for row in method_rows])
            ),
            "convergence_rate": float(
                np.mean([row["stopping_reason"] != "maximum_iterations" for row in method_rows])
            ),
            "valid_condition_number_count": sum(
                row["condition_number"] is not None for row in method_rows
            ),
            "mean_condition_number": mean("condition_number"),
            "hessian_classification_counts": dict(
                __import__("collections").Counter(
                    row["hessian_classification"] for row in method_rows
                )
            ),
            "condition_number_note": (
                "Kappa is averaged only over positive-definite Hessians; "
                "indefinite, semidefinite, and singular cases are reported "
                "without forcing a condition number."
            ),
        }
    return summary


def flatten_rows(results):
    rows = []
    for result in results:
        for method, metrics in result["methods"].items():
            rows.append({
                "scenario_id": result["scenario_id"],
                "same_initialization": result["same_initialization"],
                "waypoints": result["waypoints"],
                "method": method,
                "iterations": metrics["iterations"],
                "stopping_reason": metrics["stopping_reason"],
                "runtime_s": metrics["runtime_s"],
                "final_objective": metrics["objective"],
                "final_penalized_objective": metrics["penalized_objective"],
                "final_penalized_gradient_norm": metrics["final_gradient_norm"],
                "maximum_sampled_constraint_violation": metrics["constraint_report"]["maximum_violation"],
                "exact_feasible": metrics["feasible"],
                "minimum_clearance_m": metrics["minimum_buffered_clearance_m"],
                "condition_number": metrics["condition_number"],
                "hessian_classification": metrics["hessian_classification"],
            })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path, summary):
    rows = []
    for method, values in summary.items():
        row = {"method": method}
        row.update({key: value for key, value in values.items() if key != "hessian_classification_counts"})
        row["hessian_classification_counts"] = json.dumps(
            values["hessian_classification_counts"], sort_keys=True
        )
        rows.append(row)
    write_csv(path, rows)


def save_scenario_0(result):
    output_dir = RESULTS / "convergence"
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = Visualizer(result["environment"])
    initial = np.asarray(result["shared_initial_waypoint_vector"])
    visualizer.plot_trajectory(initial, "Scenario 0 Initial Route", RESULTS / "trajectories" / "comparison_initial.png")
    for method, solution in result["solutions"].items():
        visualizer.plot_trajectory(solution, f"Scenario 0 {method} Route", RESULTS / "trajectories" / f"comparison_{method.lower().replace(' ', '_')}.png")
    visualizer.plot_diagnostics(
        result["optimizers"]["Gradient Descent"],
        result["optimizers"]["Heavy-Ball"],
        output_dir,
        title_prefix="Representative Scenario 0",
    )
    visualizer.plot_trajectory_comparison(
        initial,
        result["solutions"]["Gradient Descent"],
        result["solutions"]["Heavy-Ball"],
        RESULTS / "trajectories" / "comparison_overlay_scenario_0.png",
    )


def representative_scenarios(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario_id"], {})[row["method"]] = row

    def score(scenario_id, key):
        pair = grouped[scenario_id]
        return max(float(pair[method][key]) for method in pair)

    scenario_ids = list(grouped)
    stable_candidates = [
        scenario_id for scenario_id in scenario_ids if all(
            grouped[scenario_id][method]["exact_feasible"]
            for method in grouped[scenario_id]
        )
    ]
    stable = min(
        stable_candidates or scenario_ids,
        key=lambda scenario_id: score(scenario_id, "final_penalized_gradient_norm"),
    )
    difficult = max(
        scenario_ids,
        key=lambda scenario_id: score(scenario_id, "maximum_sampled_constraint_violation"),
    )
    contrasting = max(
        scenario_ids,
        key=lambda scenario_id: int(
            grouped[scenario_id]["Gradient Descent"]["exact_feasible"]
            != grouped[scenario_id]["Heavy-Ball"]["exact_feasible"]
        ),
    )
    selected = {
        "scenario_0": 0,
        "easy_stable": stable,
        "difficult_high_violation": difficult,
        "contrasting_feasibility": contrasting,
    }
    return {
        name: {
            "scenario_id": scenario_id,
            "selection_reason": reason,
            "methods": {
                method: grouped[scenario_id][method]
                for method in ("Gradient Descent", "Heavy-Ball")
            },
        }
        for name, scenario_id, reason in (
            ("scenario_0", selected["scenario_0"], "primary representative scenario"),
            ("easy_stable", selected["easy_stable"], "lowest paired final penalized gradient among exactly feasible pairs"),
            ("difficult_high_violation", selected["difficult_high_violation"], "highest paired sampled constraint violation"),
            ("contrasting_feasibility", selected["contrasting_feasibility"], "GD and Heavy-Ball exact feasibility differs when available"),
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    scenarios = dataset[:args.limit] if args.limit else dataset

    results = []
    for index, scenario in enumerate(scenarios):
        print(f"Running scenario {index + 1}/{len(scenarios)}: {scenario['id']}")
        result = run_scenario(scenario, index)
        results.append(result)
        if index == 0:
            save_scenario_0(result)

    rows = flatten_rows(results)
    summary = aggregate(rows)
    (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
    configuration = {
        "dataset_path": str(DATASET_PATH),
        "scenarios": len(scenarios),
        "waypoints": WAYPOINTS,
        "learning_rate": 0.01,
        "heavy_ball_momentum": 0.9,
        "max_iter": MAX_ITER,
        "rho": RHO,
        "tol": TOL,
        "mode": MODE,
        "initialization": "straight",
        "initialization_seed": 0,
    }
    (RESULTS / "metrics" / "gd_hb_summary.json").write_text(
        json.dumps({"configuration": configuration, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    write_csv(RESULTS / "metrics" / "gd_hb_all_scenarios.csv", rows)
    write_summary_csv(RESULTS / "metrics" / "gd_hb_summary.csv", summary)
    (RESULTS / "metrics" / "gd_hb_all_scenarios.json").write_text(
        json.dumps({"configuration": configuration, "results": rows}, indent=2),
        encoding="utf-8",
    )
    (RESULTS / "metrics" / "gd_hb_representative_scenarios.json").write_text(
        json.dumps(representative_scenarios(rows), indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()