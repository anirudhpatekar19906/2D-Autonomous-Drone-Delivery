import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments import build_problem, run_method
from kkt_analysis import analyze_kkt


ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "data" / "drone_optimization_dataset.json"
OUTPUT_PATH = ROOT / "results" / "metrics" / "kkt_scenario_0.json"
REPRESENTATIVE_OUTPUT_PATH = ROOT / "results" / "metrics" / "kkt_representative_scenarios.json"
REPRESENTATIVE_SCENARIO_IDS = (0, 10, 29, 35, 44)


def stopping_reason(optimizer):
    """Explain the existing optimizer's observed termination condition."""
    if not optimizer.history:
        return "no_iterations"
    if optimizer.grad_norm_history[-1] < optimizer.tol:
        return "gradient_norm_tolerance"
    if len(optimizer.history) > 1:
        objective_change = abs(optimizer.history[-1] - optimizer.history[-2])
        if objective_change < optimizer.tol:
            return "objective_change_tolerance"
    if len(optimizer.history) >= optimizer.max_iter:
        return "maximum_iterations"
    return "nonfinite_step_or_other_exit"


def method_report(environment, objective, constraints, derivatives, feasibility, z0, method):
    candidate, optimizer, metrics = run_method(
        environment,
        objective,
        constraints,
        derivatives,
        feasibility,
        z0,
        method,
        rho=1000.0,
        max_iter=3000,
        tol=1e-8,
        mode="balanced",
    )
    report = analyze_kkt(
        environment, objective, constraints, candidate, w_time=0.5, w_energy=0.5
    )
    exact_feasible, exact_message = feasibility.verify_all(
        candidate, objective.energy_cost, objective.time_cost
    )
    report["candidate"] = {
        "optimizer": method,
        "decision_vector": candidate.tolist(),
        "iterations": int(len(optimizer.history)),
        "learning_rate": float(optimizer.eta),
        "momentum": float(getattr(optimizer, "beta", 0.0)),
        "penalty_coefficient": float(optimizer.rho),
        "maximum_iterations": int(optimizer.max_iter),
        "optimizer_tolerance": float(optimizer.tol),
        "stopping_reason": stopping_reason(optimizer),
        "last_objective_change": float(
            abs(optimizer.history[-1] - optimizer.history[-2])
        )
        if len(optimizer.history) > 1
        else None,
        "final_gradient_norm_penalized": float(optimizer.grad_norm_history[-1]),
        "objective": float(metrics["objective"]),
        "penalized_objective": float(metrics["penalized_objective"]),
        "minimum_buffered_clearance_m": float(metrics["minimum_buffered_clearance_m"]),
    }
    report["exact_feasibility"] = {
        "feasible": bool(exact_feasible),
        "message": exact_message,
    }
    report["summary"].update(
        {
            "candidate": method,
            "penalized_objective": float(metrics["penalized_objective"]),
            "optimizer_final_gradient_norm": float(metrics["final_gradient_norm"]),
            "formal_primal_feasible": bool(report["summary"]["feasible"]),
            "exact_feasible": bool(exact_feasible),
            "exact_feasibility_message": exact_message,
        }
    )
    return report


def scenario_report(scenario):
    environment, objective, constraints, derivatives, feasibility, z0 = build_problem(
        scenario, 5
    )
    reports = {
        method: method_report(
            environment, objective, constraints, derivatives, feasibility, z0, method
        )
        for method in ("Gradient Descent", "Heavy-Ball")
    }
    for report in reports.values():
        report["scenario_id"] = scenario["id"]
        report["waypoints"] = int(z0.size // 2)
        report["initial_decision_vector"] = z0.tolist()
        report["finite_difference_note"] = (
            "Central differences are applied to the original objective and the "
            "explicit sampled constraint functions. Results were stable for "
            "steps from 1e-4 through 1e-7 on Scenario 0. The exact "
            "projection-based feasibility checker is reported separately."
        )
    return {
        "scenario_id": scenario["id"],
        "waypoints": int(z0.size // 2),
        "shared_initial_decision_vector": z0.tolist(),
        "methods": reports,
        "kkt_summary_table": [
            reports[method]["summary"]
            for method in ("Gradient Descent", "Heavy-Ball")
        ],
        "stopping_diagnosis": {
            "existing_default_tolerance": 1e-5,
            "heavy_ball_default_run": (
                "With the existing default tolerance, Heavy-Ball stopped at "
                "306 iterations because objective change was below tolerance "
                "while its gradient norm remained above tolerance. The KKT "
                "runs use tol=1e-8 to prevent that premature exit."
            ),
        },
    }


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    scenarios = {
        scenario["id"]: scenario
        for scenario in dataset
        if scenario["id"] in REPRESENTATIVE_SCENARIO_IDS
    }
    missing = set(REPRESENTATIVE_SCENARIO_IDS) - set(scenarios)
    if missing:
        raise ValueError(f"Representative scenarios missing from dataset: {sorted(missing)}")

    representative_reports = {
        str(scenario_id): scenario_report(scenarios[scenario_id])
        for scenario_id in REPRESENTATIVE_SCENARIO_IDS
    }
    scenario_zero_report = representative_reports["0"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(scenario_zero_report, indent=2), encoding="utf-8")
    representative_output = {
        "scenario_ids": list(REPRESENTATIVE_SCENARIO_IDS),
        "reports": representative_reports,
        "kkt_summary_table": [
            {
                "scenario_id": int(scenario_id),
                **report["methods"][method]["summary"],
            }
            for scenario_id, report in representative_reports.items()
            for method in ("Gradient Descent", "Heavy-Ball")
        ],
    }
    REPRESENTATIVE_OUTPUT_PATH.write_text(
        json.dumps(representative_output, indent=2), encoding="utf-8"
    )
    print(json.dumps(representative_output, indent=2))


if __name__ == "__main__":
    main()
