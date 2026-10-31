import numpy as np


def constraint_report(environment, objective, constraints, feasibility, z):
    values = constraints.all_constraints(z, objective.energy_cost, objective.time_cost)
    labels = ["battery", "time"]
    labels.extend(
        f"waypoint_{i + 1}_obstacle_{j + 1}"
        for i in range(len(z) // 2)
        for j in range(len(environment.get_obstacles()))
    )
    labels.extend(
        f"segment_sample_{i + 1}_obstacle_{j + 1}"
        for i in range((len(z) // 2 + 1) * constraints.segment_samples)
        for j in range(len(environment.get_obstacles()))
    )
    labels.extend(
        f"workspace_{axis}_{bound}_waypoint_{i + 1}"
        for i in range(len(z) // 2)
        for axis in ("x", "y")
        for bound in ("lower", "upper")
    )
    return {
        "maximum_violation": float(max(0.0, np.max(values))),
        "active_count": int(np.count_nonzero(np.abs(values) <= 1e-3)),
        "values": [
            {
                "name": label,
                "value": float(value),
                "slack": float(-value),
                "status": "active" if abs(value) <= 1e-3 else "inactive" if value < 0 else "violated",
            }
            for label, value in zip(labels, values)
        ],
    }


def solution_metrics(environment, objective, constraints, feasibility, optimizer, z):
    feasible, message = feasibility.verify_all(
        z, objective.energy_cost, objective.time_cost
    )
    condition_number = optimizer.deriv.condition_number(z, rho=optimizer.rho)
    return {
        "feasible": feasible,
        "message": message,
        "objective": float(objective.combined_objective(z, optimizer.w_time, optimizer.w_energy, optimizer.w_smooth)),
        "penalized_objective": float(optimizer.deriv.penalized_objective(z, optimizer.w_time, optimizer.w_energy, optimizer.rho, optimizer.w_smooth)),
        "distance_m": float(objective.total_distance(z)),
        "time_s": float(objective.time_cost(z)),
        "time_limit_s": float(environment.max_time_s),
        "energy_Wh": float(objective.energy_cost(z)),
        "battery_limit_Wh": float(environment.usable_battery_Wh()),
        "minimum_buffered_clearance_m": float(feasibility.minimum_clearance(z)),
        "iterations": len(optimizer.history),
        "runtime_s": float(optimizer.runtime),
        "final_gradient_norm": float(np.linalg.norm(optimizer.deriv.gradient(
            z, optimizer.w_time, optimizer.w_energy, optimizer.rho, optimizer.w_smooth
        ))),
        "condition_number": float(condition_number) if np.isfinite(condition_number) else None,
        "constraint_report": constraint_report(environment, objective, constraints, feasibility, z),
    }
