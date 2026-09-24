"""Numerical KKT analysis for the waypoint trajectory problem.

The optimizers use a squared-violation penalty, but KKT conditions apply to
the original constrained problem.  This module therefore differentiates the
physical objective and the repository's explicit inequality constraints
directly, using the convention g(z) <= 0.
"""

import numpy as np


def constraint_labels(environment, constraints, waypoint_count):
    """Return labels in the same order as ``Constraints.all_constraints``."""
    obstacle_count = len(environment.get_obstacles())
    labels = ["battery", "time"]
    labels.extend(
        f"waypoint_{i + 1}_obstacle_{j + 1}"
        for i in range(waypoint_count)
        for j in range(obstacle_count)
    )
    labels.extend(
        f"segment_sample_{i + 1}_obstacle_{j + 1}"
        for i in range((waypoint_count + 1) * constraints.segment_samples)
        for j in range(obstacle_count)
    )
    labels.extend(
        f"workspace_{axis}_{bound}_waypoint_{i + 1}"
        for i in range(waypoint_count)
        for axis in ("x", "y")
        for bound in ("lower", "upper")
    )
    return labels


def _central_difference(function, z, step):
    """Return a central-difference gradient/Jacobian for a scalar/vector function."""
    z = np.asarray(z, dtype=float)
    base = np.asarray(function(z), dtype=float)
    jacobian = np.empty((base.size, z.size), dtype=float)
    for index in range(z.size):
        perturbation = np.zeros_like(z)
        perturbation[index] = step
        plus = np.asarray(function(z + perturbation), dtype=float).reshape(-1)
        minus = np.asarray(function(z - perturbation), dtype=float).reshape(-1)
        jacobian[:, index] = (plus - minus) / (2.0 * step)
    if base.ndim == 0:
        return jacobian[0]
    return jacobian


def analyze_kkt(
    environment,
    objective,
    constraints,
    z,
    w_time=1.0,
    w_energy=1.0,
    w_smooth=0.0,
    active_tolerance=1e-3,
    feasibility_tolerance=1e-6,
    finite_difference_step=1e-6,
):
    """Compute a numerical KKT report for a candidate waypoint vector.

    Multipliers are estimated from the active-constraint stationarity system
    ``J_active.T @ lambda = -gradient(f)``.  The unconstrained least-squares
    estimate is reported as-is; negative values are intentionally retained so
    dual infeasibility is visible rather than hidden by clipping.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size % 2:
        raise ValueError("z must be a flat waypoint vector with an even length")

    objective_function = lambda point: objective.combined_objective(
        point, w_time, w_energy, w_smooth
    )
    constraint_function = lambda point: constraints.all_constraints(
        point, objective.energy_cost, objective.time_cost
    )

    values = np.asarray(constraint_function(z), dtype=float).reshape(-1)
    labels = constraint_labels(environment, constraints, z.size // 2)
    if len(values) != len(labels):
        raise ValueError(
            "Constraint labels do not match Constraints.all_constraints output "
            f"({len(labels)} labels for {len(values)} values)"
        )

    objective_gradient = _central_difference(
        objective_function, z, finite_difference_step
    )
    constraint_jacobian = _central_difference(
        constraint_function, z, finite_difference_step
    )
    # KKT active-set multipliers are meaningful only for a primal-feasible
    # candidate. A violated inequality is therefore never admitted as active.
    candidate_primal_feasible = bool(
        np.max(np.maximum(values, 0.0), initial=0.0) <= feasibility_tolerance
    )
    active_mask = (
        candidate_primal_feasible
        & (values <= 0.0)
        & (values >= -active_tolerance)
    )
    active_indices = np.flatnonzero(active_mask)

    multipliers = np.zeros(values.size, dtype=float)
    licq_satisfied = True
    if active_indices.size:
        active_jacobian = constraint_jacobian[active_indices]

        # LICQ Check: Check if active constraints are linearly independent
        rank = np.linalg.matrix_rank(active_jacobian)
        if rank < len(active_indices):
            licq_satisfied = False

        multipliers[active_indices] = np.linalg.lstsq(
            active_jacobian.T, -objective_gradient, rcond=None
        )[0]

    stationarity = objective_gradient + constraint_jacobian.T @ multipliers
    complementarity = multipliers * values
    primal_violation = np.maximum(values, 0.0)
    dual_violation = np.maximum(-multipliers, 0.0)

    constraints_report = []
    for index, (label, value) in enumerate(zip(labels, values)):
        status = (
            "active"
            if active_mask[index]
            else "inactive"
            if value < 0.0
            else "violated"
        )
        constraints_report.append(
            {
                "name": label,
                "value": float(value),
                "slack": float(-value),
                "status": status,
                "multiplier": float(multipliers[index]),
                "complementary_slackness": float(complementarity[index]),
            }
        )

    residuals = {
        "primal_violation_max": float(np.max(primal_violation, initial=0.0)),
        "stationarity_norm": float(np.linalg.norm(stationarity)),
        "dual_violation_max": float(np.max(dual_violation, initial=0.0)),
        "complementary_slackness_max": float(
            np.max(np.abs(complementarity), initial=0.0)
        ),
    }
    checks = {
        "primal_feasible": residuals["primal_violation_max"] <= feasibility_tolerance,
        "dual_feasible": residuals["dual_violation_max"] <= feasibility_tolerance,
        "stationary": residuals["stationarity_norm"] <= feasibility_tolerance,
        "complementary_slackness": residuals["complementary_slackness_max"]
        <= feasibility_tolerance,
    }

    summary = {
        "objective": float(objective_function(z)),
        "feasible": checks["primal_feasible"],
        "max_constraint_violation": residuals["primal_violation_max"],
        "active_constraints": [
            constraints_report[index]["name"] for index in active_indices
        ],
        "multipliers": [
            constraints_report[index]["multiplier"] for index in active_indices
        ],
        "minimum_multiplier": float(np.min(multipliers[active_indices]))
        if active_indices.size
        else 0.0,
        "dual_feasibility": checks["dual_feasible"],
        "stationarity_residual": residuals["stationarity_norm"],
        "complementarity_residual": residuals["complementary_slackness_max"],
        "primal_feasibility": checks["primal_feasible"],
        "kkt_satisfied": bool(all(checks.values())),
        "licq_satisfied": licq_satisfied,
        "bottleneck_constraint": constraints_report[np.argmax(multipliers)]["name"] if active_indices.size else None,
        "bottleneck_multiplier": float(np.max(multipliers)) if active_indices.size else 0.0,
    }

    return {
        "formulation": {
            "constraint_convention": "g(z) <= 0",
            "lagrangian": "L(z, lambda) = f(z) + sum(lambda_j * g_j(z))",
            "objective": "f(z) = combined_objective(z, w_time, w_energy, w_smooth)",
            "decision_variables": "interior waypoint coordinates z",
            "constraint_groups": [
                "battery",
                "time",
                "waypoint NFZ constraints",
                "sampled interior-segment NFZ constraints",
                "workspace bounds",
            ],
            "theoretical_convexity": {
                "objective": "Convex (sum of Euclidean norms)",
                "feasible_region": "Non-convex (complement of disks/NFZs)",
                "overall_problem": "Non-convex (convex objective over non-convex set)",
                "kkt_implication": "KKT conditions are necessary but not sufficient for global optimality"
            },
            "segment_constraint_note": (
                "The formal KKT system uses the five interior samples per "
                "segment returned by Constraints.segment_nfz_constraint. "
                "FeasibilityChecker additionally uses the exact projection "
                "of each obstacle center onto every full segment; that exact "
                "continuous segment condition is reported separately and is "
                "not silently substituted into the formal KKT vector."
            ),
        },
        "settings": {
            "w_time": float(w_time),
            "w_energy": float(w_energy),
            "w_smooth": float(w_smooth),
            "active_tolerance": float(active_tolerance),
            "feasibility_tolerance": float(feasibility_tolerance),
            "finite_difference_step": float(finite_difference_step),
        },
        "objective_value": float(objective_function(z)),
        "objective_gradient_norm": float(np.linalg.norm(objective_gradient)),
        "active_constraint_count": int(np.count_nonzero(active_mask)),
        "active_constraints": [
            constraints_report[index]
            for index in active_indices
        ],
        "constraints": constraints_report,
        "residuals": residuals,
        "checks": checks,
        "summary": summary,
        "kkt_satisfied": bool(all(checks.values())),
    }