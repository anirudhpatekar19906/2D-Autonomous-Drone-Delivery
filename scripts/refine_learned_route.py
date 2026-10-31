"""Attempt GD/Heavy-Ball refinement of the RL-generated route; validate results."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from experiments import build_problem, run_method


def simple_metrics(env, objective, checker, z):
    safe, message = checker.verify_all(z, objective.energy_cost, objective.time_cost)
    return {
        "feasible": bool(safe),
        "message": message,
        "normalized_objective": float(objective.combined_objective(z, .5, .5)),
        "distance_m": float(objective.total_distance(z)),
        "time_s": float(objective.time_cost(z)),
        "energy_Wh": float(objective.energy_cost(z)),
        "minimum_buffered_clearance_m": float(checker.minimum_clearance(z)),
    }


def write_animated_3d_demo(environment, candidates, preferred, save_path):
    """Animate the selected 2D route at fixed altitude; not a 3D planner."""
    fig = go.Figure()
    altitude = 1.5
    for obstacle in environment.get_obstacles():
        center = np.asarray(obstacle["center"], dtype=float)
        radius = environment.effective_obstacle_radius(obstacle)
        theta = np.linspace(0, 2 * np.pi, 50)
        heights = np.linspace(0, 3, 8)
        theta_grid, z_grid = np.meshgrid(theta, heights)
        fig.add_trace(go.Surface(
            x=center[0] + radius * np.cos(theta_grid),
            y=center[1] + radius * np.sin(theta_grid), z=z_grid,
            surfacecolor=np.zeros_like(z_grid), colorscale=[[0, "#f59e0b"], [1, "#f59e0b"]],
            opacity=.18, showscale=False, hoverinfo="skip", name="Buffered obstacle",
        ))
    paths = {}
    for method, color in (("Q-learning route", "#2563eb"),
                          ("Gradient Descent", "#16a34a"),
                          ("Heavy-Ball", "#9333ea")):
        intermediate = candidates[method]["z"].reshape(-1, 2)
        points = np.vstack([environment.get_start(), intermediate, environment.get_goal()])
        paths[method] = points
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=np.full(len(points), altitude),
            mode="lines+markers", name=method, line=dict(color=color, width=7),
            marker=dict(size=3, color=color),
        ))
    points = paths[preferred]
    samples = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        samples.extend(start + (end - start) * fraction for fraction in np.linspace(.1, 1, 10))
    samples = np.asarray(samples)
    marker_index = len(fig.data)
    fig.add_trace(go.Scatter3d(
        x=[samples[0, 0]], y=[samples[0, 1]], z=[altitude], mode="markers",
        marker=dict(size=9, color="red", symbol="circle"), name="Moving drone",
    ))
    fig.frames = [go.Frame(
        data=[go.Scatter3d(x=[point[0]], y=[point[1]], z=[altitude], mode="markers",
                           marker=dict(size=9, color="red"))],
        traces=[marker_index], name=str(index),
    ) for index, point in enumerate(samples)]
    fig.update_layout(
        title=f"Animated fixed-altitude drone route: {preferred}",
        scene=dict(xaxis=dict(range=[0, 10], title="X (m)"),
                   yaxis=dict(range=[0, 10], title="Y (m)"),
                   zaxis=dict(range=[0, 3], title="Altitude (m)"), aspectmode="cube"),
        height=700,
        updatemenus=[dict(type="buttons", showactive=False, x=.02, y=.98,
                          buttons=[dict(label="Play", method="animate", args=[None, {
                              "frame": {"duration": 60, "redraw": True}, "fromcurrent": True,
                              "transition": {"duration": 0}, "mode": "immediate"}]),
                                   dict(label="Pause", method="animate", args=[[None], {
                                       "frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])])],
    )
    fig.write_html(save_path, include_plotlyjs=True, full_html=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "drone_optimization_dataset_100.json")
    parser.add_argument("--rl-summary", type=Path, default=ROOT / "results" / "rl" / "scenario_0" / "summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "hybrid")
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()
    summary = json.loads(args.rl_summary.read_text(encoding="utf-8"))
    if not summary["greedy_policy_reached_goal"]:
        raise ValueError("The RL learner did not produce a goal-reaching path")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    scenario = next(row for row in dataset if row["id"] == summary["scenario_id"])
    route = np.asarray(summary["route"], dtype=float)
    z0 = route[1:-1].flatten()
    env, obj, cons, deriv, feasibility, _ = build_problem(scenario, len(route) - 2)
    baseline = simple_metrics(env, obj, feasibility, z0)
    final = {"Q-learning route": {"z": z0, "metrics": baseline}}
    for method in ("Gradient Descent", "Heavy-Ball"):
        solution, optimizer, metrics = run_method(
            env, obj, cons, deriv, feasibility, z0.copy(), method,
            rho=1000., max_iter=args.max_iter, tol=1e-8, mode="balanced"
        )
        final[method] = {"z": solution, "metrics": simple_metrics(env, obj, feasibility, solution)}
        final[method]["metrics"]["iterations"] = len(optimizer.history)
        final[method]["metrics"]["stopping_reason"] = optimizer.stopping_reason

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_data = {
        "scenario_id": summary["scenario_id"],
        "note": "Refinement is accepted only when the independent feasibility check passes; an optimizer can worsen a learned route.",
        "methods": {method: item["metrics"] for method, item in final.items()},
        "paths": {
            method: np.vstack([env.get_start(), item["z"].reshape(-1, 2), env.get_goal()]).tolist()
            for method, item in final.items()
        },
    }
    safe_methods = {method: item["metrics"] for method, item in final.items() if item["metrics"]["feasible"]}
    save_data["best_safe_method"] = min(
        safe_methods, key=lambda method: safe_methods[method]["normalized_objective"]
    ) if safe_methods else None
    # Prefer a little spare clearance beyond the already buffered exclusion zone.
    conservative = {
        method: item for method, item in safe_methods.items()
        if item["minimum_buffered_clearance_m"] >= .05
    }
    save_data["preferred_demo_method"] = min(
        conservative or safe_methods,
        key=lambda method: (conservative or safe_methods)[method]["normalized_objective"]
    ) if safe_methods else None
    save_data["demo_clearance_preference_m"] = .05

    fig, ax = plt.subplots(figsize=(8, 7))
    for obstacle in env.get_obstacles():
        ax.add_patch(plt.Circle(obstacle["center"], obstacle["radius"], color="#9ca3af", alpha=.6))
        ax.add_patch(plt.Circle(obstacle["center"], env.effective_obstacle_radius(obstacle), fill=False,
                                linestyle="--", color="#f97316"))
    for method, (color, style) in {
        "Q-learning route": ("#2563eb", "-"),
        "Gradient Descent": ("#16a34a", "--"),
        "Heavy-Ball": ("#9333ea", ":"),
    }.items():
        values = final[method]["z"].reshape(-1, 2)
        full_path = np.vstack([env.get_start(), values, env.get_goal()])
        label = f"{method} ({'safe' if final[method]['metrics']['feasible'] else 'unsafe'})"
        ax.plot(full_path[:, 0], full_path[:, 1], style, color=color, linewidth=2, label=label)
    ax.scatter(*env.get_start(), s=85, color="green", label="Start", zorder=5)
    ax.scatter(*env.get_goal(), s=85, color="red", label="Goal", zorder=5)
    ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="X (m)", ylabel="Y (m)",
           title="Learned route and classical optimization refinements")
    ax.set_aspect("equal")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "hybrid_route_comparison.png", dpi=160)
    plt.close(fig)
    if save_data["preferred_demo_method"]:
        write_animated_3d_demo(
            env, final, save_data["preferred_demo_method"],
            output_dir / "animated_3d_demo.html",
        )
    (output_dir / "hybrid_comparison.json").write_text(json.dumps(save_data, indent=2), encoding="utf-8")
    print(json.dumps(save_data, indent=2))


if __name__ == "__main__":
    main()
