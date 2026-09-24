import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments import build_problem, run_method


def make_3d_path(start, goal, waypoints, flight_altitude=1.5):
    """Lift the project's 2D route into 3D at a constant flight altitude."""
    route = np.asarray(waypoints, dtype=float).reshape(-1, 2)
    full_path = np.vstack([start, route, goal])
    altitude = np.full(len(full_path), flight_altitude, dtype=float)
    return np.column_stack([full_path[:, 0], full_path[:, 1], altitude])


def add_cylinder(fig, scene, center, radius, height):
    """Add a translucent cylindrical no-fly zone to one Plotly 3D scene."""
    theta = np.linspace(0, 2 * np.pi, 48)
    z = np.linspace(0, height, 12)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = center[0] + radius * np.cos(theta_grid)
    y_grid = center[1] + radius * np.sin(theta_grid)
    fig.add_trace(
        go.Surface(x=x_grid, y=y_grid, z=z_grid, surfacecolor=np.zeros_like(z_grid), colorscale=[[0, "#34a853"], [1, "#34a853"]], opacity=0.28, showscale=False, hoverinfo="skip", name="No-fly zone", showlegend=False),
        row=scene[0], col=scene[1],
    )


def add_route(fig, scene, path, name, color, dash, showlegend):
    fig.add_trace(
        go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines+markers", name=name, legendgroup=name, showlegend=showlegend, line={"color": color, "width": 7, "dash": dash}, marker={"size": 5, "color": color, "symbol": "circle"}, hovertemplate="%{x:.2f} m, %{y:.2f} m, %{z:.2f} m<extra>" + name + "</extra>"),
        row=scene[0], col=scene[1],
    )


def build_interactive_dashboard(scenes, save_path):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=[f"Scenario {item['id']}" for item in scenes], horizontal_spacing=0.03)

    for column, item in enumerate(scenes, start=1):
        environment = item["environment"]
        start = np.asarray(environment.get_start(), dtype=float)
        goal = np.asarray(environment.get_goal(), dtype=float)

        for obstacle in environment.get_obstacles():
            center = np.asarray(obstacle["center"], dtype=float)
            radius = float(obstacle["radius"]) + float(environment.safety_params.get("safety_margin", 0.0)) + float(environment.safety_params.get("extra_clearance", 0.0))
            add_cylinder(fig, (1, column), center, radius, 3.0)

        initial_path = make_3d_path(start, goal, item["initial"])
        gd_path = make_3d_path(start, goal, item["gd"])
        hb_path = make_3d_path(start, goal, item["hb"])
        add_route(fig, (1, column), initial_path, "Initial path", "#8c8c8c", "dash", column == 1)
        add_route(fig, (1, column), gd_path, "Gradient Descent", "#1769e0", "solid", column == 1)
        add_route(fig, (1, column), hb_path, "Heavy-Ball", "#f28e2b", "solid", column == 1)

        for point, name, color, symbol in ((gd_path[0], "Start", "#d95f02", "circle"), (gd_path[-1], "Goal", "#e6ab02", "diamond")):
            fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[0], mode="markers+text", text=[name], textposition="top center", name=name, legendgroup=name, showlegend=column == 1, marker={"size": 9, "color": color, "symbol": symbol, "line": {"color": "white", "width": 2}}, hovertemplate=f"{name}<br>x=%{{x:.2f}} m<br>y=%{{y:.2f}} m<extra></extra>"), row=1, col=column)

        fig.update_scenes(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Flight altitude (m)", xaxis={"range": [environment.x_min, environment.x_max], "backgroundcolor": "#f4f7fb"}, yaxis={"range": [environment.y_min, environment.y_max], "backgroundcolor": "#f4f7fb"}, zaxis={"range": [0, 3], "backgroundcolor": "#f4f7fb"}, aspectmode="cube", camera={"eye": {"x": 1.55, "y": 1.55, "z": 1.15}}, row=1, col=column)

    fig.update_layout(title="Interactive 3D Drone Trajectory Optimization | Three Representative Scenarios", template="plotly_white", height=760, margin={"l": 0, "r": 0, "t": 85, "b": 20}, legend={"orientation": "h", "y": -0.02, "x": 0.02}, hovermode="closest")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path, include_plotlyjs=True, full_html=True, config={"displaylogo": False, "responsive": True})
    print(f"Saved interactive 3D dashboard to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Create an interactive 3D dashboard for two randomly selected drone scenarios.")
    parser.add_argument("--scenarios", nargs=2, type=int, help="Optional scenario IDs; otherwise two are selected randomly")
    parser.add_argument("--seed", type=int, help="Optional random seed for repeatable scenario selection")
    parser.add_argument("--save", type=str, default="results/trajectories/interactive_3d_trajectories.html", help="Output HTML path")
    args = parser.parse_args()

    dataset_path = ROOT / "data" / "drone_optimization_dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.seed is not None:
        random.seed(args.seed)
    scenario_ids = args.scenarios or random.sample([int(item["id"]) for item in dataset], 2)

    scenes = []
    for scenario_id in scenario_ids:
        scenario = next(item for item in dataset if int(item["id"]) == scenario_id)
        environment, objective, constraints, derivatives, feasibility, initial = build_problem(scenario, 5)
        gd_solution, _, _ = run_method(environment, objective, constraints, derivatives, feasibility, initial.copy(), "Gradient Descent", learning_rate=0.01, rho=1000.0, max_iter=3000, tol=1e-8, mode="balanced")
        hb_solution, _, _ = run_method(environment, objective, constraints, derivatives, feasibility, initial.copy(), "Heavy-Ball", learning_rate=0.01, momentum=0.9, rho=1000.0, max_iter=3000, tol=1e-8, mode="balanced")
        scenes.append({"id": scenario_id, "environment": environment, "initial": initial, "gd": gd_solution, "hb": hb_solution})

    build_interactive_dashboard(scenes, args.save)


if __name__ == "__main__":
    main()
