"""Generate a self-contained 3D replay with a modeled quadcopter and telemetry.

The positions are chosen by the trained Q-learning policy. The browser replays
them; this is NOT an interactive flight physics simulator or a real-time sensor.
"""

import argparse
import html
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError as exc:
    raise SystemExit("Install the project's requirements first: pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.environment import Environment2D


def cylinder(center, radius, height, color, opacity, name):
    angle = np.linspace(0, 2 * np.pi, 50)
    heights = np.linspace(0, height, 5)
    aa, zz = np.meshgrid(angle, heights)
    return go.Surface(
        x=center[0] + radius * np.cos(aa),
        y=center[1] + radius * np.sin(aa), z=zz,
        surfacecolor=np.zeros_like(zz),
        colorscale=[[0, color], [1, color]], showscale=False,
        opacity=opacity, hoverinfo="skip", name=name,
    )


def drone_outline(x, y, altitude, direction, phase):
    """Four motors, spinning blades, crossing arms and a forward nose."""
    unit = np.array([np.cos(direction), np.sin(direction)])
    side = np.array([-unit[1], unit[0]])
    motors = [np.array([x, y]) + sx * .22 * unit + sy * .22 * side
              for sx in (-1, 1) for sy in (-1, 1)]
    xs, ys, zs = [], [], []

    def segment(points, z):
        for px, py in points:
            xs.append(float(px))
            ys.append(float(py))
            zs.append(float(z))
        xs.append(None)
        ys.append(None)
        zs.append(None)

    for motor in motors:
        segment([[x, y], motor], altitude)
        t = np.linspace(0, 2 * np.pi, 17)
        ring = np.column_stack((motor[0] + .10 * np.cos(t), motor[1] + .10 * np.sin(t)))
        segment(ring, altitude + .035)
        blade = np.array([math.cos(phase), math.sin(phase)]) * .085
        segment([motor - blade, motor + blade], altitude + .05)

    segment([np.array([x, y]) + .12 * unit - .07 * side,
             np.array([x, y]) + .28 * unit,
             np.array([x, y]) + .12 * unit + .07 * side], altitude + .06)
    return xs, ys, zs


def interpolate_route(route, per_segment=9):
    samples = [route[0]]
    distances = [0.0]
    for start, end in zip(route[:-1], route[1:]):
        distance = float(np.linalg.norm(end - start))
        for fraction in np.linspace(1 / per_segment, 1, per_segment):
            samples.append(start + (end - start) * fraction)
            distances.append(distances[-1] + distance / per_segment)
    return np.asarray(samples), np.asarray(distances)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight-log", type=Path, default=ROOT / "results" / "autonomous" / "scenario_0" / "flight_log.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "autonomous" / "scenario_0" / "autopilot_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "autonomous" / "scenario_0" / "autonomous_3d_demo.html")
    parser.add_argument("--frames-per-step", type=int, default=8)
    args = parser.parse_args()
    if args.frames_per_step < 1:
        parser.error("--frames-per-step must be positive")

    log = json.loads(args.flight_log.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    env = Environment2D(log["scenario"])
    path = np.asarray(log["autonomous_path"], dtype=float)
    guidance = np.asarray(log["reference"], dtype=float)
    points, traveled = interpolate_route(path, per_segment=args.frames_per_step)
    altitude = 1.45
    power_w = summary["total_energy_Wh"] * 3600 / summary["total_time_s"]
    speed = env.drone_params["max_speed_ms"]

    figure = go.Figure()
    figure.add_trace(go.Surface(
        x=np.array([[0, 10], [0, 10]]),
        y=np.array([[0, 0], [10, 10]]), z=np.zeros((2, 2)),
        surfacecolor=np.zeros((2, 2)), colorscale=[[0, "#142a42"], [1, "#142a42"]],
        opacity=.93, showscale=False, hoverinfo="skip", name="Ground",
    ))
    for obstacle in env.get_obstacles():
        c = obstacle["center"]
        figure.add_trace(cylinder(c, obstacle["radius"], 2.4, "#475569", .9, "No-fly obstacle"))
        figure.add_trace(cylinder(c, env.effective_obstacle_radius(obstacle), 2.5,
                                  "#fb923c", .14, "Safety zone"))

    figure.add_trace(go.Scatter3d(x=guidance[:, 0], y=guidance[:, 1],
                                  z=np.full(len(guidance), altitude), mode="lines",
                                  name="GD optimized reference", line=dict(color="#4ade80", width=7, dash="dash")))
    figure.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1],
                                  z=np.full(len(path), altitude), mode="lines",
                                  name="Learned Q-learning route", line=dict(color="#60a5fa", width=4, dash="dot")))
    figure.add_trace(go.Scatter3d(x=[path[0, 0], path[-1, 0]],
                                  y=[path[0, 1], path[-1, 1]], z=[altitude, altitude],
                                  mode="markers+text", text=["START", "GOAL"],
                                  textposition="top center", name="Mission waypoints",
                                  marker=dict(size=8, color=["#22c55e", "#fb7185"])))

    trail_index = len(figure.data)
    figure.add_trace(go.Scatter3d(x=[points[0, 0]], y=[points[0, 1]], z=[altitude],
                                  mode="lines", name="Actual flight progress",
                                  line=dict(color="#facc15", width=11)))
    model_index = len(figure.data)
    xx, yy, zz = drone_outline(*points[0], altitude, 0.0, 0.0)
    figure.add_trace(go.Scatter3d(x=xx, y=yy, z=zz, mode="lines",
                                  line=dict(color="#22d3ee", width=10), name="Quadcopter model"))
    center_index = len(figure.data)
    figure.add_trace(go.Scatter3d(x=[points[0, 0]], y=[points[0, 1]], z=[altitude],
                                  mode="markers", marker=dict(size=7, color="#f8fafc", symbol="diamond"),
                                  name="Flight controller"))

    frames = []
    for i, point in enumerate(points):
        heading_step = points[min(i + 1, len(points) - 1)] - points[max(0, i - 1)]
        heading = math.atan2(heading_step[1], heading_step[0])
        outline = drone_outline(*point, altitude, heading, i * .9)
        elapsed_s = traveled[i] / speed
        energy_Wh = power_w * elapsed_s / 3600
        soc = 100 * (env.drone_params["initial_soc"] - energy_Wh / env.drone_params["battery_capacity_Wh"])
        telemetry_title = (f"<b>AUTONOMOUS FLIGHT</b> &nbsp; | &nbsp; "
                           f"Time {elapsed_s:.2f}s / {env.max_time_s:.1f}s &nbsp; | &nbsp; "
                           f"Distance {traveled[i]:.2f}m &nbsp; | &nbsp; "
                           f"Energy {energy_Wh * 1000:.3f}mWh &nbsp; | &nbsp; "
                           f"Battery {soc:.3f}%")
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=points[:i + 1, 0], y=points[:i + 1, 1],
                             z=np.full(i + 1, altitude), mode="lines",
                             line=dict(color="#facc15", width=11)),
                go.Scatter3d(x=outline[0], y=outline[1], z=outline[2],
                             mode="lines", line=dict(color="#22d3ee", width=10)),
                go.Scatter3d(x=[point[0]], y=[point[1]], z=[altitude],
                             mode="markers", marker=dict(size=7, color="#f8fafc", symbol="diamond")),
            ], traces=[trail_index, model_index, center_index],
            layout=go.Layout(title=dict(text=telemetry_title, x=.5, font=dict(size=17))),
            name=str(i),
        ))
    figure.frames = frames

    figure.update_layout(
        template="plotly_dark", paper_bgcolor="#08111f", plot_bgcolor="#08111f",
        title=dict(text="<b>AUTONOMOUS FLIGHT</b> — Press Play", x=.5, font=dict(size=19)),
        height=740,
        margin=dict(l=5, r=5, t=85, b=10),
        legend=dict(bgcolor="rgba(8,17,31,0.64)", x=.02, y=.97, font=dict(size=11)),
        scene=dict(
            xaxis=dict(range=[0, 10], title="X (m)", gridcolor="#263d59", backgroundcolor="#0d1c2e"),
            yaxis=dict(range=[0, 10], title="Y (m)", gridcolor="#263d59", backgroundcolor="#0d1c2e"),
            zaxis=dict(range=[0, 3], title="Fixed height (m)", gridcolor="#263d59", backgroundcolor="#0d1c2e"),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=.36),
            camera=dict(eye=dict(x=1.5, y=-1.6, z=1.3)),
        ),
        updatemenus=[dict(type="buttons", showactive=False, x=.38, y=1.08,
                          xanchor="left", yanchor="top", direction="right", buttons=[
            dict(label="▶ Fly", method="animate", args=[None, {
                "frame": {"duration": 75, "redraw": True},
                "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate",
            }]),
            dict(label="⏸ Pause", method="animate", args=[[None], {
                "frame": {"duration": 0, "redraw": False}, "mode": "immediate",
            }]),
            dict(label="↺ Restart", method="animate", args=[["0"], {
                "frame": {"duration": 0, "redraw": True}, "mode": "immediate",
            }]),
        ])],
        sliders=[dict(steps=[dict(label=str(int(i / args.frames_per_step)), method="animate",
                                  args=[[str(i)], {"frame": {"duration": 0, "redraw": True},
                                                   "mode": "immediate"}])
                             for i in range(len(points))],
                      visible=True, x=.12, y=.01, len=.76,
                      currentvalue=dict(prefix="Simulation frame "))],
    )

    chart = figure.to_html(full_html=False, include_plotlyjs=True, auto_play=False,
                           config={"responsive": True, "displaylogo": False})
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Optimization-Guided Autonomous Drone | Offline Demo</title>
<style>
body {{margin:0;background:#06101d;color:#e7f3ff;font:16px system-ui,-apple-system,Segoe UI,Arial}}
.shell {{max-width:1500px;margin:auto;padding:20px 24px}}
.heading {{display:flex;flex-wrap:wrap;justify-content:space-between;gap:18px;align-items:center}}
.tag {{color:#67e8f9;font-weight:800;letter-spacing:.16em;font-size:12px}}
h1 {{font-size:clamp(26px,3.6vw,44px);margin:8px 0}}
.subtitle {{color:#9ab2ca;margin:0 0 16px}}
.status {{border:1px solid #34d399;color:#a7f3d0;background:#064e3b;border-radius:30px;padding:9px 14px;font-size:13px}}
.cards {{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px;margin:14px 0}}
.card {{border:1px solid #213a55;background:#0d1c2e;border-radius:16px;padding:17px}}
.card small {{display:block;color:#9ab2ca;letter-spacing:.05em;font-size:12px}}
.card strong {{display:block;font-size:25px;margin-top:7px;color:#f8fafc}}
.plot {{border:1px solid #213a55;border-radius:18px;overflow:hidden;background:#08111f}}
.help {{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-top:16px;color:#c0d3e7}}
.help section {{padding:16px;border-radius:14px;border:1px solid #213a55;background:#0d1c2e}}
.help strong {{color:#7dd3fc}} p {{line-height:1.5;margin:9px 0 0}} 
</style></head><body><main class="shell">
<div class="heading"><div><div class="tag">OPTIMIZATION + REINFORCEMENT LEARNING</div>
<h1>Autonomous Drone Flight Simulator</h1>
<p class="subtitle">Trained Q-learning actions · Gradient Descent route guidance · 3D quadcopter replay</p></div>
<span class="status">✓ Feasible flight · Scenario {html.escape(str(summary['scenario_id']))}</span></div>
<div class="cards">
<div class="card"><small>LEARNED FLIGHT DISTANCE</small><strong>{summary['total_distance_m']:.2f} m</strong></div>
<div class="card"><small>ESTIMATED FLIGHT TIME</small><strong>{summary['total_time_s']:.2f} s</strong></div>
<div class="card"><small>ENERGY CONSUMED</small><strong>{summary['total_energy_Wh'] * 1000:.3f} mWh</strong></div>
<div class="card"><small>MIN. BUFFERED CLEARANCE</small><strong>{summary['minimum_buffered_clearance_m']:.3f} m</strong></div>
</div><div class="plot">{chart}</div>
<div class="help"><section><strong>How to use</strong><p>Click <b>▶ Fly</b>, pause, or drag the simulation slider. Drag the 3D scene to rotate; scroll to zoom. The cyan quadcopter follows actions produced by the trained Q-table.</p></section>
<section><strong>What the colors mean</strong><p>Green dashed = optimization-generated guidance. Blue dotted = learned policy route. Yellow = flight progress. Orange shells = restricted obstacle safety buffers.</p></section>
<section><strong>Scientific limits</strong><p>Offline 3D replay of a <b>2D, fixed-altitude</b> trained policy, not a physics-based simulator or real drone. Battery and time are estimated using the simplified constant-speed model. Only the tested map is demonstrated.</p></section></div>
</main></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Saved offline autonomous drone simulator: {args.output}")
    print(f"Frames: {len(frames)}, total estimated time: {summary['total_time_s']:.3f}s")


if __name__ == "__main__":
    main()
