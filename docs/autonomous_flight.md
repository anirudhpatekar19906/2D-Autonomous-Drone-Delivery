# Optimization-guided autonomous flight: actual implementation

## Architecture

1. Choose one of the 100 generated 2D obstacle scenarios.
2. The existing Gradient Descent optimizer produces a feasible reference route
   (currently loaded from `results/hybrid/hybrid_comparison.json`).
3. `GuidedGridDroneEnvironment` discretizes the same map and pre-validates all
   eight possible movement directions against circular *buffered* obstacles.
4. A fresh tabular Q-learning agent trains by repeated simulated episodes.
   It is rewarded for reaching the goal and moving closer to the optimized
   guidance route. Invalid actions are penalized, as are longer movements.
5. A greedy rollout uses the Q-table to choose each action step by step. The
   learned route is **not** the original optimized polyline copied into the
   flight controller.
6. Independent feasibility checks cover endpoints, entire route segments,
   map boundaries, estimated mission time, and remaining battery reserve.
7. The 3D HTML *replays those decisions* with a modeled quadcopter, obstacle
   cylinders, safety envelopes and a flight-time/energy/battery HUD.

## Reproduce

```powershell
python scripts/refine_learned_route.py --max-iter 1500
python scripts/train_autonomous_drone.py --episodes 3500
python scripts/visualize_autonomous_drone.py
python -m unittest discover -s tests -v
start .\results\autonomous\scenario_0\autonomous_3d_demo.html
```

No deep neural network is required. `requirements.txt` supplies NumPy,
Matplotlib and Plotly.

## Measured results on scenario 0 (seed 2026)

- Optimized GD guidance route: **6.800 m**.
- Q-learning guided autonomous flight: **7.118 m**.
- Last 100 *training* episodes success: **100%** on the same map.
- Greedy policy reached the goal: **yes**.
- Estimated time: **1.499 s** against **3.0 s** mission limit.
- Estimated energy: **0.002467 Wh** against **70 Wh** usable capacity.
- Extra obstacle clearance beyond the already buffered boundaries: **0.0303 m**.
- Exact geometric/time/battery feasibility: **passed**.

The discrete flight trajectory is longer than its continuously optimized
reference. This is expected from coarse eight-direction grid actions and is
not a claim that RL outperformed GD. Both methods can be benchmarked honestly.

## Reward and constraints

Training uses Q-learning temporal-difference updates with epsilon-greedy
exploration. Reward components include progress toward goal, goal arrival,
invalid-action penalty, traveled-step cost, and deviation from the optimized
guidance route. The step-distance cost is a proxy for both time and energy.

The geometric safety gate rejects moves whose segments intersect an obstacle
plus its configured margin. The final exact feasibility checker validates
segments, points, battery reserve and the deadline. This demo has a fixed
altitude, 2D movement grid, constant-speed time model, and simplified power
model. Configured **acceleration and turn-rate limits are not simulated or
verified**, so the output is not flight-ready or physically optimal. It also
does not model wind, moving obstacles, real sensors, battery dynamics, or
unseen obstacle-map generalization.
