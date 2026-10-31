# Drone Trajectory Optimization + Tabular Q-Learning (Mini Project)

This project implements constrained trajectory optimization and a **genuine tabular Q-learning demonstration** for a 2D drone delivery environment. Gradient Descent and Heavy-Ball are classical optimization methods. Q-learning is the reinforcement-learning component; **DQN is not implemented**.

## Reproduce the October 2026 mini-project results

From the project root (Python 3.10+):

```bash
pip install -r requirements.txt
python scripts/generate_dataset.py --count 100 --seed 2026
python scripts/train_q_learning.py --episodes 3000 --scenario-id 0 --seed 2026
python scripts/evaluate_rl_scenarios.py --count 10 --episodes 3000
python scripts/refine_learned_route.py --max-iter 1500
python -m unittest discover -s tests -v
python scripts/run_gd_hd_comparison.py --dataset data/drone_optimization_dataset_100.json --results-dir results_100 --max-iter 3000
```

- The original `data/drone_optimization_dataset.json` (50 scenarios) remains unchanged.
- The seeded 100-map benchmark is saved as `data/drone_optimization_dataset_100.json`.
- Q-learning writes its Q-table, reward curve, route plot, and metrics under `results/rl/scenario_0/`.
- Independent per-map Q-learning experiments are stored in `results/rl/multi_scenario_evaluation.json` and `.csv`.
- Hybrid route comparisons and an animated fixed-altitude Plotly replay are under `results/hybrid/`.
- The 100-map optimization study writes comparison CSV/JSON and representative plots under `results_100/`.

**Important:** The older optimizer files already present under
`results/metrics/` came from the original 50-map version, before the energy
unit and objective-weight corrections. They are retained for comparison only.
**Use the new `results_100/metrics/` results for final claims**, once the full
benchmark finishes successfully.

**Interpretation:** A Q-table trained on one fixed map is not guaranteed to work on unseen maps. The 100 scenario benchmark measures classical optimizers, not RL generalization. The map is 2D; `3D_Visualization.py` shows a fixed-altitude visualization of 2D routes.

**Hybrid selection:** Both classical optimizers can start from the learned route. Their outputs are independently validated, and the animation prefers a feasible method with at least another 0.05 m of clearance beyond the configured no-fly-zone safety buffer when available. A faster or shorter route is not automatically preferred if it only grazes the safety boundary.

**Energy convention:** power is in watts and flight time in seconds, so estimated energy is `power_W * time_s / 3600` watt-hours. The simplified constant-speed model should not be represented as a validated physical drone/battery model.

## 🚀 Core Objectives
- **Optimization**: Use Gradient Descent and Heavy-Ball Momentum to find an efficient, safe route.
- **Constraints**: 
  - Avoid circular obstacles (No-Fly Zones).
  - Maintain safety buffers around obstacles.
  - Stay within defined map boundaries.
  - Ensure the total energy cost does not exceed the available battery.
- **Comparison**: Analyze the convergence speed and final route quality between first-order optimization methods.

## 🛠️ Project Structure
- `src/`: Core implementation
  - `environment.py`: Environment and drone parameter management.
  - `objective.py`: Time and Energy cost functions.
  - `constraints.py`: Mathematical formulation of constraints.
  - `derivatives.py`: Analytical gradients and numerical Hessian computation.
  - `gradient_descent.py`: GD optimizer implemented from scratch.
  - `heavy_ball.py`: Heavy-Ball optimizer implemented from scratch.
  - `feasibility.py`: Rigorous trajectory safety verification.
  - `visualization.py`: Plotting of routes and convergence metrics.
  - `experiments.py`: Main entry point to run the optimization study.
- `data/`: Scenario datasets; `scripts/generate_dataset.py` provides seeded generation.
- `results/`: Generated plots and logs (ignored by git).
- `docs/optimization_project_specification.md`: Detailed mathematical and academic requirements.

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 🏃 How to Run
For the main 100-scenario study, use the comparison command in the reproduction
section above. The older exploratory study can still be run with:
```bash
python src/experiments.py
```

## 📊 Mathematical Approach
The project uses a **Penalty Method** to transform a constrained optimization problem into an unconstrained one:
$$F_{penalty}(z) = F_{objective}(z) + \rho \sum \max(0, g_j(z))^2$$
Where $g_j(z)$ represents the constraints (NFZ, Battery, Workspace).

## Delivery priority modes

The optimizer supports three delivery modes. Time and energy are normalized by reference scales before weights are applied, so `alpha` and `beta` represent priorities rather than raw units:

| Mode | Weights | Priority |
|---|---|---|
| Normal / Battery Saving | `alpha = 0.2`, `beta = 0.8` | Energy |
| Express / Urgent Delivery | `alpha = 0.9`, `beta = 0.1` | Time |
| Balanced | `alpha = 0.5`, `beta = 0.5` | Equal priority |

Results are saved in `results/metrics/delivery_modes.json`.

Because the current model uses fixed speed and energy proportional to distance, all three modes can produce similar routes. A richer energy model or speed as a decision variable is required for visibly different speed-energy trade-offs.
#
