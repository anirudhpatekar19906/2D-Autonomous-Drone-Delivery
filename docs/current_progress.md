# Mini-project status — 27 October 2026

## Completed and tested

- Original optimization modules reused; energy converted to Wh and reporting
  weights corrected. The optimizer import/output-folder issues were fixed.
- Generated 100 deterministic obstacle scenarios (`seed=2026`), while keeping
  the original 50-scenario dataset unchanged. Reproduction hashes matched.
- Completed Gradient Descent and Heavy-Ball comparisons on all 100 scenarios,
  3,000 maximum iterations each, with the same straight-line initialization.
- Built a genuine tabular Q-learning algorithm with eight movement actions,
  a reward signal, an epsilon-greedy policy, and Q-value updates.
- Saved training curves and the learned Q-table for scenario 0.
- Evaluated ten independently trained Q-learning agents on separate maps.
- Fed the learned route into the existing optimizers, checked safety again,
  and produced a fixed-altitude animated 3D replay.
- All five regression tests passed.

## Measured 100-scenario optimization baseline

| Metric | Gradient Descent | Heavy-Ball |
| --- | ---: | ---: |
| Exact feasible final routes | 54 / 100 | 77 / 100 |
| Exact feasibility rate | 54% | 77% |
| Mean normalized objective (including infeasible routes) | 0.31239 | 0.33707 |
| Mean iterations | 2919.66 | 2954.57 |
| Mean runtime (seconds, machine-specific) | 4.42 | 4.44 |

The mean objective above is **not an apples-to-apples ranking of feasible
routes** because it includes infeasible trajectories. Feasibility must be
evaluated separately. Both optimizers still have many runs that reach the
iteration limit; the baseline needs reliability work.

## Measured RL results

- Scenario 0: trained for 3,000 episodes. The greedy policy reached the goal
  with a 7.118 m fully feasible route.
- On ten **independently trained** maps, nine greedy policies reached the goal
  and were fully feasible. One did not converge in 3,000 episodes.
- These measurements **do not** test one fixed Q-table on unseen maps.
- On scenario 0, GD refinement reduced the route to 6.800 m with additional
  clearance. Heavy-Ball produced 6.535 m but passed very close to the
  configured safety boundary; the replay therefore selects GD conservatively.

## Still pending before submission

1. Improve trajectory-optimization feasibility beyond the current 54% / 77%
   benchmark, or clearly present failed scenarios and a safe fallback.
2. Decide whether any additional RL robustness experiments are required.
3. Polish the final presentation with measured outcomes. Replace the original
   PPT's proposed DQN slides with the **implemented Q-learning** method.
4. Rehearse a demo, check dependencies on the presentation laptop, and keep
   screenshots/video as backup.

## Strict claim boundary

No neural-network DQN, unseen-map generalization, genuine 3D path planning,
or real drone physics is claimed. The 3D view is fixed-altitude playback of
a planned 2D path.
