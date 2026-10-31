# Reinforcement Learning Extension: Tabular Q-learning

## What is implemented

The project has a genuine **tabular Q-learning** agent. DQN is a deep RL variant
that requires a neural network and is **not** implemented. The existing Gradient
Descent and Heavy-Ball code remains the classical optimization component.

## Environment and state

- Continuous scenario coordinates cover a 10 m by 10 m horizontal plane.
- RL discretizes that plane into a 0.5 m grid (21 by 21 nodes).
- The **state** is the drone's `(grid_x, grid_y)` index, for a fixed obstacle map.
- The **actions** are eight movements: four cardinal and four diagonal.
- A move is blocked if it leaves the map or crosses a circular No-Fly Zone
  plus the configured safety buffer. Collision/invalid moves receive a penalty.
- The original continuous start and goal are connected to nearby safe grid nodes.

## Reward and update

At each movement the agent receives

`reward = 2 * (distance_before - distance_after) - 0.10`.

Reaching the goal adds `+60`, while an invalid or unsafe move receives `-4`
and the drone stays in place. This reward encourages progress, but does not
guarantee an optimal or feasible solution.

The learning rule is:

`Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]`.

Default hyperparameters:

| Setting | Value |
| --- | ---: |
| Learning rate `alpha` | 0.25 |
| Discount `gamma` | 0.97 |
| Training episodes | 3,000 |
| Steps per episode | 160 |
| Initial exploration `epsilon` | 0.95 |
| Exploration decay | 0.997 per episode |
| Minimum exploration | 0.03 |
| Random seed | 2026 |

## Observed results and what they mean

On generated scenario 0, the deterministic greedy policy reached the goal,
and the continuous route passed the project's full feasibility check.
The route was about **7.12 m**, with estimated flight time **1.50 seconds**
under the simplified constant-speed model.

In a separate run, ten **independently trained** agents were evaluated on the
first ten generated maps. Nine of the ten greedy policies reached the goal and
passed the complete feasibility checker. One map did not converge within
3,000 episodes. This is **90% per-map success for separately trained agents**,
not a claim that one trained Q-table works on unseen scenarios.

The hybrid script can take an RL-generated waypoint route as input to Gradient
Descent and Heavy-Ball. It checks each refined route independently, and prefers
extra clearance when selecting the replay demonstration. Optimization can
make a learned route shorter but can also introduce an unsafe trajectory.

## Limitations to disclose in the viva

- This is a **2D** simulator; the 3D HTML replay uses a fixed altitude.
- The Q-table state does not encode arbitrary obstacle layouts, so a single
  trained table does not generalize to new maps without retraining.
- Reported energy is an estimate from a simplified constant-speed power model.
- Training success on one fixed map is not independent held-out performance.
- Neither a neural-network DQN nor real-world flight control is included.

## Reproduce

```bash
python scripts/train_q_learning.py --episodes 3000 --scenario-id 0 --seed 2026
python scripts/evaluate_rl_scenarios.py --count 10 --episodes 3000 --seed 2026
python scripts/refine_learned_route.py --max-iter 1500
```
