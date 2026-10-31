import numpy as np
import matplotlib.pyplot as plt
import json
import random
import os
import argparse
from pathlib import Path


# ============================================================
# 1. FIXED DRONE PARAMETERS
# ============================================================

DRONE_PARAMS = {

    "mass_kg": 1.5,
    "weight_N": 14.7,
    # Fixed vehicle configuration; scenarios vary in geometry, not vehicle.
    "max_speed_ms": 4.75,
    "max_accel_ms2": 1.9,
    "max_turn_rate_rad_s": 1.5,
    "battery_capacity_Wh": 100.0,
    "initial_soc": 0.9,
    "reserve_soc": 0.2,
    "power_model": {
        "c0": 5.0,
        "c1": 0.1,
        "c2": 0.02
    }
}


# ============================================================
# 2. SAFETY PARAMETERS
# ============================================================

# Extra distance maintained around every No-Fly Zone
SAFETY_MARGIN = 0.5

# Small additional clearance so that Start/Goal
# do not touch the safety boundary
EXTRA_CLEARANCE = 0.2


# Effective distance that Start/Goal must maintain
# from the center of an obstacle
TOTAL_SAFETY_CLEARANCE = SAFETY_MARGIN + EXTRA_CLEARANCE


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def distance_to_obstacle(point, obstacle):
    """
    Calculates Euclidean distance between a point
    and the center of an obstacle.
    """

    point = np.array(point)
    center = np.array(obstacle["center"])

    return np.linalg.norm(point - center)


# ------------------------------------------------------------
# Check if point is inside actual No-Fly Zone
# ------------------------------------------------------------

def is_colliding(point, obstacles):
    """
    Checks whether a point is inside any actual No-Fly Zone.
    """

    for obs in obstacles:

        distance = distance_to_obstacle(point, obs)

        if distance < obs["radius"]:
            return True

    return False


# ------------------------------------------------------------
# Check whether point is safely outside NFZ
# ------------------------------------------------------------

def is_point_safe(point, obstacles):
    """
    Checks whether a point is outside the No-Fly Zone
    AND outside its safety buffer.

    The point must be strictly farther than:

        radius + safety margin + extra clearance
    """

    for obs in obstacles:

        distance = distance_to_obstacle(point, obs)

        required_distance = (
            obs["radius"]
            + SAFETY_MARGIN
            + EXTRA_CLEARANCE
        )

        if distance <= required_distance:
            return False

    return True


# ------------------------------------------------------------
# Check whether straight path is blocked
# ------------------------------------------------------------

def is_line_blocked(start, goal, obstacles):
    """
    Checks whether the straight path from Start to Goal
    intersects an obstacle or its safety buffer.

    Safety margin is included because the drone should
    not plan through the orange safety region.
    """

    start = np.array(start)
    goal = np.array(goal)

    line_vec = goal - start
    line_len_squared = np.dot(line_vec, line_vec)

    if line_len_squared == 0:
        return False

    for obs in obstacles:

        center = np.array(obs["center"])

        # Projection of obstacle center onto line segment
        t = np.dot(center - start, line_vec) / line_len_squared

        # Keep projection inside Start-Goal segment
        t = max(0.0, min(1.0, t))

        # Closest point on line segment
        closest_point = start + t * line_vec

        # Distance from obstacle center to path
        dist_to_line = np.linalg.norm(
            center - closest_point
        )

        # Consider safety margin as part of restricted area
        effective_radius = (
            obs["radius"]
            + SAFETY_MARGIN
        )

        if dist_to_line < effective_radius:
            return True

    return False


# ------------------------------------------------------------
# Check obstacles are not too close to Start/Goal
# ------------------------------------------------------------

def endpoints_are_safe(start, goal, obstacles):

    return (
        is_point_safe(start, obstacles)
        and
        is_point_safe(goal, obstacles)
    )


# ============================================================
# 4. SCENARIO GENERATOR
# ============================================================

def generate_single_scenario(scenario_id):

    MAX_SCENARIO_ATTEMPTS = 500

    for scenario_attempt in range(MAX_SCENARIO_ATTEMPTS):

        # ----------------------------------------------------
        # Generate Start and Goal
        # ----------------------------------------------------

        start = np.random.uniform(
            0, 10, size=2
        )

        goal = np.random.uniform(
            0, 10, size=2
        )

        # Start and Goal should not be too close
        if np.linalg.norm(start - goal) <= 3.0:
            continue


        # ----------------------------------------------------
        # Number of obstacles
        # ----------------------------------------------------

        num_obstacles = random.randint(2, 3)

        obstacles = []


        # ----------------------------------------------------
        # FIRST OBSTACLE
        #
        # Must block direct Start → Goal path
        # ----------------------------------------------------

        first_obstacle_found = False

        for _ in range(200):

            center = np.random.uniform(
                1.0, 9.0, size=2
            )

            radius = np.random.uniform(
                0.8, 1.5
            )

            candidate = {
                "center": center.tolist(),
                "radius": float(radius)
            }

            # Candidate must block the direct path
            if is_line_blocked(
                start,
                goal,
                [candidate]
            ):

                obstacles.append(candidate)

                first_obstacle_found = True

                break


        # If suitable first obstacle was not found,
        # generate another scenario
        if not first_obstacle_found:
            continue


        # ----------------------------------------------------
        # REMAINING OBSTACLES
        # ----------------------------------------------------

        for _ in range(num_obstacles - 1):

            obstacle_found = False

            for _ in range(200):

                center = np.random.uniform(
                    1.0, 9.0, size=2
                )

                radius = np.random.uniform(
                    0.5, 1.5
                )

                candidate = {
                    "center": center.tolist(),
                    "radius": float(radius)
                }


                # ------------------------------------------------
                # Do not place obstacle too close to Start/Goal
                # ------------------------------------------------

                if not is_point_safe(
                    start,
                    [candidate]
                ):
                    continue

                if not is_point_safe(
                    goal,
                    [candidate]
                ):
                    continue


                # ------------------------------------------------
                # Avoid excessive obstacle overlap
                # ------------------------------------------------

                overlap = False

                for existing in obstacles:

                    center_distance = np.linalg.norm(
                        np.array(candidate["center"])
                        -
                        np.array(existing["center"])
                    )

                    minimum_distance = (
                        candidate["radius"]
                        +
                        existing["radius"]
                        +
                        SAFETY_MARGIN
                    )

                    if center_distance < minimum_distance:

                        overlap = True

                        break


                if overlap:
                    continue


                obstacles.append(candidate)

                obstacle_found = True

                break


            if not obstacle_found:
                break


        # ----------------------------------------------------
        # FINAL SAFETY VALIDATION
        # ----------------------------------------------------

        if not endpoints_are_safe(
            start,
            goal,
            obstacles
        ):
            continue


        # ----------------------------------------------------
        # Confirm direct path is blocked
        # ----------------------------------------------------

        if not is_line_blocked(
            start,
            goal,
            obstacles
        ):
            continue


        # ----------------------------------------------------
        # Scenario successfully generated
        # ----------------------------------------------------

        scenario = {

            "id": scenario_id,

            "start": start.tolist(),

            "goal": goal.tolist(),

            "obstacles": obstacles,

            "safety_parameters": {

                "safety_margin": SAFETY_MARGIN,

                "extra_clearance": EXTRA_CLEARANCE,

                "total_extra_clearance":
                    TOTAL_SAFETY_CLEARANCE

            },

            "drone_params": DRONE_PARAMS
        }


        return scenario


    # --------------------------------------------------------
    # If no valid scenario was generated
    # --------------------------------------------------------

    raise RuntimeError(
        f"Could not generate a valid scenario "
        f"for scenario {scenario_id}."
    )


# ============================================================
# 5. DATASET GENERATION
# ============================================================

def generate_dataset(num_scenarios=100, output_path=None):

    print(
        f"Generating {num_scenarios} scenarios..."
    )

    dataset = []


    for i in range(num_scenarios):

        scenario = generate_single_scenario(i)

        dataset.append(scenario)


    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------

    filename = Path(output_path) if output_path else (
        Path(__file__).parent.parent / "data" / f"drone_optimization_dataset_{num_scenarios}.json"
    )
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(
        filename,
        "w"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=4
        )


    print(
        f"\nDataset successfully saved to: {filename}"
    )

    print(
        f"Total scenarios: {len(dataset)}"
    )

    print(
        f"Safety Margin: {SAFETY_MARGIN} units"
    )

    print(
        f"Extra Clearance: {EXTRA_CLEARANCE} units"
    )


    return dataset


# ============================================================
# 6. VISUALIZATION
# ============================================================

def visualize_samples(
    dataset,
    num_samples=6
):

    num_samples = min(
        num_samples,
        len(dataset)
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10)
    )

    axes = axes.flatten()


    # Random scenarios
    sample_indices = random.sample(
        range(len(dataset)),
        num_samples
    )


    for ax, idx in zip(
        axes,
        sample_indices
    ):

        scenario = dataset[idx]

        start = np.array(
            scenario["start"]
        )

        goal = np.array(
            scenario["goal"]
        )

        obstacles = scenario["obstacles"]


        # ----------------------------------------------------
        # Start
        # ----------------------------------------------------

        ax.plot(
            start[0],
            start[1],
            "go",
            markersize=10,
            label="Start"
        )


        # ----------------------------------------------------
        # Goal
        # ----------------------------------------------------

        ax.plot(
            goal[0],
            goal[1],
            "ro",
            markersize=10,
            label="Goal"
        )


        # ----------------------------------------------------
        # Direct Start → Goal Path
        # ----------------------------------------------------

        ax.plot(
            [start[0], goal[0]],
            [start[1], goal[1]],
            "k--",
            alpha=0.35,
            label="Direct Path"
        )


        # ----------------------------------------------------
        # Draw obstacles
        # ----------------------------------------------------

        for obstacle_number, obs in enumerate(
            obstacles
        ):

            center = obs["center"]

            radius = obs["radius"]


            # -----------------------------------------------
            # Actual No-Fly Zone
            # -----------------------------------------------

            nfz = plt.Circle(
                center,
                radius,
                color="gray",
                alpha=0.55,
                label=(
                    "No-Fly Zone"
                    if obstacle_number == 0
                    else None
                )
            )

            ax.add_patch(nfz)


            # -----------------------------------------------
            # Safety Buffer
            # -----------------------------------------------

            safety_circle = plt.Circle(
                center,
                radius + SAFETY_MARGIN,
                color="orange",
                alpha=0.18,
                linestyle="--",
                label=(
                    "Safety Margin (0.5)"
                    if obstacle_number == 0
                    else None
                )
            )

            ax.add_patch(safety_circle)


        # ----------------------------------------------------
        # Plot settings
        # ----------------------------------------------------

        ax.set_xlim(-1, 11)

        ax.set_ylim(-1, 11)

        ax.set_aspect("equal")

        ax.set_xlabel(
            "X Position"
        )

        ax.set_ylabel(
            "Y Position"
        )

        ax.set_title(
            f"Scenario {scenario['id']}"
        )

        ax.grid(
            True,
            linestyle=":",
            alpha=0.5
        )


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    axes[0].legend(
        loc="upper right"
    )


    plt.suptitle(
        "Drone Navigation Environment\n"
        "No-Fly Zones + 0.5 Unit Safety Margin",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 7. RUN DATASET GENERATION
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate reproducible drone environments")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot", action="store_true", help="Display six sample maps")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    np.random.seed(args.seed)
    random.seed(args.seed)
    dataset = generate_dataset(args.count, args.output)
    if args.plot:
        visualize_samples(dataset, num_samples=6)


if __name__ == "__main__":
    main()
