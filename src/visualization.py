import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    """
    Handles plotting of trajectories and convergence curves.
    """
    def __init__(self, environment):
        self.env = environment

    def plot_trajectory(self, z, title="Trajectory", save_path=None):
        """
        Plots the trajectory including start, goal, and obstacles.
        """
        start = self.env.get_start()
        goal = self.env.get_goal()
        obstacles = self.env.get_obstacles()
        waypoints = z.reshape(-1, 2)
        full_path = np.vstack([start, waypoints, goal])

        plt.figure(figsize=(8, 8))

        # Plot Start and Goal
        plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
        plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

        # Plot Path
        plt.plot(full_path[:, 0], full_path[:, 1], 'b-o', label='Optimized Path')

        # Plot Obstacles
        for obs in obstacles:
            center = obs['center']
            radius = obs['radius']
            safe_radius = self.env.effective_obstacle_radius(obs)
            plt.gca().add_patch(plt.Circle(center, radius, color='gray', alpha=0.5,
                                           label='No-fly zone' if obs is obstacles[0] else None))
            plt.gca().add_patch(plt.Circle(center, safe_radius, color='orange',
                                           fill=False, linestyle='--', alpha=0.8,
                                           label='Safety buffer' if obs is obstacles[0] else None))

        plt.xlim(self.env.x_min, self.env.x_max)
        plt.ylim(self.env.y_min, self.env.y_max)
        plt.gca().set_aspect('equal')
        plt.title(title)
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_convergence(self, gd_history, hb_history, save_path=None):
        """
        Compares GD and Heavy-Ball convergence.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(gd_history, label='Gradient Descent')
        plt.plot(hb_history, label='Heavy-Ball')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value (log scale)')
        plt.title('Convergence Comparison')
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_diagnostics(self, gd, hb, output_dir, title_prefix=""):
        """Write the convergence, gradient, and constraint diagnostics required by the study."""
        output_dir = str(output_dir)
        prefix = f"{title_prefix} " if title_prefix else ""
        self._plot_series(
            [gd.history], ["Gradient Descent"], "Iteration", "Penalized objective",
            f"{prefix}Gradient Descent convergence", f"{output_dir}/gd_convergence.png", log_y=True,
        )
        self._plot_series(
            [hb.history], ["Heavy-Ball"], "Iteration", "Penalized objective",
            f"{prefix}Heavy-Ball convergence", f"{output_dir}/hb_convergence.png", log_y=True,
        )
        self._plot_series(
            [gd.history, hb.history], ["Gradient Descent", "Heavy-Ball"],
            "Iteration", "Penalized objective", f"{prefix}Objective convergence",
            f"{output_dir}/convergence.png", log_y=True,
        )
        self._plot_series(
            [gd.grad_norm_history, hb.grad_norm_history], ["Gradient Descent", "Heavy-Ball"],
            "Iteration", "Penalized gradient norm", f"{prefix}Gradient norm convergence",
            f"{output_dir}/gradient_norm.png", log_y=True,
        )
        self._plot_series(
            [gd.constraint_violation_history, hb.constraint_violation_history],
            ["Gradient Descent", "Heavy-Ball"], "Iteration", "Maximum sampled constraint violation",
            f"{prefix}Constraint violation convergence", f"{output_dir}/constraint_violation.png", log_y=True,
        )

    def plot_trajectory_comparison(self, initial, gd, hb, save_path):
        """Plot initial, GD, and Heavy-Ball routes for one representative scenario."""
        start = self.env.get_start()
        goal = self.env.get_goal()
        plt.figure(figsize=(8, 8))
        plt.plot(start[0], start[1], "go", markersize=10, label="Start")
        plt.plot(goal[0], goal[1], "ro", markersize=10, label="Goal")
        for values, label, style in (
            (initial, "Initial route", "k--"),
            (gd, "Gradient Descent", "b-o"),
            (hb, "Heavy-Ball", "m-o"),
        ):
            path = np.vstack([start, np.asarray(values).reshape(-1, 2), goal])
            plt.plot(path[:, 0], path[:, 1], style, label=label)
        for obstacle in self.env.get_obstacles():
            center = obstacle["center"]
            safe_radius = self.env.effective_obstacle_radius(obstacle)
            plt.gca().add_patch(plt.Circle(center, obstacle["radius"], color="gray", alpha=0.5))
            plt.gca().add_patch(plt.Circle(center, safe_radius, color="orange", fill=False, linestyle="--"))
        plt.xlim(self.env.x_min, self.env.x_max)
        plt.ylim(self.env.y_min, self.env.y_max)
        plt.gca().set_aspect("equal")
        plt.title("Representative Scenario 0 Trajectory Comparison")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def _plot_series(series, labels, xlabel, ylabel, title, save_path, log_y=False):
        plt.figure(figsize=(10, 6))
        for values, label in zip(series, labels):
            plt.plot(values, label=label)
        if log_y:
            plt.yscale("log")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_waypoint_study(records, save_path):
        plt.figure(figsize=(8, 5))
        for method in ("Gradient Descent", "Heavy-Ball"):
            rows = [row for row in records if row["method"] == method]
            plt.plot([row["waypoints"] for row in rows], [row["objective"] for row in rows], "o-", label=method)
        plt.xlabel("Number of intermediate waypoints")
        plt.ylabel("Final objective")
        plt.title("Objective versus waypoint count")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
