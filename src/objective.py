import numpy as np


DELIVERY_MODES = {
    "normal": {
        "label": "Normal / Battery Saving",
        "alpha": 0.2,
        "beta": 0.8,
    },
    "express": {
        "label": "Express / Urgent Delivery",
        "alpha": 0.9,
        "beta": 0.1,
    },
    "balanced": {
        "label": "Balanced",
        "alpha": 0.5,
        "beta": 0.5,
    },
}


class Objective:
    """
    Calculates the cost associated with a drone trajectory.
    Main objective: Minimize Time and Energy.
    """
    def __init__(self, environment):
        self.env = environment
        self.drone = environment.get_drone_params()

        # Constants for calculation
        self.v_max = self.drone['max_speed_ms']

        # Power model coefficients: P = c0 + c1*v + c2*v^2
        self.c0 = self.drone['power_model']['c0']
        self.c1 = self.drone['power_model']['c1']
        self.c2 = self.drone['power_model']['c2']

        self.alpha = 1.0
        self.smoothness_weight = 0.0

        # Reference scales make seconds and watt-hours comparable in the
        # weighted objective instead of letting units determine priority.
        self.time_reference_s = max(self.env.max_time_s, 1e-12)
        self.energy_reference_Wh = max(self.env.usable_battery_Wh(), 1e-12)

    def total_distance(self, z):
        """
        z: decision vector [x1, y1, x2, y2, ..., xN, yN]
        """
        # Reconstruct the full trajectory: Start -> Waypoints -> Goal
        waypoints = z.reshape(-1, 2)
        full_path = np.vstack([self.env.get_start(), waypoints, self.env.get_goal()])

        # Calculate Euclidean distance between consecutive points
        diffs = np.diff(full_path, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)

    def time_cost(self, z):
        """
        Simplified time cost: Total Distance / Max Speed
        """
        dist = self.total_distance(z)
        return dist / self.v_max

    def energy_cost(self, z):
        """
        Simplified energy cost: (Power at max speed) * (Time at max speed)
        Power P = c0 + c1*v + c2*v^2
        Energy E = P * T / 3600 in watt-hours, because power is in watts
        and T is measured in seconds.
        """
        dist = self.total_distance(z)
        time = dist / self.v_max
        power = self.c0 + self.c1 * self.v_max + self.c2 * (self.v_max**2)
        return power * time / 3600.0

    def combined_objective(self, z, w_time=1.0, w_energy=1.0, w_smooth=0.0):
        """
        Normalized weighted sum of time and energy costs.
        """
        t_cost = self.time_cost(z) / self.time_reference_s
        e_cost = self.energy_cost(z) / self.energy_reference_Wh
        s_cost = self.smoothness_cost(z) if w_smooth > 0 else 0.0

        return w_time * t_cost + w_energy * e_cost + w_smooth * s_cost

    def mode_weights(self, mode="balanced"):
        """Return normalized time/energy weights for a delivery mode."""
        try:
            selected = DELIVERY_MODES[mode.lower()]
        except KeyError as error:
            valid = ", ".join(sorted(DELIVERY_MODES))
            raise ValueError(f"Unknown delivery mode {mode!r}; choose from {valid}") from error
        return selected["alpha"], selected["beta"]

    def smoothness_gradient(self, z):
        """Gradient of the squared second-difference smoothness term."""
        waypoints = z.reshape(-1, 2)
        full_path = np.vstack([self.env.get_start(), waypoints, self.env.get_goal()])
        gradient = np.zeros_like(full_path)
        for i in range(1, len(full_path) - 1):
            difference = full_path[i + 1] - 2 * full_path[i] + full_path[i - 1]
            gradient[i - 1] += 2 * difference
            gradient[i] -= 4 * difference
            gradient[i + 1] += 2 * difference
        return gradient[1:-1].flatten()

    def smoothness_cost(self, z):
        """
        Penalty for sharp turns.
        S(z) = sum ||P_{i+1} - 2P_i + P_{i-1}||^2
        """
        waypoints = z.reshape(-1, 2)
        full_path = np.vstack([self.env.get_start(), waypoints, self.env.get_goal()])

        smoothness = 0.0
        for i in range(1, len(full_path) - 1):
            # Second-order difference (curvature approximation)
            diff = full_path[i+1] - 2*full_path[i] + full_path[i-1]
            smoothness += np.sum(diff**2)

        return smoothness
