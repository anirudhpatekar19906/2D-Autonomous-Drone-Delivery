import numpy as np
import time
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """
    Abstract Base Class for Drone Trajectory Optimizers.
    Handles shared state, workspace clipping, stopping criteria,
    and the penalty scheduling logic.
    """
    def __init__(self, env, objective, constraints, derivatives,
                 learning_rate=0.01, max_iter=1000, tol=1e-5, rho=100.0,
                 max_step=0.25, rho_schedule=None):
        self.env = env
        self.obj = objective
        self.cons = constraints
        self.deriv = derivatives
        self.eta = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.rho = rho
        self.max_step = max_step
        self.rho_schedule = rho_schedule

        # Default weights for the objective function
        self.w_time = 1.0
        self.w_energy = 1.0
        self.w_smooth = 0.0

        # State tracking
        self.history = []
        self.grad_norm_history = []
        self.feasibility_history = []
        self.constraint_violation_history = []
        self.runtime = 0
        self.stopping_reason = None

    def _reset_state(self):
        """Resets optimizer history and state for a new fit run."""
        self.history = []
        self.grad_norm_history = []
        self.feasibility_history = []
        self.constraint_violation_history = []
        self.stopping_reason = "maximum_iterations"

    def _clip_to_workspace(self, z):
        """Ensures waypoints stay within the defined map bounds."""
        z_clipped = np.copy(z)
        z_clipped[0::2] = np.clip(z_clipped[0::2], self.env.x_min, self.env.x_max)
        z_clipped[1::2] = np.clip(z_clipped[1::2], self.env.y_min, self.env.y_max)
        return z_clipped

    def _limit_step(self, z, z_next):
        """Clips the step size to prevent divergence (Max Step)."""
        step = z_next - z
        step_norm = np.linalg.norm(step)
        if not np.isfinite(step_norm):
            return z, "nonfinite_step"

        if step_norm > self.max_step:
            step = step * (self.max_step / step_norm)

        return z + step, None

    def _record_stats(self, z, g_norm, w_time, w_energy, w_smooth):
        """Records objective value and constraint violations."""
        obj_val = self.deriv.penalized_objective(z, w_time, w_energy, self.rho, w_smooth)
        constraints = self.cons.all_constraints(z, self.obj.energy_cost, self.obj.time_cost)
        max_violation = float(max(0.0, np.max(constraints)))

        self.history.append(obj_val)
        self.grad_norm_history.append(g_norm)
        self.constraint_violation_history.append(max_violation)
        self.feasibility_history.append(bool(max_violation <= 1e-6))

    def _check_convergence(self, g_norm):
        """Checks if the stopping criteria are met."""
        if g_norm < self.tol:
            return "gradient_norm_tolerance"

        if len(self.history) > 1:
            if abs(self.history[-1] - self.history[-2]) < self.tol:
                return "objective_change_tolerance"

        return None

    @abstractmethod
    def _compute_update(self, z, grad, **kwargs):
        """Subclasses implement their specific update rule here."""
        pass

    def fit(self, z0, w_time=1.0, w_energy=1.0, w_smooth=0.0, **kwargs):
        """Main optimization loop shared by all descendants."""
        z = np.array(z0, dtype=float)
        self._reset_state()

        start_time = time.time()

        for k in range(self.max_iter):
            # 1. Update penalty rho via schedule
            if self.rho_schedule:
                violation = self.constraint_violation_history[-1] if k > 0 else 0.0
                self.rho = self.rho_schedule(k, self.rho, violation)

            # 2. Compute Gradient
            g = self.deriv.gradient(z, w_time, w_energy, self.rho, w_smooth)
            g_norm = np.linalg.norm(g)

            # 3. Record current state stats
            self._record_stats(z, g_norm, w_time, w_energy, w_smooth)

            # 4. Check stopping criteria
            reason = self._check_convergence(g_norm)
            if reason:
                self.stopping_reason = reason
                break

            # 5. Compute update from subclass
            z_next = self._compute_update(z, g, **kwargs)

            # 6. Clip step and apply workspace bounds
            z, stability_reason = self._limit_step(z, z_next)
            if stability_reason:
                self.stopping_reason = stability_reason
                break

            z = self._clip_to_workspace(z)

        self.runtime = time.time() - start_time
        return z

    def predict(self):
        """Returns the final penalized objective value."""
        return self.history[-1] if self.history else None
