import numpy as np
from src.base_optimizer import BaseOptimizer

class HeavyBallOptimizer(BaseOptimizer):
    """
    Implements Heavy-Ball Momentum by inheriting from BaseOptimizer.
    Update: z_{k+1} = z_k - eta * grad(F_penalty) + beta * (z_k - z_{k-1})
    """
    def __init__(self, env, objective, constraints, derivatives,
                 learning_rate=0.01, momentum=0.9, max_iter=1000, tol=1e-5, rho=100.0,
                 max_step=0.25, rho_schedule=None):
        super().__init__(env, objective, constraints, derivatives,
                         learning_rate, max_iter, tol, rho, max_step, rho_schedule)
        self.beta = momentum
        self.z_prev = None

    def _compute_update(self, z, grad, **kwargs):
        """Implements the Heavy-Ball momentum update rule."""
        if self.z_prev is None:
            self.z_prev = np.copy(z)

        # HB Update: z_next = z - eta * grad + beta * (z - z_prev)
        z_next = z - self.eta * grad + self.beta * (z - self.z_prev)

        # Update state for next iteration
        self.z_prev = np.copy(z)

        return z_next
