import numpy as np
from src.base_optimizer import BaseOptimizer

class GradientDescentOptimizer(BaseOptimizer):
    """
    Implements Gradient Descent by inheriting from BaseOptimizer.
    Update: z_{k+1} = z_k - eta * grad(F_penalty)
    """
    def _compute_update(self, z, grad, **kwargs):
        """Implements the vanilla Gradient Descent update rule."""
        return z - self.eta * grad
