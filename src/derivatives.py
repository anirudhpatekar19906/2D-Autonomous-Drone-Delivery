import numpy as np

class Derivatives:
    """
    Calculates gradients and Hessians for the drone optimization problem.
    """
    def __init__(self, environment, objective, constraints):
        self.env = environment
        self.obj = objective
        self.cons = constraints
        self.eps = 1e-6  # For numerical stability

    def gradient(self, z, w_time=1.0, w_energy=1.0, rho=100.0, w_smooth=0.0):
        """
        Calculates the gradient of the penalized objective:
        F_penalty = w_t*T + w_e*E + rho * sum(max(0, g_j)^2)
        """
        grad = np.zeros_like(z)

        # 1. Gradient of Distance-based objective (T and E)
        # Time and energy are normalized before applying delivery-mode weights.
        power = self.obj.c0 + self.obj.c1 * self.obj.v_max + self.obj.c2 * self.obj.v_max**2
        coeff = (w_time / self.obj.time_reference_s / self.obj.v_max) + (
            w_energy / self.obj.energy_reference_Wh * power / (3600.0 * self.obj.v_max)
        )

        # Distance Gradient
        waypoints = z.reshape(-1, 2)
        full_path = np.vstack([self.env.get_start(), waypoints, self.env.get_goal()])

        # Gradient of dist d_i = sqrt((x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2)
        # d d_i / d P_i = (P_i - P_{i+1}) / d_i
        # d d_i / d P_{i+1} = (P_{i+1} - P_i) / d_i

        dist_grad = np.zeros_like(waypoints)
        for i in range(len(full_path) - 1):
            p_curr = full_path[i]
            p_next = full_path[i+1]
            d = np.linalg.norm(p_next - p_curr) + self.eps

            g_curr = (p_curr - p_next) / d
            g_next = (p_next - p_curr) / d

            # P_0 is fixed, start from i=1 for waypoints
            if i > 0:
                dist_grad[i-1] += g_curr
            # P_{N+1} is fixed, end at i=N
            if i < len(full_path) - 2:
                dist_grad[i] += g_next

        grad += coeff * dist_grad.flatten()
        if w_smooth > 0:
            grad += w_smooth * self.obj.smoothness_gradient(z)

        # 2. Gradient of Penalty Terms: 2 * rho * max(0, g) * grad(g)
        # Battery constraint g_B = Energy - B_max
        # grad(g_B) = grad(Energy) = coeff_energy * grad(Distance)
        energy_coeff = (self.obj.c0 + self.obj.c1*self.obj.v_max + self.obj.c2*self.obj.v_max**2) / (3600.0 * self.obj.v_max)
        g_B = self.cons.battery_constraint(z, self.obj.energy_cost)
        if g_B > 0:
            grad += 2 * rho * g_B * (energy_coeff * dist_grad.flatten())

        g_T = self.cons.time_constraint(z, self.obj.time_cost)
        if g_T > 0:
            grad += 2 * rho * g_T * (dist_grad.flatten() / self.obj.v_max)

        # Buffered waypoint NFZ constraints g_NFZ = R_safe^2 - ||P_i - C||^2
        # grad(g_NFZ) = -2(P_i - C)
        nfz_vals = self.cons.nfz_constraint(z)
        obstacles = self.env.get_obstacles()

        # Map NFZ constraints back to waypoints
        for i, p in enumerate(waypoints):
            for j, obs in enumerate(obstacles):
                g_val = nfz_vals[i * len(obstacles) + j]
                if g_val > 0:
                    center = np.array(obs['center'])
                    grad_g = -2 * (p - center)
                    grad[2*i : 2*i+2] += 2 * rho * g_val * grad_g

        # Penalize interior segment samples as well as waypoints. This prevents
        # a pair of safe waypoints from hiding a collision between them.
        full_path = np.vstack([self.env.get_start(), waypoints, self.env.get_goal()])
        sample_index = 0
        for segment_index, (start, end) in enumerate(zip(full_path[:-1], full_path[1:])):
            for fraction in np.linspace(0.0, 1.0, self.cons.segment_samples + 2)[1:-1]:
                point = start + fraction * (end - start)
                for obstacle in obstacles:
                    center = np.asarray(obstacle['center'])
                    radius = self.env.effective_obstacle_radius(obstacle)
                    g_val = radius**2 - np.sum((point - center) ** 2)
                    if g_val > 0:
                        grad_point = -2 * (point - center)
                        if segment_index > 0:
                            grad[2 * (segment_index - 1):2 * segment_index] += (
                                2 * rho * g_val * (1 - fraction) * grad_point
                            )
                        if segment_index < len(waypoints):
                            grad[2 * segment_index:2 * (segment_index + 1)] += (
                                2 * rho * g_val * fraction * grad_point
                            )
                    sample_index += 1

        # Workspace constraints
        ws_vals = self.cons.workspace_constraint(z)
        # Gradient of (x_min - x) is [-1, 0]
        # Gradient of (x - x_max) is [1, 0]
        for i in range(len(waypoints)):
            # x_min - x
            if ws_vals[4*i] > 0:
                grad[2*i] += 2 * rho * ws_vals[4*i] * (-1)
            # x - x_max
            if ws_vals[4*i+1] > 0:
                grad[2*i] += 2 * rho * ws_vals[4*i+1] * (1)
            # y_min - y
            if ws_vals[4*i+2] > 0:
                grad[2*i+1] += 2 * rho * ws_vals[4*i+2] * (-1)
            # y - y_max
            if ws_vals[4*i+3] > 0:
                grad[2*i+1] += 2 * rho * ws_vals[4*i+3] * (1)

        return grad

    def penalized_objective(self, z, w_time=1.0, w_energy=1.0, rho=100.0, w_smooth=0.0):
        """Objective plus squared violation penalties used by both optimizers."""
        violations = np.maximum(
            self.cons.all_constraints(z, self.obj.energy_cost, self.obj.time_cost), 0.0
        )
        return self.obj.combined_objective(z, w_time, w_energy, w_smooth) + rho * np.sum(violations**2)

    def numerical_hessian(self, z, w_time=1.0, w_energy=1.0, rho=100.0):
        """
        Calculates the Hessian using finite differences:
        H_ij = [grad_i(z + h*e_j) - grad_i(z - h*e_j)] / 2h
        """
        h = 1e-4
        n = len(z)
        hessian = np.zeros((n, n))

        # Base gradient for efficiency (though we use central difference)
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0

            g_plus = self.gradient(z + h*ej, w_time, w_energy, rho)
            g_minus = self.gradient(z - h*ej, w_time, w_energy, rho)

            hessian[:, j] = (g_plus - g_minus) / (2 * h)

        # Symmetrize
        return (hessian + hessian.T) / 2

    def condition_number(self, z, w_time=1.0, w_energy=1.0, rho=100.0):
        """
        Kappa = lambda_max / lambda_min
        """
        H = self.numerical_hessian(z, w_time, w_energy, rho)
        eigvals = np.linalg.eigvalsh(H)

        l_max = np.max(eigvals)
        if l_max <= self.eps:
            return np.inf
        l_min = np.min(eigvals)
        if l_min <= self.eps:
            return np.inf

        return l_max / l_min
