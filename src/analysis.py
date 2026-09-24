import numpy as np


def hessian_analysis(derivatives, z, w_time=1.0, w_energy=1.0, rho=100.0):
    hessian = derivatives.numerical_hessian(
        z, w_time=w_time, w_energy=w_energy, rho=rho
    )
    eigenvalues = np.linalg.eigvalsh(hessian)
    tolerance = 1e-6
    if np.all(eigenvalues > tolerance):
        classification = "positive definite"
    elif np.all(eigenvalues >= -tolerance):
        classification = "positive semidefinite"
    elif np.any(eigenvalues > tolerance) and np.any(eigenvalues < -tolerance):
        classification = "indefinite"
    else:
        classification = "negative semidefinite"

    # Note: This analysis is LOCAL to the point z.
    # The original problem is globally non-convex due to NFZ constraints.
    condition_number = derivatives.condition_number(
        z, w_time=w_time, w_energy=w_energy, rho=rho
    )
    return {
        "eigenvalues": eigenvalues.tolist(),
        "minimum_eigenvalue": float(np.min(eigenvalues)),
        "maximum_eigenvalue": float(np.max(eigenvalues)),
        "classification": classification,
        "condition_number": float(condition_number) if np.isfinite(condition_number) else None,
        "symmetry_error": float(np.max(np.abs(hessian - hessian.T))),
        "local_convexity": classification == "positive definite" or classification == "positive semidefinite",
        "global_convexity": False, # Non-convex due to NFZ obstacles
    }
