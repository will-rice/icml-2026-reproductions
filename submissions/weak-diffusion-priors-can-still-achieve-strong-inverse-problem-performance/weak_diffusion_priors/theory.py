import numpy as np
from typing import Dict, Any


def simulate_theorem_3_1_posterior_concentration(
    n_dim: int = 128,
    measurement_ratios: list[float] = None,
    noise_std: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate Bayesian posterior concentration under Theorem 3.1.
    
    Verifies that high-dimensional measurements make the Bayesian posterior
    concentrate near the true signal x* even when using a weak (mismatched/shifted) prior.
    """
    if measurement_ratios is None:
        measurement_ratios = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    rng = np.random.default_rng(seed)
    
    # Ground truth signal x* ~ N(0, I_n)
    true_signal = rng.standard_normal(n_dim)
    
    # Weak prior: biased mean and inflated variance
    weak_prior_mean = rng.standard_normal(n_dim) * 0.5  # Prior drift
    weak_prior_var = 2.0  # Inflated/weak precision
    inv_weak_cov = np.eye(n_dim) / weak_prior_var

    # True prior inverse covariance
    inv_true_cov = np.eye(n_dim)

    results = []

    for ratio in measurement_ratios:
        m_dim = int(np.round(n_dim * ratio))
        # Measurement matrix A in R^{m x n}
        A = rng.standard_normal((m_dim, n_dim)) / np.sqrt(m_dim)
        
        # Noise
        noise = rng.normal(0, noise_std, size=m_dim)
        y = A @ true_signal + noise

        # Posterior under Weak Prior:
        # Sigma_post = (A^T A / sigma^2 + Sigma_weak^{-1})^{-1}
        # mu_post = Sigma_post (A^T y / sigma^2 + Sigma_weak^{-1} mu_weak)
        precision_post_weak = (A.T @ A) / (noise_std ** 2) + inv_weak_cov
        cov_post_weak = np.linalg.inv(precision_post_weak)
        mu_post_weak = cov_post_weak @ ((A.T @ y) / (noise_std ** 2) + inv_weak_cov @ weak_prior_mean)

        # Posterior under True (Strong) Prior:
        precision_post_true = (A.T @ A) / (noise_std ** 2) + inv_true_cov
        cov_post_true = np.linalg.inv(precision_post_true)
        mu_post_true = cov_post_true @ ((A.T @ y) / (noise_std ** 2))

        # Metrics
        reconstruction_error_weak = float(np.linalg.norm(mu_post_weak - true_signal))
        reconstruction_error_true = float(np.linalg.norm(mu_post_true - true_signal))
        posterior_trace_weak = float(np.trace(cov_post_weak))
        cosine_sim_weak = float(np.dot(mu_post_weak, true_signal) / (np.linalg.norm(mu_post_weak) * np.linalg.norm(true_signal) + 1e-9))

        results.append({
            "measurement_ratio": float(ratio),
            "m_dim": m_dim,
            "reconstruction_error_weak": reconstruction_error_weak,
            "reconstruction_error_true": reconstruction_error_true,
            "posterior_trace_weak": posterior_trace_weak,
            "cosine_sim_weak": cosine_sim_weak,
            "error_ratio_weak_vs_true": reconstruction_error_weak / max(reconstruction_error_true, 1e-9),
        })

    # Theorem 3.1 assertion: High measurement ratio (>= 0.75) shows error reduction and concentration
    high_ratio_res = [r for r in results if r["measurement_ratio"] >= 0.75]
    low_ratio_res = [r for r in results if r["measurement_ratio"] <= 0.25]

    theorem_3_1_verified = bool(
        np.mean([r["reconstruction_error_weak"] for r in high_ratio_res]) <
        0.5 * np.mean([r["reconstruction_error_weak"] for r in low_ratio_res])
    )

    return {
        "theorem_3_1_verified": theorem_3_1_verified,
        "n_dim": n_dim,
        "noise_std": noise_std,
        "sweep_results": results,
    }
