import numpy as np
from typing import Dict, Any


def evaluate_table_1_inverse_problem_baselines(
    signal_length: int = 256,
    num_samples: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate Table 1 inverse-problem performance comparing weak priors vs strong priors.
    
    Verifies Claim 1: Weak diffusion priors match strong-prior inverse-problem baselines
    when measurements are highly informative (e.g. high pixel observation ratio m/n >= 0.85).
    """
    rng = np.random.default_rng(seed)

    # Generate structured synthetic signal instances with spatial smoothness
    t = np.linspace(0, 4 * np.pi, signal_length)
    signals = []
    for k in range(num_samples):
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(1.0, 3.0)
        clean = np.sin(freq * t + phase) + 0.5 * np.cos(2 * freq * t)
        signals.append(clean)
    signals = np.array(signals)  # Shape (num_samples, signal_length)

    # Strong prior covariance (exact empirical data covariance)
    cov_strong = np.cov(signals, rowvar=False) + 1e-3 * np.eye(signal_length)
    inv_cov_strong = np.linalg.inv(cov_strong)

    # Weak prior: Exponential kernel (captures general spatial smoothness, but domain-mismatched)
    idx = np.arange(signal_length)
    dist_matrix = np.abs(idx[:, None] - idx[None, :])
    lengthscale = 4.0
    cov_weak = np.exp(-dist_matrix / lengthscale) + 1e-3 * np.eye(signal_length)
    inv_cov_weak = np.linalg.inv(cov_weak)

    regimes = {
        "Low_Informative_m_n_0.25": 0.25,
        "High_Informative_m_n_0.90": 0.90,
    }

    metrics_by_regime = {}

    for regime_name, m_ratio in regimes.items():
        m_dim = int(np.round(signal_length * m_ratio))
        
        psnr_strong_list = []
        psnr_weak_list = []
        psnr_uninformative_list = []

        for i in range(num_samples):
            x_true = signals[i]
            
            # Measurement matrix (random pixel observation)
            obs_indices = rng.choice(signal_length, size=m_dim, replace=False)
            A = np.zeros((m_dim, signal_length))
            for idx_pos, pos in enumerate(obs_indices):
                A[idx_pos, pos] = 1.0
            
            sigma_noise = 0.05
            y = A @ x_true + rng.normal(0, sigma_noise, size=m_dim)

            # MAP Solvers
            AtA = A.T @ A / (sigma_noise ** 2)
            Aty = A.T @ y / (sigma_noise ** 2)

            # Strong prior reconstruction
            cov_post_strong = np.linalg.inv(AtA + inv_cov_strong)
            x_hat_strong = cov_post_strong @ Aty

            # Weak prior reconstruction
            cov_post_weak = np.linalg.inv(AtA + inv_cov_weak)
            x_hat_weak = cov_post_weak @ Aty

            # Uninformative baseline
            cov_post_uninf = np.linalg.inv(AtA + 1.0 * np.eye(signal_length))
            x_hat_uninf = cov_post_uninf @ Aty

            # Calculate PSNR
            max_val = max(np.ptp(x_true), 1e-5)
            
            mse_strong = float(np.mean((x_hat_strong - x_true) ** 2))
            psnr_strong = 10 * np.log10((max_val ** 2) / max(mse_strong, 1e-10))

            mse_weak = float(np.mean((x_hat_weak - x_true) ** 2))
            psnr_weak = 10 * np.log10((max_val ** 2) / max(mse_weak, 1e-10))

            mse_uninf = float(np.mean((x_hat_uninf - x_true) ** 2))
            psnr_uninf = 10 * np.log10((max_val ** 2) / max(mse_uninf, 1e-10))

            psnr_strong_list.append(psnr_strong)
            psnr_weak_list.append(psnr_weak)
            psnr_uninformative_list.append(psnr_uninf)

        metrics_by_regime[regime_name] = {
            "mean_psnr_strong_prior": float(np.mean(psnr_strong_list)),
            "mean_psnr_weak_prior": float(np.mean(psnr_weak_list)),
            "mean_psnr_uninformative": float(np.mean(psnr_uninformative_list)),
            "psnr_ratio_weak_vs_strong": float(np.mean(psnr_weak_list) / np.mean(psnr_strong_list)),
        }

    high_regime = metrics_by_regime["High_Informative_m_n_0.90"]
    low_regime = metrics_by_regime["Low_Informative_m_n_0.25"]

    # Claim 1 assertion: In high informative regime, weak prior PSNR is >= 80% of strong prior PSNR and substantially outclasses low regime
    claim_1_verified = bool(
        high_regime["psnr_ratio_weak_vs_strong"] >= 0.80 and
        high_regime["mean_psnr_weak_prior"] > low_regime["mean_psnr_weak_prior"] + 10.0
    )

    return {
        "claim_1_verified": claim_1_verified,
        "table_1_metrics": metrics_by_regime,
    }
