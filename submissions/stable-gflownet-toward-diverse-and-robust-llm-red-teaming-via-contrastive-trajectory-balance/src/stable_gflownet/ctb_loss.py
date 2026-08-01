import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

def compute_ctb_loss(
    log_pf: torch.Tensor,
    log_pb: torch.Tensor,
    log_rewards: torch.Tensor,
    pairs: torch.Tensor = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute Contrastive Trajectory Balance (CTB) loss for a batch of trajectories.

    Args:
        log_pf: [B] Forward trajectory log-probabilities log P_F(tau)
        log_pb: [B] Backward trajectory log-probabilities log P_B(tau)
        log_rewards: [B] Terminal reward log-values log R(x)
        pairs: Optional [N_pairs, 2] indices of pairwise trajectory comparisons.
               If None, all distinct pairs (i, j) with i < j are constructed.

    Returns:
        loss: Scalar CTB loss tensor
        metrics: Dictionary containing CTB evaluation metrics (e.g. mean error, Z estimation parameter count=0)
    """
    batch_size = log_pf.size(0)
    if pairs is None:
        # Construct all unique pairs (i, j) with i < j
        i_idx, j_idx = torch.triu_indices(batch_size, batch_size, offset=1)
    else:
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]

    if i_idx.numel() == 0:
        return torch.tensor(0.0, device=log_pf.device), {"num_pairs": 0, "explicit_z_used": False}

    # Trajectory log-ratios delta_tau = log P_F(tau) - log P_B(tau)
    traj_ratio = log_pf - log_pb

    # Pairwise differences
    delta_traj = traj_ratio[i_idx] - traj_ratio[j_idx]  # (log P_F(i) - log P_B(i)) - (log P_F(j) - log P_B(j))
    delta_reward = log_rewards[i_idx] - log_rewards[j_idx]  # log R(x_i) - log R(x_j)

    # CTB residual: (delta_traj - delta_reward)^2
    residuals = delta_traj - delta_reward
    loss = torch.mean(residuals ** 2)

    metrics = {
        "num_pairs": float(i_idx.numel()),
        "ctb_loss": float(loss.item()),
        "mean_residual": float(torch.mean(torch.abs(residuals)).item()),
        "explicit_z_used": False,
        "partition_function_params": 0
    }

    return loss, metrics


def contrastive_trajectory_balance(
    log_pf_batch: torch.Tensor,
    log_pb_batch: torch.Tensor,
    log_rewards_batch: torch.Tensor
) -> Dict[str, float]:
    """Helper wrapper for CTB validation."""
    loss, metrics = compute_ctb_loss(log_pf_batch, log_pb_batch, log_rewards_batch)
    return metrics
