import torch
import torch.nn as nn
from typing import Dict, Tuple

def noisy_gradient_pruning(
    log_rewards: torch.Tensor,
    threshold: float = 0.1,
    soft_gate: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Noisy Gradient Pruning (NGP) mask/weights for pairwise trajectory comparisons.
    
    Args:
        log_rewards: [B] Log reward tensor
        threshold: Threshold for minimum informative reward difference tau_NGP
        soft_gate: If True, uses sigmoid gating instead of hard thresholding
        
    Returns:
        pairs: [N_valid, 2] tensor of valid pair indices
        weights: [N_valid] weights for each pair
    """
    batch_size = log_rewards.size(0)
    i_idx, j_idx = torch.triu_indices(batch_size, batch_size, offset=1)
    
    reward_diff = torch.abs(log_rewards[i_idx] - log_rewards[j_idx])
    
    if soft_gate:
        weights = torch.sigmoid((reward_diff - threshold) * 10.0)
        valid_mask = weights > 0.05
    else:
        valid_mask = reward_diff >= threshold
        weights = torch.ones_like(reward_diff)
        
    valid_pairs = torch.stack([i_idx[valid_mask], j_idx[valid_mask]], dim=1)
    valid_weights = weights[valid_mask]
    
    return valid_pairs, valid_weights


def filter_noisy_rewards(
    log_pf: torch.Tensor,
    log_pb: torch.Tensor,
    log_rewards: torch.Tensor,
    threshold: float = 0.1
) -> Dict[str, float]:
    """Evaluate NGP pruning statistics on a batch of rewards."""
    batch_size = log_rewards.size(0)
    total_pairs = batch_size * (batch_size - 1) // 2
    
    valid_pairs, valid_weights = noisy_gradient_pruning(log_rewards, threshold=threshold)
    num_kept = valid_pairs.size(0)
    pruned_ratio = 1.0 - (num_kept / max(total_pairs, 1))
    
    return {
        "total_pairs": float(total_pairs),
        "kept_pairs": float(num_kept),
        "pruned_ratio": float(pruned_ratio),
        "ngp_threshold": float(threshold),
        "filtering_active": True
    }
