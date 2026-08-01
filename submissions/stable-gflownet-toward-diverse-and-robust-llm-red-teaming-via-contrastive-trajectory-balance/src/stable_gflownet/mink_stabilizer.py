import torch
import torch.nn as nn
from typing import Dict, Tuple

def mink_fluency_loss(
    token_log_probs: torch.Tensor,
    k_percent: float = 0.2,
    fluency_threshold: float = -3.5,
    penalty_scale: float = 2.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute Min-K Fluency Stabilizer penalty for prompt token sequences.
    
    Args:
        token_log_probs: [B, L] Log probabilities of prompt tokens under a reference language model
        k_percent: Fraction k of lowest token log-probs to average (e.g. 0.2 for Min-K 20%)
        fluency_threshold: Threshold below which prompts are considered gibberish / low-fluency
        penalty_scale: Multiplier for penalty strength
        
    Returns:
        penalty: [B] Penalty values for each sequence
        metrics: Summary dictionary
    """
    batch_size, seq_len = token_log_probs.shape
    k = max(1, int(seq_len * k_percent))
    
    # Sort log probabilities along sequence dimension to find lowest k tokens
    sorted_log_probs, _ = torch.sort(token_log_probs, dim=-1, descending=False)
    min_k_probs = sorted_log_probs[:, :k]
    min_k_scores = torch.mean(min_k_probs, dim=-1)  # [B]
    
    # Apply hinge penalty for scores below fluency_threshold
    penalties = penalty_scale * torch.relu(fluency_threshold - min_k_scores)  # [B]
    
    metrics = {
        "mean_min_k_score": float(torch.mean(min_k_scores).item()),
        "mean_fluency_penalty": float(torch.mean(penalties).item()),
        "flagged_gibberish_count": int((min_k_scores < fluency_threshold).sum().item()),
        "total_prompts": batch_size
    }
    
    return penalties, metrics


def compute_mink_penalty(
    fluent_log_probs: torch.Tensor,
    gibberish_log_probs: torch.Tensor,
    k_percent: float = 0.2,
    fluency_threshold: float = -3.5
) -> Dict[str, float]:
    """Test helper comparing Min-K scores and penalties on fluent vs gibberish prompts."""
    pen_fluent, m_fluent = mink_fluency_loss(fluent_log_probs, k_percent=k_percent, fluency_threshold=fluency_threshold)
    pen_gibberish, m_gibberish = mink_fluency_loss(gibberish_log_probs, k_percent=k_percent, fluency_threshold=fluency_threshold)
    
    return {
        "fluent_min_k": m_fluent["mean_min_k_score"],
        "gibberish_min_k": m_gibberish["mean_min_k_score"],
        "fluent_penalty": m_fluent["mean_fluency_penalty"],
        "gibberish_penalty": m_gibberish["mean_fluency_penalty"],
        "fluency_separation_valid": bool(m_gibberish["mean_fluency_penalty"] > m_fluent["mean_fluency_penalty"])
    }
