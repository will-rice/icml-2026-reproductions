"""Activation Subtraction Module for Prompt Steering Interventions (Figure 1)."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Any

def compute_intervention_vectors(
    prompt_activations: torch.Tensor,
    base_activations: torch.Tensor,
) -> torch.Tensor:
    """
    Compute prompt steering intervention vectors: v_t = h_t^prompt - h_t^base.
    
    Args:
        prompt_activations: Tensor of shape (batch_size, seq_len, hidden_dim)
        base_activations: Tensor of shape (batch_size, seq_len, hidden_dim)
        
    Returns:
        Intervention vectors v_t of shape (batch_size, seq_len, hidden_dim)
    """
    if prompt_activations.shape != base_activations.shape:
        raise ValueError(
            f"Shape mismatch: {prompt_activations.shape} vs {base_activations.shape}"
        )
    return prompt_activations - base_activations

def analyze_token_dependent_strengths(
    interventions: torch.Tensor,
) -> Dict[str, Any]:
    """
    Analyze token-dependent intervention strengths (norms) across sequence positions (Figure 2).
    
    Returns metrics demonstrating token-dependent variation.
    """
    norms = torch.norm(interventions, dim=-1)  # (batch_size, seq_len)
    mean_norm = torch.mean(norms).item()
    std_norm = torch.std(norms).item()
    max_norm = torch.max(norms).item()
    min_norm = torch.min(norms).item()
    token_variance = torch.var(norms, dim=-1).mean().item()
    
    return {
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "max_norm": max_norm,
        "min_norm": min_norm,
        "token_variance": token_variance,
        "is_token_dependent": std_norm > 0.01,
    }
