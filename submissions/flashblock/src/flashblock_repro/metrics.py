"""Metrics and empirical analysis for FlashBlock reproduction."""

from typing import Dict
import torch

def compute_cross_step_stability(
    A_out_s: torch.Tensor,
    A_out_s1: torch.Tensor,
    A_in_s: torch.Tensor,
    A_in_s1: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes cross-step cosine similarity and L1 distance for block-external
    and block-internal attention outputs.
    """
    # Flatten spatial/token dimensions for pairwise similarity
    flat_A_out_s = A_out_s.reshape(-1)
    flat_A_out_s1 = A_out_s1.reshape(-1)
    
    flat_A_in_s = A_in_s.reshape(-1)
    flat_A_in_s1 = A_in_s1.reshape(-1)
    
    ext_cos = torch.cosine_similarity(flat_A_out_s.unsqueeze(0), flat_A_out_s1.unsqueeze(0)).item()
    ext_l1 = torch.mean(torch.abs(flat_A_out_s - flat_A_out_s1)).item()
    
    int_cos = torch.cosine_similarity(flat_A_in_s.unsqueeze(0), flat_A_in_s1.unsqueeze(0)).item()
    int_l1 = torch.mean(torch.abs(flat_A_in_s - flat_A_in_s1)).item()
    
    return {
        "external_cosine_similarity": ext_cos,
        "external_l1_distance": ext_l1,
        "internal_cosine_similarity": int_cos,
        "internal_l1_distance": int_l1,
    }

def compute_composition_error(
    A_full: torch.Tensor,
    A_composed: torch.Tensor
) -> Dict[str, float]:
    """Computes L1 and L_infinity errors between full single-pass and log-space composed attention."""
    diff = torch.abs(A_full - A_composed)
    l1_err = torch.mean(diff).item()
    linf_err = torch.max(diff).item()
    return {
        "l1_error": l1_err,
        "linf_error": linf_err,
    }

def compute_speedup_and_flops(
    batch_size: int,
    num_heads: int,
    d_k: int,
    context_len: int,
    block_size: int,
    num_steps: int,
    update_threshold: int = 2,
) -> Dict[str, float]:
    """
    Computes theoretical FLOPs and estimated speedup for standard block diffusion vs FlashBlock caching.
    
    Standard Block Diffusion FLOPs per step:
        2 * B * (N_context + B) * d_k * num_heads
    FlashBlock FLOPs when recomputing (step 1 or num_updated >= tau):
        2 * B * (N_context + B) * d_k * num_heads
    FlashBlock FLOPs when reusing cache (steps > 1 when num_updated < tau):
        2 * B * B * d_k * num_heads
    """
    dense_flops_per_step = 2.0 * batch_size * num_heads * block_size * (context_len + block_size) * d_k
    total_dense_flops = dense_flops_per_step * num_steps
    
    # Calculate FlashBlock step FLOPs
    total_fb_flops = 0.0
    for s in range(num_steps):
        # Step 0 always computes full
        # For simplicity in simulation, assuming step 0 recomputes, remaining steps reuse if num_updated < threshold
        num_updated = block_size if s == 0 else 1
        if num_updated >= update_threshold:
            total_fb_flops += dense_flops_per_step
        else:
            cached_step_flops = 2.0 * batch_size * num_heads * block_size * block_size * d_k
            total_fb_flops += cached_step_flops
            
    speedup = total_dense_flops / (total_fb_flops + 1e-8)
    
    return {
        "dense_flops": total_dense_flops,
        "flashblock_flops": total_fb_flops,
        "theoretical_speedup": speedup,
    }
