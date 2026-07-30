from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def compute_token_distance_matrix(embeddings: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """Compute pairwise distance matrix D for candidate token embeddings."""
    if metric == "cosine":
        norm_emb = F.normalize(embeddings, p=2, dim=-1)
        sim = torch.mm(norm_emb, norm_emb.t())
        dist = torch.clamp(1.0 - sim, min=0.0)
    elif metric == "euclidean":
        dist = torch.cdist(embeddings, embeddings, p=2) ** 2
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return dist


def min_p_filter(probs: torch.Tensor, min_p: float = 0.05) -> torch.Tensor:
    """Min-p truncation baseline."""
    max_p = probs.max()
    threshold = min_p * max_p
    mask = probs >= threshold
    filtered_probs = probs * mask.float()
    return filtered_probs / filtered_probs.sum()


def top_p_filter(probs: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """Top-p (nucleus) truncation baseline."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    filtered_probs = probs.masked_fill(indices_to_remove, 0.0)
    return filtered_probs / filtered_probs.sum()


def top_h_filter(logits: torch.Tensor, top_h_ratio: float = 0.8) -> torch.Tensor:
    """Top-H (Entropy-based) truncation baseline."""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12))
    target_mass = min(1.0, max(0.1, float(top_h_ratio * torch.exp(-entropy).item())))
    return top_p_filter(probs, top_p=target_mass)


def top_w_filter(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    temperature: float = 1.0,
    top_p_prefilter: float = 0.95,
    lambda_ent: float = 0.1,
    lambda_mass: float = 1.0,
    max_iter: int = 20,
) -> torch.Tensor:
    """Top-W geometry-aware decoding algorithm (Section 3, Algorithm 1 & 4.2)."""
    scaled_logits = logits / max(temperature, 1e-5)
    full_probs = F.softmax(scaled_logits, dim=-1)

    # Pre-filter candidate pool using mild top-p
    sorted_probs, sorted_idx = torch.sort(full_probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cum_probs <= top_p_prefilter
    mask[0] = True  # Keep at least top 1
    cand_idx = sorted_idx[mask]

    cand_probs = full_probs[cand_idx]
    cand_probs = cand_probs / cand_probs.sum()
    cand_emb = embeddings[cand_idx]

    dist_matrix = compute_token_distance_matrix(cand_emb, metric="cosine")

    # Alternating solver for optimal candidate subset S
    q = cand_probs.clone()
    for _ in range(max_iter):
        cost = torch.mv(dist_matrix, q)
        penalty = lambda_mass * (1.0 - q.sum())
        entropy_grad = lambda_ent * (torch.log(q + 1e-12) + 1.0)
        
        logits_update = -cost - entropy_grad + penalty
        q_next = F.softmax(logits_update, dim=-1)
        if torch.norm(q_next - q) < 1e-4:
            break
        q = q_next

    # Map back to full vocabulary
    out_probs = torch.zeros_like(full_probs)
    out_probs[cand_idx] = q
    return out_probs / out_probs.sum()


def evaluate_decoding_metrics(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Evaluate Top-W against Min-p, Top-p, and Top-H baselines."""
    probs_orig = F.softmax(logits / temperature, dim=-1)

    p_top_w = top_w_filter(logits, embeddings, temperature=temperature)
    p_min_p = min_p_filter(probs_orig, min_p=0.05)
    p_top_p = top_p_filter(probs_orig, top_p=0.9)
    p_top_h = top_h_filter(logits, top_h_ratio=0.8)

    def calc_entropy(p: torch.Tensor) -> float:
        p_sub = p[p > 0]
        return float(-torch.sum(p_sub * torch.log(p_sub)).item())

    def calc_subset_size(p: torch.Tensor) -> float:
        return float((p > 1e-5).sum().item())

    return {
        "entropy_orig": calc_entropy(probs_orig),
        "entropy_top_w": calc_entropy(p_top_w),
        "entropy_min_p": calc_entropy(p_min_p),
        "entropy_top_p": calc_entropy(p_top_p),
        "entropy_top_h": calc_entropy(p_top_h),
        "subset_size_top_w": calc_subset_size(p_top_w),
        "subset_size_min_p": calc_subset_size(p_min_p),
        "subset_size_top_p": calc_subset_size(p_top_p),
        "subset_size_top_h": calc_subset_size(p_top_h),
    }
