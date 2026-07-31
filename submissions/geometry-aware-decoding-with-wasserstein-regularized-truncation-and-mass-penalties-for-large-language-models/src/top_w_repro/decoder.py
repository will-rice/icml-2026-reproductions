"""Faithful reimplementation of Top-W decoding (arXiv:2602.10346v2).

Mirrors the official implementation at
github.com/arashgholami/top-w-decoding@5949bfae (logit_processor_w1.py),
which is vendored byte-exactly under evidence/inputs/upstream/ and used
as a cross-check oracle in the tests and the evidence audits.

Notation follows the paper: at each step the model distribution p over
the candidate pool is cropped to a subset S maximizing the fixed-
potential surrogate

    L_f(S) = E_{q_S}[varphi] + (beta - lam) * log Gamma_S,
    varphi_i = geom_scale * f_i + lam * log p_i,

where f_i = -min_{j in S} d_cos(i, j) is the nearest-set potential and
Gamma_S the retained mass. Theorem 3.4: for fixed f the optimal S is a
prefix of the varphi-sorted order, found by a linear scan.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-12


def whiten_and_normalize(embeddings: torch.Tensor) -> torch.Tensor:
    """Diagonal-whiten over the vocabulary then L2-normalize each row.

    Matches the official preprocessing (Appendix A: normalization and
    diagonal whitening) applied before cosine distances.
    """
    mean = embeddings.mean(dim=0, keepdim=True)
    centered = embeddings - mean
    scale = 1.0 / torch.sqrt(centered.pow(2).mean(dim=0, keepdim=True).clamp(min=1e-6))
    return F.normalize(centered * scale, p=2, dim=-1)


def pairwise_cosine_distance(embeddings: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine distance 1 - cos for unit-norm rows."""
    sim = embeddings @ embeddings.t()
    return (1.0 - sim.clamp(-1.0, 1.0)).clamp(min=0.0).fill_diagonal_(0.0)


def nearest_set_potential(
    embeddings: torch.Tensor, kept: torch.Tensor
) -> torch.Tensor:
    """f_i = -min_{j in S} (1 - cos(e_i, e_j)) for unit-norm rows (Lemma 4.2)."""
    sim = embeddings @ embeddings[kept].t()
    d_min = (1.0 - sim.clamp(-1.0, 1.0).max(dim=-1).values).clamp(min=0.0)
    return -d_min


def subset_objective(
    probs: torch.Tensor,
    potential: torch.Tensor,
    subset: torch.Tensor,
    lam: float,
    beta: float,
) -> float:
    """L_f(S) for an arbitrary nonempty subset of the candidate pool."""
    p_s = probs[subset]
    mass = float(p_s.sum().clamp(min=EPS))
    varphi = potential[subset] + lam * torch.log(probs[subset].clamp(min=EPS))
    return float((p_s * varphi).sum()) / mass + (beta - lam) * float(
        torch.log(torch.tensor(mass))
    )


def prefix_subset_update(
    probs: torch.Tensor,
    potential: torch.Tensor,
    lam: float,
    beta: float,
) -> torch.Tensor:
    """Exact S-step: scan prefixes of the varphi-sorted order (Theorem 3.4)."""
    varphi = potential + lam * torch.log(probs.clamp(min=EPS))
    order = torch.argsort(varphi, descending=True, stable=True)
    p_sorted = probs[order]
    mass_prefix = torch.cumsum(p_sorted, dim=0).clamp(min=EPS)
    value_prefix = torch.cumsum(p_sorted * varphi[order], dim=0)
    score = value_prefix / mass_prefix + (beta - lam) * torch.log(mass_prefix)
    k_star = int(torch.argmax(score)) + 1
    return order[:k_star].sort().values


def brute_force_subset_update(
    probs: torch.Tensor,
    potential: torch.Tensor,
    lam: float,
    beta: float,
) -> tuple[torch.Tensor, float]:
    """Enumerate every nonempty subset and return the exact argmax of L_f.

    Only feasible for small candidate pools (2^n - 1 subsets); used as
    the control for the prefix-form exact S-step.
    """
    pool = probs.shape[0]
    if pool > 16:
        raise ValueError("brute force is limited to pools of at most 16 tokens")
    best_subset, best_value = None, None
    for mask in range(1, 1 << pool):
        subset = torch.tensor(
            [i for i in range(pool) if mask >> i & 1], dtype=torch.long
        )
        value = subset_objective(probs, potential, subset, lam=lam, beta=beta)
        if best_value is None or value > best_value:
            best_subset, best_value = subset, value
    return best_subset, best_value


def top_w_mask(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    temperature: float = 1.0,
    top_m: int = 1200,
    init_top_p: float = 0.999,
    alt_iters: int = 9,
    geom_scale: float = 0.6,
    lam: float = 2.2,
    beta: float = 2.8,
) -> dict:
    """Alternating Top-W decoder (Section 4.2) over one logits vector.

    Returns the kept global token indices, the renormalized cropped
    distribution over the full vocabulary, and convergence diagnostics.
    """
    probs_full = F.softmax(logits / max(temperature, EPS), dim=-1)
    vocab = probs_full.shape[0]
    pool_size = min(max(top_m, 1), vocab)
    pool_probs, pool_index = torch.topk(probs_full, pool_size)
    pool_emb = whiten_and_normalize(embeddings)[pool_index]

    target_mass = min(max(init_top_p, 0.0), 1.0) * float(pool_probs.sum())
    cumulative = torch.cumsum(pool_probs, dim=0)
    k0 = int(torch.searchsorted(cumulative, torch.tensor(target_mass))) + 1
    kept = torch.arange(min(max(k0, 1), pool_size))

    iterations, converged = 0, False
    kept_sizes = []
    for iterations in range(1, max(alt_iters, 1) + 1):
        potential = geom_scale * nearest_set_potential(pool_emb, kept)
        new_kept = prefix_subset_update(pool_probs, potential, lam=lam, beta=beta)
        kept_sizes.append(int(new_kept.numel()))
        if new_kept.numel() == kept.numel() and bool((new_kept == kept).all()):
            converged = True
            kept = new_kept
            break
        kept = new_kept

    kept_global = pool_index[kept]
    probs = torch.zeros_like(probs_full)
    probs[kept_global] = probs_full[kept_global]
    return {
        "kept": kept_global.sort().values,
        "probs": probs / probs.sum(),
        "iterations": iterations,
        "converged": converged,
        "kept_sizes": kept_sizes,
    }


def min_p_filter(probs: torch.Tensor, min_p: float = 0.05) -> torch.Tensor:
    """Min-p truncation baseline."""
    mask = probs >= min_p * probs.max()
    filtered = probs * mask.float()
    return filtered / filtered.sum()


def top_p_filter(probs: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """Top-p (nucleus) truncation baseline."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove_sorted = cumulative > top_p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    remove = remove_sorted.scatter(0, sorted_indices, remove_sorted)
    filtered = probs.masked_fill(remove, 0.0)
    return filtered / filtered.sum()


def top_h_filter(logits: torch.Tensor, top_h_ratio: float = 0.8) -> torch.Tensor:
    """Top-H (entropy-bounded) truncation baseline."""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + EPS))
    target_mass = min(1.0, max(0.1, float(top_h_ratio * torch.exp(-entropy))))
    return top_p_filter(probs, top_p=target_mass)


def evaluate_decoding_metrics(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    temperature: float = 1.0,
    top_m: int = 128,
) -> dict[str, float]:
    """Entropy and kept-pool size for Top-W against the paper's baselines."""
    probs_orig = F.softmax(logits / temperature, dim=-1)
    distributions = {
        "top_w": top_w_mask(
            logits, embeddings, temperature=temperature, top_m=top_m
        )["probs"],
        "min_p": min_p_filter(probs_orig, min_p=0.05),
        "top_p": top_p_filter(probs_orig, top_p=0.9),
        "top_h": top_h_filter(logits / temperature, top_h_ratio=0.8),
    }

    def entropy(p: torch.Tensor) -> float:
        support = p[p > 0]
        return float(-torch.sum(support * torch.log(support)))

    metrics: dict[str, float] = {"entropy_orig": entropy(probs_orig)}
    for name, dist in distributions.items():
        metrics[f"entropy_{name}"] = entropy(dist)
        metrics[f"subset_size_{name}"] = float((dist > 1e-5).sum())
    return metrics
