"""
Zero-parameter routing mechanisms and adaptive top-k expert selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroParamRouter(nn.Module):
    """
    Zero-parameter router using fixed reference prototypes / token similarity.
    Does NOT introduce trainable router parameters.
    """

    def __init__(self, in_features: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.in_features = in_features
        self.num_experts = num_experts
        self.top_k = top_k

        # Fixed orthogonal expert prototype vectors (untrainable / zero parameter count)
        prototypes = torch.randn(num_experts, in_features)
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        self.register_buffer("prototypes", prototypes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes routing weights via zero-parameter prototype similarity.
        x: (..., in_features)
        Returns: (..., num_experts)
        """
        x_norm = F.normalize(x, p=2, dim=-1)
        # Cosine similarity: (..., num_experts)
        sim = torch.matmul(x_norm, self.prototypes.T)

        # Adaptive top-k selection
        topk_sim, topk_idx = torch.topk(sim, self.top_k, dim=-1)
        probs = F.softmax(topk_sim, dim=-1)

        weights = torch.zeros_like(sim)
        weights.scatter_(-1, topk_idx, probs)
        return weights


def adaptive_top_k_select(logits: torch.Tensor, min_k: int = 1, max_k: int = 2, threshold: float = 0.8) -> torch.Tensor:
    """
    Adaptive expert selection: selects between min_k and max_k experts based on cumulative probability.
    """
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Determine k for each sample
    k_mask = (cum_probs >= threshold)
    # Get first index where cum_probs >= threshold
    first_pass = (k_mask.cumsum(dim=-1) == 1)
    selected_k = first_pass.long().argmax(dim=-1) + 1
    selected_k = torch.clamp(selected_k, min=min_k, max=max_k)

    weights = torch.zeros_like(probs)
    for b in range(probs.size(0)):
        k = selected_k[b].item()
        top_idx = sorted_indices[b, :k]
        top_probs = F.softmax(logits[b, top_idx], dim=-1)
        weights[b, top_idx] = top_probs

    return weights
