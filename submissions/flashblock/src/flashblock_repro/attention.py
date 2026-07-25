"""FlashBlock Attention module implementation."""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scaled dot-product attention and returns output and log-normalizer.

    Q: (B, H, N_q, d_k)
    K: (B, H, N_k, d_k)
    V: (B, H, N_k, d_k)

    Returns:
        A: Attention output (B, H, N_q, d_k)
        L: Log-normalizer (B, H, N_q, 1)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    m = torch.max(scores, dim=-1, keepdim=True).values  # (B, H, N_q, 1)
    # Handle all-masked cases gracefully
    m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

    exp_scores = torch.exp(scores - m)
    sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)  # (B, H, N_q, 1)

    L = m + torch.log(sum_exp + 1e-8)  # (B, H, N_q, 1)
    probs = exp_scores / (sum_exp + 1e-8)

    A = torch.matmul(probs, V)  # (B, H, N_q, d_k)
    return A, L

def log_space_attention_composition(
    A_out: torch.Tensor,
    L_out: torch.Tensor,
    A_in: torch.Tensor,
    L_in: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Composes block-external (A_out, L_out) and block-internal (A_in, L_in)
    attention in log space in a numerically stable manner.

    A_out: (B, H, N_q, d_k)
    L_out: (B, H, N_q, 1)
    A_in: (B, H, N_q, d_k)
    L_in: (B, H, N_q, 1)

    Returns:
        A_full: Combined attention output (B, H, N_q, d_k)
        L_full: Combined log-normalizer (B, H, N_q, 1)
    """
    m = torch.maximum(L_out, L_in)

    exp_out = torch.exp(L_out - m)
    exp_in = torch.exp(L_in - m)

    sum_exp = exp_out + exp_in
    L_full = m + torch.log(sum_exp + 1e-8)

    w_out = exp_out / (sum_exp + 1e-8)
    w_in = exp_in / (sum_exp + 1e-8)

    A_full = w_out * A_out + w_in * A_in
    return A_full, L_full

class BlockCausalAttentionCache:
    """Cache for block-external attention outputs (A_out, L_out) per layer."""

    def __init__(self, update_threshold: int = 2):
        self.update_threshold = update_threshold
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def has_cache(self, layer_idx: int) -> bool:
        return layer_idx in self.cache

    def get_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.has_cache(layer_idx):
            raise KeyError(f"No cache entry for layer {layer_idx}")
        return self.cache[layer_idx]

    def update_cache(self, layer_idx: int, A_out: torch.Tensor, L_out: torch.Tensor) -> None:
        self.cache[layer_idx] = (A_out.detach().clone(), L_out.detach().clone())

    def should_reuse_cache(self, layer_idx: int, num_updated_tokens: int) -> bool:
        """Reuse cache if layer cache exists AND updated token count in current step is below threshold."""
        return self.has_cache(layer_idx) and (num_updated_tokens < self.update_threshold)

    def clear(self) -> None:
        self.cache.clear()

class FlashBlockAttention(nn.Module):
    """
    Multi-head attention layer supporting FlashBlock block-external attention caching.
    """
    def __init__(self, embed_dim: int, num_heads: int, update_threshold: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.update_threshold = update_threshold

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x_in: torch.Tensor,
        x_out: Optional[torch.Tensor],
        layer_idx: int = 0,
        cache: Optional[BlockCausalAttentionCache] = None,
        num_updated_tokens: int = 0,
    ) -> torch.Tensor:
        """
        x_in: (B, N_in, C) query / current block representation
        x_out: (B, N_out, C) optional block-external context representation
        """
        B, N_in, C = x_in.shape

        # Project queries and internal K, V
        Q = self.q_proj(x_in).view(B, N_in, self.num_heads, self.d_k).transpose(1, 2)
        K_in = self.k_proj(x_in).view(B, N_in, self.num_heads, self.d_k).transpose(1, 2)
        V_in = self.v_proj(x_in).view(B, N_in, self.num_heads, self.d_k).transpose(1, 2)

        # Block-internal attention
        A_in, L_in = scaled_dot_product_attention(Q, K_in, V_in)

        if x_out is None or x_out.size(1) == 0:
            # No external context
            A_full = A_in
        else:
            N_out = x_out.size(1)

            should_reuse = cache is not None and cache.should_reuse_cache(layer_idx, num_updated_tokens)

            if should_reuse:
                A_out, L_out = cache.get_cache(layer_idx)
            else:
                # Compute block-external attention
                K_out = self.k_proj(x_out).view(B, N_out, self.num_heads, self.d_k).transpose(1, 2)
                V_out = self.v_proj(x_out).view(B, N_out, self.num_heads, self.d_k).transpose(1, 2)
                A_out, L_out = scaled_dot_product_attention(Q, K_out, V_out)

                if cache is not None:
                    cache.update_cache(layer_idx, A_out, L_out)

            A_full, _ = log_space_attention_composition(A_out, L_out, A_in, L_in)

        A_full = A_full.transpose(1, 2).contiguous().view(B, N_in, C)
        return self.out_proj(A_full)
