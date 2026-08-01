"""
Unit tests for zero-parameter router and adaptive top-k expert selection.
"""

import torch
from lime_peft.routing import ZeroParamRouter, adaptive_top_k_select

def test_zero_param_router():
    batch_size = 8
    in_features = 32
    num_experts = 4
    top_k = 2

    router = ZeroParamRouter(in_features, num_experts=num_experts, top_k=top_k)
    x = torch.randn(batch_size, in_features)

    weights = router(x)
    assert weights.shape == (batch_size, num_experts)
    # Check that exactly top_k experts have non-zero weights per sample
    non_zeros = (weights > 0.0).sum(dim=-1)
    assert torch.all(non_zeros == top_k)

def test_adaptive_top_k_select():
    batch_size = 4
    logits = torch.tensor([
        [10.0, 1.0, 0.0, 0.0],  # Peak probability -> 1 expert suffice
        [2.5, 2.4, 0.1, 0.1],   # Distributed -> 2 experts required
        [5.0, 4.0, 1.0, 0.0],   # Top 2
        [1.0, 1.0, 1.0, 1.0],   # Uniform
    ])

    weights = adaptive_top_k_select(logits, min_k=1, max_k=2, threshold=0.8)
    assert weights.shape == logits.shape
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
