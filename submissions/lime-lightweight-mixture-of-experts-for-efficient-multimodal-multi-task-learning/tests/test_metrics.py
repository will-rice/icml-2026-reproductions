"""
Unit tests for parameter counts, reduction ratios, and representation fidelity.
"""

import torch
from lime_peft.metrics import (
    compute_parameter_counts,
    compute_parameter_reduction_ratio,
    compute_representation_fidelity,
)

def test_parameter_counts():
    in_features = 4096
    out_features = 4096
    r = 8
    num_experts = 4

    lime_p, moe_p = compute_parameter_counts(in_features, out_features, r, num_experts)

    # Shared LoRA: (8 * 4096) + (4096 * 8) = 65,536
    # Expert vectors: 4 * 8 = 32
    assert lime_p == 65536 + 32

    # MoE-LoRA: 4 * 65,536 = 262,144
    assert moe_p == 262144

def test_parameter_reduction_ratio():
    in_features = 4096
    out_features = 4096
    r = 8
    num_experts = 4

    ratio = compute_parameter_reduction_ratio(in_features, out_features, r, num_experts)
    # Ratio should be ~3.998 (up to 4x)
    assert 3.9 < ratio < 4.0

def test_representation_fidelity():
    x1 = torch.randn(10, 64)
    x2 = x1 + torch.randn(10, 64) * 0.05

    fidelity = compute_representation_fidelity(x1, x2)
    assert 0.90 <= fidelity <= 1.0
