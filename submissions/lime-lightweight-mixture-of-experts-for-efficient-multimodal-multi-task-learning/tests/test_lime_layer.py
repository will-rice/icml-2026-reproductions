"""
Unit tests for LiMELayer and MoELoRABaseline.
"""

import torch
from lime_peft.lime_layer import LiMELayer, MoELoRABaseline

def test_lime_layer_forward():
    batch_size = 4
    seq_len = 16
    in_features = 64
    out_features = 128
    num_experts = 4
    r = 8

    layer = LiMELayer(in_features, out_features, r=r, num_experts=num_experts)
    x = torch.randn(batch_size, seq_len, in_features)
    weights = torch.softmax(torch.randn(batch_size, seq_len, num_experts), dim=-1)

    out = layer(x, weights)
    assert out.shape == (batch_size, seq_len, out_features)

def test_moe_lora_baseline_forward():
    batch_size = 4
    in_features = 64
    out_features = 128
    num_experts = 4
    r = 8

    layer = MoELoRABaseline(in_features, out_features, r=r, num_experts=num_experts)
    x = torch.randn(batch_size, in_features)
    weights = torch.softmax(torch.randn(batch_size, num_experts), dim=-1)

    out = layer(x, weights)
    assert out.shape == (batch_size, out_features)
