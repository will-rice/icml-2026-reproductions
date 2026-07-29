"""Unit tests for LoMDM joint training (Section 4.1)."""

import pytest
import torch
from unifying_mdm_repro.lomdm import LoMDMBackbone, verify_lomdm_joint_training


def test_lomdm_backbone_forward():
    vocab_size = 40
    seq_len = 8
    hidden_dim = 16
    model = LoMDMBackbone(vocab_size, seq_len, hidden_dim)

    B = 2
    x_t = torch.randint(0, vocab_size, (B, seq_len))
    mask = torch.rand(B, seq_len) > 0.5

    out = model(x_t, mask)
    assert out["token_logits"].shape == (B, seq_len, vocab_size)
    assert out["order_logits"].shape == (B, seq_len)


def test_lomdm_joint_training_verification():
    res = verify_lomdm_joint_training(vocab_size=50, seq_len=8, hidden_dim=16)
    assert res["verified"] is True
    assert res["final_joint_loss"] < res["initial_joint_loss"]
    assert res["single_objective_unified"] is True
