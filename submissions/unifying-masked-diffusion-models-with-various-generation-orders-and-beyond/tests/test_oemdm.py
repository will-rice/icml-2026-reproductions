"""Unit tests for OeMDM (Proposition 3.2 and Proposition 3.3)."""

import pytest
import torch
from unifying_mdm_repro.oemdm import OeMDMNELBO, verify_left_to_right_ar_recovery


def test_oemdm_nelbo_decomposition():
    vocab_size = 50
    seq_len = 10
    mask_token_id = 0
    model = OeMDMNELBO(vocab_size, seq_len, mask_token_id)

    B = 2
    logits = torch.randn(B, seq_len, vocab_size)
    targets = torch.randint(1, vocab_size, (B, seq_len))
    mask = torch.rand(B, seq_len) > 0.5
    pred_vel = torch.randn(B, seq_len)
    target_vel = torch.randn(B, seq_len)

    res = model.compute_nelbo(logits, targets, mask, pred_vel, target_vel)

    assert "total_nelbo" in res
    assert "reconstruction_loss" in res
    assert "velocity_mismatch_loss" in res
    assert torch.isclose(res["total_nelbo"], res["decomposed_sum"])


def test_proposition_33_ar_recovery():
    res = verify_left_to_right_ar_recovery(seq_len=8, batch_size=2)
    assert res["verified"] is True
    assert res["all_masked_at_start"] is True
    assert res["none_masked_at_end"] is True
    assert res["exact_l2r_step_unmasking"] is True
