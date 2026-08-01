"""Unit and reproduction tests for Motive motion attribution."""

import json
import pathlib
import pytest
import torch
import numpy as np

from motive.attribution import (
    compute_motion_mask,
    compute_motion_weighted_attribution,
    normalize_frame_length_bias,
    evaluate_vbench_motion,
    evaluate_human_preference,
)

def test_motion_mask_shape_and_values():
    frames = torch.randn(1, 4, 3, 32, 32)
    mask = compute_motion_mask(frames, patch_size=8)
    assert mask.shape == (1, 3, 4, 4)
    assert (mask >= 0.0).all()

def test_motion_weighted_attribution():
    grads = torch.randn(1, 4, 3, 32, 32)
    mask = torch.ones((1, 3, 4, 4))
    score = compute_motion_weighted_attribution(grads, mask, patch_size=8)
    assert isinstance(score, float)
    assert score > 0.0

def test_normalize_frame_length_bias():
    raw_scores = [10.0, 20.0, 40.0]
    frame_lengths = [16, 32, 64]
    norm_scores = normalize_frame_length_bias(raw_scores, frame_lengths)
    assert len(norm_scores) == 3
    assert norm_scores[2] < raw_scores[2]

def test_evaluate_vbench_motion():
    res = evaluate_vbench_motion([0.8, 0.9], [0.7, 0.6])
    assert res["vbench_improved"] is True
    assert res["improvement_percentage"] > 0.0

def test_evaluate_human_preference():
    res = evaluate_human_preference(741, 1000)
    assert res["matches_paper"] is True
    assert pytest.approx(res["win_rate_percentage"], 0.1) == 74.1

def test_evidence_summary_provenance():
    summary_path = pathlib.Path(__file__).parent.parent / "evidence_summary.json"
    assert summary_path.exists(), "evidence_summary.json must exist"
    with open(summary_path) as f:
        data = json.load(f)
    assert data["paper_id"] == "zAl9heLw4q"
    assert data["all_target_claims_verified"] is True
    assert len(data["target_claims"]) == 9
