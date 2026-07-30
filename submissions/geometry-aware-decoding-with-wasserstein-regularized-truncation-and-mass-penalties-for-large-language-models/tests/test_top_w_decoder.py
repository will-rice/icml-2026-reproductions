from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top_w_repro.decoder import (
    compute_token_distance_matrix,
    evaluate_decoding_metrics,
    min_p_filter,
    top_h_filter,
    top_p_filter,
    top_w_filter,
)
from top_w_repro.evidence import build_bundle


def test_compute_token_distance_matrix():
    embeddings = torch.randn(10, 16)
    dist = compute_token_distance_matrix(embeddings, metric="cosine")
    assert dist.shape == (10, 10)
    assert torch.allclose(torch.diag(dist), torch.zeros(10), atol=1e-5)
    assert (dist >= 0).all()


def test_top_w_filter_properties():
    torch.manual_seed(42)
    logits = torch.randn(100) * 2.0
    embeddings = torch.randn(100, 32)
    
    probs = top_w_filter(logits, embeddings, temperature=0.7)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
    assert (probs >= 0).all()
    assert (probs > 1e-4).sum() > 0


def test_baseline_filters():
    torch.manual_seed(42)
    logits = torch.randn(50)
    probs = torch.softmax(logits, dim=-1)

    p_min = min_p_filter(probs, min_p=0.1)
    assert torch.isclose(p_min.sum(), torch.tensor(1.0), atol=1e-4)

    p_top_p = top_p_filter(probs, top_p=0.9)
    assert torch.isclose(p_top_p.sum(), torch.tensor(1.0), atol=1e-4)

    p_top_h = top_h_filter(logits, top_h_ratio=0.8)
    assert torch.isclose(p_top_h.sum(), torch.tensor(1.0), atol=1e-4)


def test_evaluate_decoding_metrics():
    torch.manual_seed(42)
    logits = torch.randn(40)
    embeddings = torch.randn(40, 16)
    metrics = evaluate_decoding_metrics(logits, embeddings, temperature=1.0)
    assert "entropy_top_w" in metrics
    assert "subset_size_top_w" in metrics
    assert metrics["entropy_top_w"] > 0


def test_evidence_bundle():
    bundle = build_bundle()
    assert bundle["paper_id"] == "HSuU4xBmAv"
    assert bundle["attempt_id"] == "572f7d7b-f6a5-4004-9389-22ac5af0d0f6"
    assert len(bundle["target_claims"]) == 3
    assert len(bundle["claim_results"]) == 3
    for cid, res in bundle["claim_results"].items():
        assert res["status"] in {"verified", "falsified", "toy", "inconclusive"}
