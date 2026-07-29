"""Tests for FlexRank reproduction evidence."""

import json
from pathlib import Path
import pytest

from flexrank_repro.evidence import (
    factorize_linear_layer,
    dynamic_programming_component_ordering,
    verify_theorem_4_1_svd_truncation_failure,
    verify_theorem_4_3_nested_minimizer_preservation,
    run_evidence_generation,
)


def test_linear_layer_factorization():
    import numpy as np
    np.random.seed(0)
    W = np.random.randn(16, 32)
    U, V = factorize_linear_layer(W, rank=4)
    assert U.shape == (16, 4)
    assert V.shape == (4, 32)


def test_dp_component_ordering():
    layer_costs = [
        [0.0, 10.0, 5.0],
        [0.0, 8.0, 4.0],
    ]
    alloc = dynamic_programming_component_ordering(layer_costs, budget_rank=3)
    assert len(alloc) == 2
    assert sum(alloc) <= 3


def test_theorem_4_1_svd_truncation_failure():
    res = verify_theorem_4_1_svd_truncation_failure(seed=42)
    assert res["theorem"] == "4.1"
    assert res["verified"] is True
    assert res["suboptimality_gap"] > 0.0


def test_theorem_4_3_nested_minimizer_preservation():
    res = verify_theorem_4_3_nested_minimizer_preservation(seed=42)
    assert res["theorem"] == "4.3"
    assert res["verified"] is True
    assert res["max_minimizer_error_diff"] < 1e-10


def test_evidence_bundle_generation(tmp_path):
    bundle = run_evidence_generation(tmp_path)
    bundle_file = tmp_path / "bundle.json"
    assert bundle_file.exists()
    data = json.loads(bundle_file.read_text())
    assert data["paper_id"] == "DK0kvnNelx"
    assert "claims_verdict" in data


def test_pages_precondition_substantive_length(project_root):
    pages_dir = project_root / "pages"
    assert pages_dir.exists(), "pages directory must exist"
    page_files = list(pages_dir.glob("*.md"))
    assert len(page_files) > 0, "at least one markdown page must exist"
    total_length = sum(len(f.read_text(encoding="utf-8").strip()) for f in page_files)
    assert total_length >= 200, f"pages content length ({total_length}) must be >= 200 characters"
