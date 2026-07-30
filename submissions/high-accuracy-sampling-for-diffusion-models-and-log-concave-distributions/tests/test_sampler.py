"""Unit tests for high-accuracy sampling verification."""

import json
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sampler import (
    verify_polylog_step_scaling,
    verify_intrinsic_dimension_scaling,
    verify_log_concave_gradient_sampler,
)


def test_polylog_step_scaling():
    res = verify_polylog_step_scaling([1e-2, 1e-4, 1e-6])
    assert res["verified"] is True
    assert res["polylog_exponent_estimate"] < 3.0
    assert len(res["step_data"]) == 3


def test_intrinsic_dimension_scaling():
    res = verify_intrinsic_dimension_scaling(d_star=5, full_d=500, delta=1e-3)
    assert res["verified"] is True
    assert pytest.approx(res["theoretical_speedup"]) == 100.0


def test_log_concave_gradient_sampler():
    res = verify_log_concave_gradient_sampler(dimension=3, target_accuracy=1e-2)
    assert res["verified"] is True
    assert res["empirical_mean_error"] < 0.3
    assert res["empirical_cov_error"] < 0.4


def test_evidence_bundle():
    bundle_path = Path(__file__).parent.parent / "evidence" / "bundle.json"
    if bundle_path.exists():
        data = json.loads(bundle_path.read_text())
        assert data["paper_id"] == "71132"
        assert len(data["claims"]) == 3
