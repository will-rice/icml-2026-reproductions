import json

import numpy as np
import pytest

from diffusion_gmm_repro.assumptions import (
    check_mean_growth,
    contaminated_target_tv,
)
from diffusion_gmm_repro.convergence import gmm_family


def test_contamination_tv_scales_exactly_with_epsilon() -> None:
    base = gmm_family("rank1-k2", seed=2)
    unit = contaminated_target_tv(base, epsilon=1.0, grid_points=65537)
    tenth = contaminated_target_tv(base, epsilon=0.1, grid_points=65537)
    assert tenth["tv"] == pytest.approx(0.1 * unit["tv"], abs=1e-15)
    assert tenth["tail_error_bound"] < 1e-6


def test_zero_contamination_has_zero_tv() -> None:
    base = gmm_family("rank1-k2", seed=2)
    record = contaminated_target_tv(base, epsilon=0.0, grid_points=4097)
    assert record["tv"] == 0.0
    assert record["numerical_integral"] == 0.0
    assert record["scope"] == "theorem-family"


def test_contamination_record_is_certified_and_json_serializable() -> None:
    base = gmm_family("rank1-k2", seed=3)
    record = contaminated_target_tv(base, epsilon=0.02, grid_points=4097)
    assert record["quadrature"] == "symmetric-trapezoid"
    assert record["tail_error_bound"] < 1e-6
    assert record["integration_error_bound"] >= 0.0
    assert record["total_error_bound"] < record["certificate_tolerance"]
    assert record["integration_radius"] > np.max(np.abs(base.means))
    json.dumps(record)


def test_coarse_initial_grid_refines_to_certified_tv() -> None:
    base = gmm_family("rank1-k2", seed=2)
    coarse = contaminated_target_tv(base, epsilon=1.0, grid_points=3)
    reference = contaminated_target_tv(base, epsilon=1.0, grid_points=65537)
    difference = abs(coarse["tv"] - reference["tv"])
    assert coarse["grid_points_used"] > coarse["initial_grid_points"]
    assert 0.0 <= coarse["tv"] <= 1.0
    assert difference <= coarse["total_error_bound"]
    assert coarse["total_error_bound"] < coarse["certificate_tolerance"]


def test_mean_growth_checker_has_violating_control() -> None:
    assert check_mean_growth(
        means=np.array([[2.0], [-2.0]]), steps=128, c_r=1.0
    )["satisfied"]
    assert not check_mean_growth(
        means=np.array([[129.0], [-129.0]]), steps=128, c_r=1.0
    )["satisfied"]


def test_mean_growth_uses_assumption_one_power_threshold() -> None:
    passing = check_mean_growth(
        means=np.array([[10.0], [-10.0]]), steps=128, c_r=0.5
    )
    violating = check_mean_growth(
        means=np.array([[12.0], [-12.0]]), steps=128, c_r=0.5
    )
    assert passing["threshold"] == pytest.approx(np.sqrt(128.0))
    assert passing["satisfied"] is True
    assert violating["satisfied"] is False
    assert violating["scope"] == "out-of-scope-control"


def test_mean_growth_control_is_explicitly_out_of_scope() -> None:
    record = check_mean_growth(
        means=np.array([[129.0], [-129.0]]), steps=128, c_r=1.0
    )
    assert record["scope"] == "out-of-scope-control"
    assert record["max_mean_norm"] == 129.0
