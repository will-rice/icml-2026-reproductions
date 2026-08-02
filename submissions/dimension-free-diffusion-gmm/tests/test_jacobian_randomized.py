import json

import pytest

from diffusion_gmm_repro.jacobian import (
    random_mixture_case,
    run_jacobian_cell,
    summarize_jacobian,
)


def test_random_mixture_case_uses_full_available_mean_rank() -> None:
    case = random_mixture_case(
        components=8,
        ambient_dimension=64,
        geometry="imbalanced",
        seed=6,
    )
    assert case["active_rank"] == 7
    assert case["ambient_dimension"] == 64


@pytest.mark.parametrize(
    "geometry", ["balanced", "imbalanced", "overlapping", "separated"]
)
def test_random_mixture_case_labels_theorem_geometries(geometry: str) -> None:
    case = random_mixture_case(
        components=4,
        ambient_dimension=8,
        geometry=geometry,
        seed=4,
    )
    assert case["scope"] == "theorem-family"
    assert case["component_covariance"] == "identity"


def test_randomized_trace_matches_finite_difference_small_case() -> None:
    record = run_jacobian_cell(
        components=4,
        ambient_dimension=4,
        geometry="overlapping",
        steps=128,
        time=64,
        samples=256,
        seed=8,
        finite_difference_check=True,
    )
    assert record["finite_difference_max_error"] < 1e-5
    assert record["metric"] == "trace-i-plus-score-jacobian"


def test_jacobian_record_has_normalized_q95_and_is_serializable() -> None:
    record = run_jacobian_cell(
        components=8,
        ambient_dimension=16,
        geometry="separated",
        steps=128,
        time=64,
        samples=128,
        seed=9,
    )
    assert record["quantiles"]["q95"] >= 0.0
    assert record["q95_over_log_kt"] == pytest.approx(
        record["quantiles"]["q95"] / record["log_kt"]
    )
    json.dumps(record)


def test_anisotropic_control_remains_outside_theorem_results() -> None:
    theorem = run_jacobian_cell(
        components=2,
        ambient_dimension=4,
        geometry="balanced",
        steps=128,
        time=64,
        samples=64,
        seed=2,
    )
    control = run_jacobian_cell(
        components=2,
        ambient_dimension=4,
        geometry="anisotropic-control",
        steps=128,
        time=64,
        samples=64,
        seed=2,
    )
    summary = summarize_jacobian([theorem, control])
    assert control["scope"] == "out-of-scope-control"
    assert control["theorem_identity_applied"] is False
    assert summary["theorem_row_count"] == 1
    assert summary["control_row_count"] == 1


def test_jacobian_summary_rejects_nonfinite_theorem_ratio() -> None:
    summary = summarize_jacobian(
        [
            {
                "scope": "theorem-family",
                "q95_over_log_kt": float("nan"),
            }
        ]
    )
    assert summary["finite_support"] is False
    assert "finite-ratios" in summary["failed_rules"]


def test_jacobian_summary_fails_closed_on_missing_or_unknown_scope() -> None:
    summary = summarize_jacobian(
        [
            {"scope": "theorem-family", "q95_over_log_kt": 0.5},
            {"q95_over_log_kt": 0.1},
            {"scope": "unknown", "q95_over_log_kt": 0.1},
        ]
    )
    assert summary["theorem_row_count"] == 1
    assert summary["control_row_count"] == 0
    assert summary["excluded_scope_row_count"] == 2
    assert summary["finite_support"] is False
    assert "row-scope" in summary["failed_rules"]


def test_chunked_jacobian_matches_single_batch_small_case() -> None:
    arguments = {
        "components": 4,
        "ambient_dimension": 4,
        "geometry": "overlapping",
        "steps": 128,
        "time": 64,
        "samples": 256,
        "seed": 8,
        "finite_difference_check": True,
    }
    single_batch = run_jacobian_cell(**arguments, batch_size=256)
    chunked = run_jacobian_cell(**arguments, batch_size=17)
    assert chunked["quantiles"] == pytest.approx(single_batch["quantiles"])
    assert chunked["finite_difference_max_error"] == pytest.approx(
        single_batch["finite_difference_max_error"]
    )
    assert chunked["resource_method"] == "bounded-responsibility-batches"
    assert chunked["responsibility_chunk_size"] == 17
    assert chunked["responsibility_chunk_count"] == 16
