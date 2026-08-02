import json

import numpy as np
import pytest

from diffusion_gmm_repro.score_error import (
    run_score_error_cell,
    summarize_score_error,
)


@pytest.mark.parametrize("shape", ["uniform", "front-loaded", "back-loaded"])
def test_score_error_cells_record_exact_time_average(shape: str) -> None:
    record = run_score_error_cell(
        family="rank1-k2",
        steps=128,
        seed=4,
        samples=2048,
        epsilon_score=0.04,
        profile_shape=shape,
    )
    assert record["rms_score_error"] == pytest.approx(0.04, abs=1e-12)
    assert record["bound_scale"] == pytest.approx(0.04 * np.sqrt(np.log(128)))


def test_zero_score_error_reuses_exact_noise_stream() -> None:
    record = run_score_error_cell(
        family="rank2-k4",
        steps=128,
        seed=5,
        samples=128,
        epsilon_score=0.0,
        profile_shape="uniform",
    )
    assert record["paired_excess"]["rms_l2"] == 0.0
    assert record["normalized_ratio"] == 0.0
    assert record["pairing"] == "identical-start-and-noise-streams"


def test_score_error_cell_is_deterministic_and_json_serializable() -> None:
    arguments = {
        "family": "rank1-k2",
        "steps": 128,
        "seed": 7,
        "samples": 64,
        "epsilon_score": 0.02,
        "profile_shape": "back-loaded",
    }
    first = run_score_error_cell(**arguments)
    second = run_score_error_cell(**arguments)
    assert first == second
    json.dumps(first)


def synthetic_score_rows(
    *, excess: tuple[float, ...] = (0.0, 0.01, 0.02, 0.04)
) -> list[dict[str, object]]:
    return [
        {
            "family": "rank1-k2",
            "profile_shape": "uniform",
            "seed": index,
            "epsilon_score": epsilon,
            "bound_scale": epsilon,
            "paired_excess": {"rms_l2": value},
            "normalized_ratio": 0.0 if epsilon == 0.0 else value / epsilon,
            "scope": "theorem-family",
        }
        for index, (epsilon, value) in enumerate(
            zip((0.0, 0.01, 0.02, 0.04), excess, strict=True)
        )
    ]


def test_score_error_summary_accepts_monotone_bounded_response() -> None:
    summary = summarize_score_error(
        synthetic_score_rows(), seed=9, bootstrap_replicates=200
    )
    assert summary["finite_support"] is True
    assert summary["failed_rules"] == []
    assert summary["acceptance_rule"] == "two-times-pooled-normalized-ratio"


def test_score_error_summary_rejects_nonmonotone_response() -> None:
    summary = summarize_score_error(
        synthetic_score_rows(excess=(0.0, 0.03, 0.02, 0.05)),
        seed=9,
        bootstrap_replicates=200,
    )
    assert summary["finite_support"] is False
    assert "monotonicity" in summary["failed_rules"]


def test_score_error_summary_excludes_controls_from_theorem_result() -> None:
    rows = synthetic_score_rows()
    control = dict(rows[-1])
    control["scope"] = "out-of-scope-control"
    control["normalized_ratio"] = 1000.0
    summary = summarize_score_error(
        [*rows, control], seed=9, bootstrap_replicates=200
    )
    assert summary["finite_support"] is True
    assert summary["theorem_row_count"] == 4
    assert summary["control_row_count"] == 1


def test_score_error_summary_fails_closed_on_missing_or_unknown_scope() -> None:
    rows = synthetic_score_rows()
    missing_scope = dict(rows[-1])
    missing_scope.pop("scope")
    unknown_scope = dict(rows[-1])
    unknown_scope["scope"] = "unknown"
    summary = summarize_score_error(
        [*rows, missing_scope, unknown_scope],
        seed=9,
        bootstrap_replicates=200,
    )
    assert summary["theorem_row_count"] == 4
    assert summary["excluded_scope_row_count"] == 2
    assert summary["finite_support"] is False
    assert "row-scope" in summary["failed_rules"]


def test_score_error_uses_per_profile_normalized_upper_bound() -> None:
    rows: list[dict[str, object]] = []
    for index, epsilon in enumerate((0.001, 0.002)):
        rows.append(
            {
                "family": "rank1-k2",
                "profile_shape": "uniform",
                "seed": index,
                "epsilon_score": epsilon,
                "bound_scale": epsilon,
                "paired_excess": {"rms_l2": 3.0 * epsilon},
                "normalized_ratio": 3.0,
                "scope": "theorem-family",
            }
        )
    for index, epsilon in enumerate((0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
        rows.append(
            {
                "family": "rank1-k2",
                "profile_shape": "back-loaded",
                "seed": 10 + index,
                "epsilon_score": epsilon,
                "bound_scale": epsilon,
                "paired_excess": {"rms_l2": epsilon},
                "normalized_ratio": 1.0,
                "scope": "theorem-family",
            }
        )

    assert max(
        row["paired_excess"]["rms_l2"]
        for row in rows
        if row["profile_shape"] == "uniform"
    ) < max(
        row["paired_excess"]["rms_l2"]
        for row in rows
        if row["profile_shape"] == "back-loaded"
    )
    summary = summarize_score_error(rows, seed=13, bootstrap_replicates=200)
    uniform = next(
        item
        for item in summary["profile_ratio_bounds"]
        if item["profile_shape"] == "uniform"
    )
    assert summary["pooled_normalized_ratio"] == 1.0
    assert uniform["upper_95"] == 3.0
    assert uniform["passed"] is False
    assert "normalized-ratio-upper-95" in summary["failed_rules"]
