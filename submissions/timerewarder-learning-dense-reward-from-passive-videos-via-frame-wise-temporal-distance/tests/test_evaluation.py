import numpy as np
import pytest

from timerewarder_repro.evaluation import (
    compute_distance_metrics,
    cumulative_anchor_values,
    task_passes,
    tie_aware_spearman,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0.0, 1.0, 2.0], 1.0),
        ([2.0, 1.0, 0.0], -1.0),
        ([1.0, 1.0, 1.0], 0.0),
        ([1.0, 1.0, 3.0], np.sqrt(3.0) / 2.0),
    ],
)
def test_tie_aware_spearman_matches_hand_cases(values, expected) -> None:
    observed = tie_aware_spearman(
        np.asarray(values, dtype=np.float64),
        np.arange(len(values), dtype=np.float64),
    )
    assert observed == pytest.approx(expected)


def test_cumulative_values_use_forward_minus_reverse_rewards() -> None:
    values = cumulative_anchor_values(
        np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float64),
        np.array([-0.2, -0.3, -0.4, -0.5], dtype=np.float64),
    )

    assert values == pytest.approx([0.0, 0.4, 1.0, 1.8, 2.8])


def test_distance_metrics_are_float64_and_include_antisymmetry() -> None:
    target = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float64)
    prediction = np.array([0.4, -0.4, 0.2, -0.2], dtype=np.float64)

    metrics = compute_distance_metrics(prediction, target)

    assert metrics["prediction_mae"] == pytest.approx(0.075)
    assert metrics["zero_baseline_mae"] == pytest.approx(0.375)
    assert metrics["relative_improvement"] == pytest.approx(0.8)
    assert metrics["sign_accuracy"] == 1.0
    assert metrics["mean_antisymmetry_error"] == 0.0


def test_task_passes_uses_preregistered_thresholds_and_tolerance() -> None:
    passing = {
        "prediction_mae": 0.200001,
        "relative_improvement": 0.099999,
        "sign_accuracy": 0.799999,
        "mean_antisymmetry_error": 0.150001,
    }
    assert task_passes(passing, tolerance=1e-6)

    failing = passing | {"prediction_mae": 0.200002}
    assert not task_passes(failing, tolerance=1e-6)
