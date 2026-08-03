import numpy as np
import pytest

from timerewarder_repro.theory import (
    audit_theory,
    bellman_residual,
    discounted_time_to_goal,
)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.9, 0.99, 1.0])
def test_closed_form_matches_bellman_recurrence(gamma: float) -> None:
    values = np.asarray(
        [discounted_time_to_goal(remaining, gamma) for remaining in range(10)],
        dtype=np.float64,
    )
    assert np.max(np.abs(bellman_residual(values, gamma))) <= 1e-12


def test_gamma_one_is_negative_remaining_distance() -> None:
    assert [discounted_time_to_goal(step, 1.0) for step in range(6)] == [
        -0.0,
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
    ]


def test_audit_records_assumptions_and_aliasing_counterexample() -> None:
    result = audit_theory()

    assert result["all_checks_pass"] is True
    assert result["max_absolute_bellman_residual"] <= 1e-12
    assert len(result["checks"]) == 6 * 5
    assert set(result["assumptions"]) == {
        "fully observable",
        "deterministic transitions",
        "optimal expert trajectory",
        "terminal goal",
        "observations uniquely identify phase/state",
    }
    counterexample = result["aliasing_counterexample"]
    assert counterexample["observations"] == ["o0", "o1", "o2", "o3", "o1", "og"]
    assert counterexample["o1_remaining_distances"] == [4, 1]
    assert counterexample["single_frame_average"] == 2.5


@pytest.mark.parametrize(
    ("remaining", "gamma"),
    [(-1, 0.9), (1, -0.1), (1, 1.1)],
)
def test_theory_rejects_invalid_bounds(remaining: int, gamma: float) -> None:
    with pytest.raises(ValueError):
        discounted_time_to_goal(remaining, gamma)
