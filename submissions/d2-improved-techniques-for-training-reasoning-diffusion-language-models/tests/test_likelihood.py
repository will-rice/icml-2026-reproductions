import math

from d2_repro.evidence import (
    anyorder_log_likelihood,
    enumerate_anyorder_log_likelihood,
    stepmerge_approximation,
)


def test_anyorder_single_pass_matches_order_enumeration():
    probabilities = [
        {"A": 0.70, "B": 0.30},
        {"A": 0.20, "B": 0.80},
        {"A": 0.55, "B": 0.45},
    ]
    target = ["A", "B", "A"]

    single = anyorder_log_likelihood(probabilities, target)
    enumerated = enumerate_anyorder_log_likelihood(probabilities, target)

    assert math.isclose(single["log_likelihood"], enumerated["log_likelihood"], abs_tol=1e-12)
    assert single["model_passes"] == 1
    assert enumerated["orders_checked"] == 6


def test_stepmerge_converges_to_exact_as_groups_get_finer():
    step_logps = [-0.08, -0.11, -0.28, -0.31, -0.72, -0.77, -1.10, -1.18]

    coarse = stepmerge_approximation(step_logps, groups=2)
    medium = stepmerge_approximation(step_logps, groups=4)
    exact = stepmerge_approximation(step_logps, groups=8)

    assert coarse["model_passes"] == 2
    assert medium["model_passes"] == 4
    assert exact["model_passes"] == 8
    assert abs(medium["error_vs_exact"]) < abs(coarse["error_vs_exact"])
    assert exact["error_vs_exact"] == 0.0
