from itertools import pairwise
import pytest

from conditional_dpo_repro.cpo import run_cpo_margin_lane
from conditional_dpo_repro.math import (
    TwoResponsePolicy,
    cpo_optimal_policy_margin,
    cpo_reference_margin,
    rlhf_optimal_delta,
    solve_exact_constrained_rlhf,
)


def test_exact_constrained_rlhf_gamma_zero_recovers_rlhf_optimum():
    reference = TwoResponsePolicy(0.25, 0.75)
    exact = solve_exact_constrained_rlhf(reference, 0.5, 1.0, gamma=0.0)
    assert exact["status"] == "finite_optimum"
    assert abs(
        exact["policy"].delta
        - rlhf_optimal_delta(reference.delta, 0.5, 1.0)
    ) <= 1e-12
    assert exact["first_order_residual"] <= 1e-12


def test_positive_gamma_exact_objective_is_not_replaced_by_equation_17():
    reference = TwoResponsePolicy(0.25, 0.75)
    exact = solve_exact_constrained_rlhf(reference, 0.5, 1.0, gamma=0.10)
    assert exact["status"] == "unbounded"
    assert exact["approached_boundary"] == "preferred"
    assert all(
        right > left
        for left, right in pairwise(exact["increasing_tail_values"])
    )
    approximation = cpo_reference_margin(reference, gamma=0.10)
    assert abs(approximation - (0.10 * (4.0 + 4.0 / 3.0))) <= 1e-12


def test_equations_13_16_margin_uses_certified_optimal_policy():
    optimal_policy = TwoResponsePolicy(0.80, 0.20)
    reference = TwoResponsePolicy(0.25, 0.75)
    exact_margin = cpo_optimal_policy_margin(optimal_policy, gamma=0.10)
    approximate_margin = cpo_reference_margin(reference, gamma=0.10)
    assert abs(exact_margin - 0.10 * (1.25 + 5.00)) <= 1e-12
    assert exact_margin != approximate_margin


def test_cpo_lane_keeps_exact_result_and_approximation_separate():
    result = run_cpo_margin_lane()
    assert result["case_count"] == 180
    assert result["exact_constrained_rlhf"]["finite_optimum_count"] == 45
    assert result["exact_constrained_rlhf"]["unbounded_count"] == 135
    assert result[
        "equations_13_16_optimal_policy_margin"
    ]["evaluated_only_for_certified_optima"] is True
    approximation = result["equation_17_reference_policy_approximation"]
    assert approximation["case_count"] == 180
    assert approximation["shift_identity_max_abs_error"] <= 1e-12
    assert approximation["stationary_derivative_max_abs_error"] <= 1e-8
    assert approximation["margin_parameter_derivative"] == 0.0
    assert approximation["labeled_approximation"] is True
