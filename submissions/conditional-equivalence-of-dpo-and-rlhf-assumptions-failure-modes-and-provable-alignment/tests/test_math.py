import math
import pytest

from conditional_dpo_repro.math import (
    TwoResponsePolicy,
    bt_population_loss,
    constrained_rlhf_objective,
    cpo_loss,
    cpo_loss_derivative,
    cpo_optimal_policy_margin,
    cpo_reference_margin,
    dpo_loss,
    dpo_loss_derivative,
    logit,
    policy_from_delta,
    rlhf_optimal_delta,
    scaled_dpo_soft_margin,
    sigmoid,
    softplus,
    solve_exact_constrained_rlhf,
)


@pytest.mark.parametrize("delta", [-8.0, -1.0, 0.0, 1.0, 8.0])
def test_policy_delta_round_trip(delta):
    policy = policy_from_delta(delta)
    assert abs(policy.delta - delta) <= 1e-12
    assert abs(policy.preferred + policy.dispreferred - 1.0) <= 1e-12


def test_dpo_derivative_is_strictly_negative():
    assert dpo_loss_derivative(delta=-2.0, delta_ref=-3.0, beta=1.0) < 0.0
    assert dpo_loss(-2.0, -3.0, 1.0) < dpo_loss(-3.0, -3.0, 1.0)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((float("nan"), 0.0, 1.0), "finite"),
        ((0.0, 0.0, 0.0), "beta"),
    ],
)
def test_invalid_dpo_inputs_fail_closed(arguments, message):
    with pytest.raises(ValueError, match=message):
        dpo_loss(*arguments)


def test_exact_constrained_rlhf_solver_recovers_unconstrained_case():
    reference = TwoResponsePolicy(0.25, 0.75)
    exact = solve_exact_constrained_rlhf(reference, 0.5, 1.0, gamma=0.0)
    assert exact["status"] == "finite_optimum"
    assert abs(
        exact["policy"].delta
        - rlhf_optimal_delta(reference.delta, 0.5, 1.0)
    ) <= 1e-12


def test_exact_constrained_rlhf_solver_certifies_positive_gamma_boundary():
    reference = TwoResponsePolicy(0.25, 0.75)
    exact = solve_exact_constrained_rlhf(reference, 0.5, 1.0, gamma=0.10)
    assert exact["status"] == "unbounded"
    assert exact["approached_boundary"] == "preferred"
