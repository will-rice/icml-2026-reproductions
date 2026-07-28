import itertools
import math

from conditional_dpo_repro.equivalence import centered_derivative
from conditional_dpo_repro.grids import (
    CPO_BETAS,
    CPO_GAMMAS,
    CPO_P_REF_W,
    CPO_REWARD_GAPS,
)
from conditional_dpo_repro.math import (
    TwoResponsePolicy,
    cpo_loss,
    cpo_optimal_policy_margin,
    cpo_reference_margin,
    equations_13_16_residual,
    rlhf_optimal_delta,
    sigmoid,
    softplus,
    solve_exact_constrained_rlhf,
)


def serialize_exact_result(exact: dict[str, object]) -> dict[str, object]:
    if exact["status"] == "finite_optimum":
        policy = exact["policy"]
        return {
            "status": "finite_optimum",
            "policy": {
                "preferred": policy.preferred,
                "dispreferred": policy.dispreferred,
                "delta": policy.delta,
            },
            "objective": exact["objective"],
            "first_order_residual": exact["first_order_residual"],
            "curvature": exact["curvature"],
        }
    return {
        "status": "unbounded",
        "approached_boundary": exact["approached_boundary"],
        "increasing_tail_values": list(exact["increasing_tail_values"]),
        "analytic_reason": exact["analytic_reason"],
    }


def derive_cpo_outcome(
    finite_optimum_count: int,
    unbounded_count: int,
    shift_identity_max_abs_error: float,
    stationary_derivative_max_abs_error: float,
) -> str:
    if (
        unbounded_count > 0
        and finite_optimum_count > 0
        and shift_identity_max_abs_error <= 1e-6
        and stationary_derivative_max_abs_error <= 1e-6
    ):
        return "mixed"
    if unbounded_count > 0:
        return "contradiction"
    return "consistent"


def _summarize_cpo(rows: list[dict[str, object]]) -> dict[str, object]:
    finite_optimum_count = sum(
        1 for r in rows if r["exact_constrained_rlhf"]["status"] == "finite_optimum"
    )
    unbounded_count = sum(
        1 for r in rows if r["exact_constrained_rlhf"]["status"] == "unbounded"
    )
    evaluated_optima_count = sum(
        1
        for r in rows
        if r["equations_13_16_optimal_policy_margin"]["defined"]
    )

    max_shift_err = max(
        r["equation_17_reference_policy_approximation"]["shift_identity_abs_error"]
        for r in rows
    )
    max_stationary_err = max(
        r["equation_17_reference_policy_approximation"]["stationary_derivative_abs_error"]
        for r in rows
    )

    outcome = derive_cpo_outcome(
        finite_optimum_count,
        unbounded_count,
        max_shift_err,
        max_stationary_err,
    )

    return {
        "case_count": len(rows),
        "exact_constrained_rlhf": {
            "finite_optimum_count": finite_optimum_count,
            "unbounded_count": unbounded_count,
        },
        "equations_13_16_optimal_policy_margin": {
            "evaluated_only_for_certified_optima": True,
            "evaluated_count": evaluated_optima_count,
        },
        "equation_17_reference_policy_approximation": {
            "case_count": len(rows),
            "shift_identity_max_abs_error": max_shift_err,
            "stationary_derivative_max_abs_error": max_stationary_err,
            "margin_parameter_derivative": 0.0,
            "labeled_approximation": True,
        },
        "cases": rows,
        "outcome": outcome,
    }


def run_cpo_margin_lane() -> dict[str, object]:
    rows = []
    for p_ref, reward_gap, beta, gamma in itertools.product(
        CPO_P_REF_W, CPO_REWARD_GAPS, CPO_BETAS, CPO_GAMMAS
    ):
        reference = TwoResponsePolicy(p_ref, 1.0 - p_ref)

        # Stage 1: solve or classify the exact constrained-RLHF objective.
        exact = solve_exact_constrained_rlhf(reference, reward_gap, beta, gamma)

        # Stage 2: audit Equations 13–16 only for a certified global optimum.
        exact_margin = None
        exact_margin_residual = None
        if exact["status"] == "finite_optimum":
            exact_margin = cpo_optimal_policy_margin(exact["policy"], gamma)
            exact_margin_residual = equations_13_16_residual(
                exact["policy"],
                reference,
                reward_gap,
                beta,
                exact_margin,
            )

        # Stage 3: audit Equation 17 as a reference-policy approximation.
        reference_margin = cpo_reference_margin(reference, gamma)
        dpo_delta = rlhf_optimal_delta(reference.delta, reward_gap, beta)
        approximate_cpo_delta = dpo_delta + reference_margin / beta

        expected_shifted_delta = reference.delta + (reward_gap + reference_margin) / beta
        shift_identity_abs_error = abs(approximate_cpo_delta - expected_shifted_delta)

        derivative = centered_derivative(
            lambda delta: (
                sigmoid(reward_gap)
                * cpo_loss(delta, reference.delta, beta, reference_margin)
                + (1.0 - sigmoid(reward_gap))
                * softplus(beta * (delta - reference.delta) - reference_margin)
            ),
            approximate_cpo_delta,
        )

        rows.append(
            {
                "p_ref_w": p_ref,
                "reward_gap": reward_gap,
                "beta": beta,
                "gamma": gamma,
                "exact_constrained_rlhf": serialize_exact_result(exact),
                "equations_13_16_optimal_policy_margin": {
                    "defined": exact_margin is not None,
                    "margin": exact_margin,
                    "residual": exact_margin_residual,
                },
                "equation_17_reference_policy_approximation": {
                    "reference_margin": reference_margin,
                    "approximate_cpo_delta": approximate_cpo_delta,
                    "shift_identity_abs_error": shift_identity_abs_error,
                    "stationary_derivative_abs_error": abs(derivative),
                },
            }
        )
    return _summarize_cpo(rows)
