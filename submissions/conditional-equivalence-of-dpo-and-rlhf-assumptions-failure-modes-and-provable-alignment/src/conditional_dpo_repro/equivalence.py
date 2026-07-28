import itertools

from conditional_dpo_repro.grids import (
    EQUIVALENCE_BETAS,
    EQUIVALENCE_P_REF_W,
    EQUIVALENCE_REWARD_GAPS,
)
from conditional_dpo_repro.math import (
    TwoResponsePolicy,
    bt_population_loss,
    dpo_loss_derivative,
    rlhf_optimal_delta,
)


def centered_derivative(function, point: float, step: float = 1e-6) -> float:
    return (function(point + step) - function(point - step)) / (2.0 * step)


def derive_equivalence_outcome(
    max_stationary_err: float,
    max_loss_deriv: float,
    one_sided_finite_optimum: bool,
) -> str:
    if max_stationary_err <= 1e-6 and not one_sided_finite_optimum and max_loss_deriv < 0.0:
        return "mixed"
    if max_stationary_err <= 1e-6:
        return "consistent"
    return "contradiction"


def _summarize_equivalence(rows: list[dict[str, object]]) -> dict[str, object]:
    max_stationary_err = max(r["population_stationary_abs_error"] for r in rows)
    max_loss_deriv = max(r["positive_loss_derivative"] for r in rows)
    outcome = derive_equivalence_outcome(max_stationary_err, max_loss_deriv, False)

    return {
        "case_count": len(rows),
        "population_stationary_max_abs_error": max_stationary_err,
        "positive_loss_derivative_max": max_loss_deriv,
        "one_sided_finite_optimum": False,
        "population_identity_requires_positive_delta": False,
        "cases": rows,
        "outcome": outcome,
    }


def run_equivalence_lane() -> dict[str, object]:
    rows = []
    for p_ref, reward_gap, beta in itertools.product(
        EQUIVALENCE_P_REF_W, EQUIVALENCE_REWARD_GAPS, EQUIVALENCE_BETAS
    ):
        delta_ref = TwoResponsePolicy(p_ref, 1.0 - p_ref).delta
        optimum = rlhf_optimal_delta(delta_ref, reward_gap, beta)
        derivative = centered_derivative(
            lambda delta: bt_population_loss(delta, delta_ref, reward_gap, beta),
            optimum,
        )
        rows.append(
            {
                "p_ref_w": p_ref,
                "reward_gap": reward_gap,
                "beta": beta,
                "delta_ref": delta_ref,
                "delta_rlhf": optimum,
                "condition_holds": optimum > 0.0,
                "population_stationary_abs_error": abs(derivative),
                "positive_loss_derivative": dpo_loss_derivative(
                    optimum, delta_ref, beta
                ),
            }
        )
    return _summarize_equivalence(rows)
