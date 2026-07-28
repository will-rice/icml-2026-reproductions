import itertools

from conditional_dpo_repro.grids import (
    SOFT_MARGIN_BETAS,
    SOFT_MARGIN_DELTAS,
    SOFT_MARGIN_DELTAS_REF,
)
from conditional_dpo_repro.math import scaled_dpo_soft_margin


def derive_soft_margin_outcome(
    max_error_highest_beta: float,
    has_negative_targets: bool,
) -> str:
    if max_error_highest_beta <= 1e-2 and has_negative_targets:
        return "consistent"
    return "contradiction"


def run_soft_margin_lane() -> dict[str, object]:
    rows = []
    for delta_ref, delta, beta in itertools.product(
        SOFT_MARGIN_DELTAS_REF, SOFT_MARGIN_DELTAS, SOFT_MARGIN_BETAS
    ):
        scaled = scaled_dpo_soft_margin(delta, delta_ref, beta)
        hinge = max(0.0, delta_ref - delta)
        rows.append(
            {
                "delta_ref": delta_ref,
                "delta": delta,
                "beta": beta,
                "scaled_dpo_loss": scaled,
                "hinge": hinge,
                "abs_error": abs(scaled - hinge),
            }
        )
    errors = {
        format(beta, "g"): max(r["abs_error"] for r in rows if r["beta"] == beta)
        for beta in SOFT_MARGIN_BETAS
    }
    examples = tuple(
        {
            "delta_ref": r["delta_ref"],
            "delta": r["delta"],
            "target_margin": r["delta_ref"],
        }
        for r in rows
        if r["beta"] == 256.0 and r["delta_ref"] < r["delta"] < 0.0
    )
    outcome = derive_soft_margin_outcome(
        errors.get(format(max(SOFT_MARGIN_BETAS), "g"), 1.0),
        len(examples) > 0,
    )
    return {
        "case_count": len(rows),
        "max_abs_error_by_beta": errors,
        "negative_target_examples": list(examples),
        "finite_beta_loss_is_literal_hinge": False,
        "cases": rows,
        "outcome": outcome,
    }
