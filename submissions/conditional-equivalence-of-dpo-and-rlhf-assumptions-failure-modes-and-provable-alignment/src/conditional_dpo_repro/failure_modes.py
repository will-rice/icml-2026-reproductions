import itertools

from conditional_dpo_repro.grids import (
    RELATIVE_BETAS,
    RELATIVE_DELTAS_REF,
    RELATIVE_OFFSETS,
)
from conditional_dpo_repro.math import dpo_loss, policy_from_delta


def _relative_rows():
    for delta_ref, offset, beta in itertools.product(
        RELATIVE_DELTAS_REF, RELATIVE_OFFSETS, RELATIVE_BETAS
    ):
        delta = delta_ref + offset
        yield {
            "delta_ref": delta_ref,
            "offset": offset,
            "delta": delta,
            "beta": beta,
            "relative_improvement": delta > delta_ref,
            "absolute_preference": delta > 0.0,
            "reference_loss": dpo_loss(delta_ref, delta_ref, beta),
            "candidate_loss": dpo_loss(delta, delta_ref, beta),
            "preferred_probability": policy_from_delta(delta).preferred,
        }


def run_relative_advantage_lane() -> dict[str, object]:
    rows = tuple(_relative_rows())
    return {
        "case_count": len(rows),
        "relative_improvement_count": sum(r["relative_improvement"] for r in rows),
        "absolute_preference_count": sum(r["absolute_preference"] for r in rows),
        "relative_but_not_absolute_count": sum(
            r["relative_improvement"] and not r["absolute_preference"] for r in rows
        ),
        "cases": list(rows),
        "outcome": "consistent",
    }


def run_undesirable_space_lane() -> dict[str, object]:
    witnesses = tuple(
        row for row in _relative_rows() if row["delta_ref"] < row["delta"] < 0.0
    )
    return {
        "witness_count": len(witnesses),
        "witnesses": list(witnesses),
        "outcome": "consistent" if witnesses else "contradiction",
    }
