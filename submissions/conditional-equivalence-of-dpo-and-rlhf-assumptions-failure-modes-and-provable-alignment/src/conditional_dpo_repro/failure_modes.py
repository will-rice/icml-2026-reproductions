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


def derive_relative_advantage_outcome(
    case_count: int,
    relative_improvement_count: int,
    relative_but_not_absolute_count: int,
) -> str:
    if relative_improvement_count == case_count and relative_but_not_absolute_count > 0:
        return "consistent"
    if relative_improvement_count == 0:
        return "contradiction"
    return "mixed"


def derive_undesirable_space_outcome(witness_count: int) -> str:
    return "consistent" if witness_count > 0 else "contradiction"


def run_relative_advantage_lane() -> dict[str, object]:
    rows = tuple(_relative_rows())
    rel_improvement = sum(r["relative_improvement"] for r in rows)
    rel_not_abs = sum(
        r["relative_improvement"] and not r["absolute_preference"] for r in rows
    )
    outcome = derive_relative_advantage_outcome(len(rows), rel_improvement, rel_not_abs)
    return {
        "case_count": len(rows),
        "relative_improvement_count": rel_improvement,
        "absolute_preference_count": sum(r["absolute_preference"] for r in rows),
        "relative_but_not_absolute_count": rel_not_abs,
        "cases": list(rows),
        "outcome": outcome,
    }


def run_undesirable_space_lane() -> dict[str, object]:
    witnesses = tuple(
        row for row in _relative_rows() if row["delta_ref"] < row["delta"] < 0.0
    )
    outcome = derive_undesirable_space_outcome(len(witnesses))
    return {
        "witness_count": len(witnesses),
        "witnesses": list(witnesses),
        "outcome": outcome,
    }
