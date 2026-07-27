"""Independent exact evaluators for the seven named objective/score variants."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from .types import MODEL_VARIANTS, Instance, ModelVariant, Vertex, ordered_selection


def evaluate_mwcp_edges(
    instance: Instance,
    selected: frozenset[Vertex],
) -> Fraction:
    """Evaluate Eq. (3) by one independent unordered-edge traversal."""

    ordered = ordered_selection(instance, selected)
    total = sum(
        (instance.vertex_weights[vertex] for vertex in ordered),
        start=Fraction(),
    )
    for left, right in combinations(ordered, 2):
        total += instance.interactions.get((left, right), Fraction())
    return total


def evaluate_samplewise_literal(
    instance: Instance,
    selected: frozenset[Vertex],
) -> Fraction:
    """Evaluate literal Eqs. (4)--(5) by an independent sample traversal."""

    ordered = ordered_selection(instance, selected)
    total = Fraction()
    for left in ordered:
        total += instance.vertex_weights[left]
        for right in ordered:
            if right != left:
                total += instance.interactions.get((left, right), Fraction())
    return total


def evaluate_objective(
    instance: Instance,
    selected: frozenset[Vertex],
    model_variant: ModelVariant,
) -> Fraction:
    """Dispatch one exact set-function variant and reject the score-only one."""

    ordered = ordered_selection(instance, selected)
    if type(model_variant) is not str or model_variant not in MODEL_VARIANTS:
        raise ValueError("unknown model variant")
    if model_variant == "appendix_eq26_score":
        raise ValueError("appendix_eq26_score is a score, not a set function")
    if model_variant in {"paper_mwcp", "single_counted_pairwise"}:
        return evaluate_mwcp_edges(instance, selected)
    if model_variant == "paper_samplewise_literal":
        return evaluate_samplewise_literal(instance, selected)
    if model_variant == "half_corrected_samplewise":
        vertex_total = sum(
            (instance.vertex_weights[vertex] for vertex in ordered),
            start=Fraction(),
        )
        literal_total = evaluate_samplewise_literal(instance, selected)
        return vertex_total + (literal_total - vertex_total) / 2
    if model_variant == "appendix_inline_shift_literal":
        literal_total = evaluate_samplewise_literal(instance, selected)
        return (
            literal_total
            + instance.alpha * instance.eta * len(ordered) ** 2
        )
    if model_variant == "modular_shift_candidate":
        literal_total = evaluate_samplewise_literal(instance, selected)
        return literal_total + instance.eta * len(ordered)
    raise AssertionError("validated model variant was not dispatched")
