"""Symbolic and finite exact audits of Eq. (3) versus literal Eqs. (4)--(5)."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product

from .objectives import evaluate_mwcp_edges, evaluate_samplewise_literal
from .types import Instance, Vertex, Witness


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def symbolic_coefficients() -> dict[str, dict[str, object]]:
    """Return the direct symbolic coefficients for every named variant."""

    return {
        "paper_mwcp": {
            "kind": "set_function",
            "vertex_weight_coefficient": 1,
            "symmetric_edge_coefficient": 1,
            "cardinality_term": "0",
        },
        "paper_samplewise_literal": {
            "kind": "set_function",
            "vertex_weight_coefficient": 1,
            "symmetric_edge_coefficient": 2,
            "cardinality_term": "0",
        },
        "single_counted_pairwise": {
            "kind": "set_function",
            "vertex_weight_coefficient": 1,
            "symmetric_edge_coefficient": 1,
            "cardinality_term": "0",
        },
        "half_corrected_samplewise": {
            "kind": "set_function",
            "vertex_weight_coefficient": 1,
            "symmetric_edge_coefficient": 1,
            "cardinality_term": "0",
        },
        "appendix_inline_shift_literal": {
            "kind": "set_function",
            "vertex_weight_coefficient": 1,
            "symmetric_edge_coefficient": 2,
            "cardinality_term": "alpha*eta*|S|^2",
        },
        "appendix_eq26_score": {
            "kind": "score_only",
            "vertex_weight_coefficient": None,
            "symmetric_edge_coefficient": None,
            "cardinality_term": None,
        },
        "modular_shift_candidate": {
            "kind": "set_function",
            "vertex_weight_coefficient": 1,
            "symmetric_edge_coefficient": 2,
            "cardinality_term": "eta*|S|",
        },
    }


def compare_objectives(
    instance: Instance,
    selected: frozenset[Vertex],
) -> dict[str, object]:
    """Compare independently traversed MWCP and literal samplewise totals."""

    mwcp_total = evaluate_mwcp_edges(instance, selected)
    samplewise_total = evaluate_samplewise_literal(instance, selected)
    return {
        "paper_mwcp": _fraction_text(mwcp_total),
        "paper_samplewise_literal": _fraction_text(samplewise_total),
        "samplewise_minus_mwcp": _fraction_text(
            samplewise_total - mwcp_total
        ),
        "mwcp_edge_coefficient": 1,
        "samplewise_edge_coefficient": 2,
    }


def _nonempty_selected_sets(
    vertices: tuple[Vertex, ...],
) -> tuple[frozenset[Vertex], ...]:
    return tuple(
        frozenset(selected)
        for size in range(1, len(vertices) + 1)
        for selected in combinations(vertices, size)
    )


def _symmetric_interactions(
    edges: tuple[tuple[Vertex, Vertex], ...],
    edge_weights: tuple[Fraction, ...],
) -> dict[tuple[Vertex, Vertex], Fraction]:
    interactions: dict[tuple[Vertex, Vertex], Fraction] = {}
    for (left, right), weight in zip(edges, edge_weights, strict=True):
        interactions[(left, right)] = weight
        interactions[(right, left)] = weight
    return interactions


def _case_id(
    vertices: tuple[Vertex, ...],
    selected: frozenset[Vertex],
    vertex_weights: tuple[Fraction, ...],
    edge_weights: tuple[Fraction, ...],
) -> str:
    selected_text = ",".join(sorted(selected))
    vertex_text = ",".join(_fraction_text(value) for value in vertex_weights)
    edge_text = (
        ",".join(_fraction_text(value) for value in edge_weights)
        if edge_weights
        else "-"
    )
    return (
        f"n={len(vertices)};selected={selected_text};"
        f"vw={vertex_text};ew={edge_text}"
    )


def _mismatch_witness(
    instance: Instance,
    selected: frozenset[Vertex],
    case_id: str,
    edges: tuple[tuple[Vertex, Vertex], ...],
    edge_weights: tuple[Fraction, ...],
    comparison: dict[str, object],
) -> dict[str, object]:
    interactions: list[list[str]] = []
    for (left, right), weight in zip(edges, edge_weights, strict=True):
        text = _fraction_text(weight)
        interactions.append([left, right, text])
        interactions.append([right, left, text])
    return {
        "id": "objective-equivalence-n2-edge1",
        "property": "paper_mwcp_vs_paper_samplewise_literal",
        "model_variant": "paper_samplewise_literal",
        "case_id": case_id,
        "vertices": list(instance.vertices),
        "selected": sorted(selected),
        "vertex_weights": [
            [vertex, _fraction_text(instance.vertex_weights[vertex])]
            for vertex in instance.vertices
        ],
        "interactions": interactions,
        "comparison": comparison,
    }


def run_equivalence_audit(source_revision: str) -> dict[str, object]:
    """Run the symbolic result and exact deterministic 26-case control."""

    if type(source_revision) is not str:
        raise TypeError("source_revision must be a string")
    if not source_revision.strip():
        raise ValueError("source_revision must be nonempty")

    cases_examined = 0
    case_ids: list[str] = []
    smallest_mismatch: Witness | None = None
    domain_values = (Fraction(0), Fraction(1))

    for vertex_count in (1, 2):
        vertices = tuple(f"v{index}" for index in range(vertex_count))
        edges = tuple(combinations(vertices, 2))
        for selected in _nonempty_selected_sets(vertices):
            for vertex_values in product(domain_values, repeat=vertex_count):
                vertex_weights = dict(zip(vertices, vertex_values, strict=True))
                for edge_values in product(domain_values, repeat=len(edges)):
                    interactions = _symmetric_interactions(edges, edge_values)
                    instance = Instance(
                        vertices=vertices,
                        vertex_weights=vertex_weights,
                        interactions=interactions,
                    )
                    current_case_id = _case_id(
                        vertices,
                        selected,
                        vertex_values,
                        edge_values,
                    )
                    comparison = compare_objectives(instance, selected)
                    cases_examined += 1
                    case_ids.append(current_case_id)
                    if (
                        smallest_mismatch is None
                        and any(weight != 0 for weight in edge_values)
                        and comparison["paper_mwcp"]
                        != comparison["paper_samplewise_literal"]
                    ):
                        smallest_mismatch = _mismatch_witness(
                            instance,
                            selected,
                            current_case_id,
                            edges,
                            edge_values,
                            comparison,
                        )

    if cases_examined != 26:
        raise AssertionError("equivalence search accounting is not 26")
    if smallest_mismatch is None:
        raise AssertionError("equivalence search found no nonzero-edge mismatch")

    coefficients = symbolic_coefficients()
    return {
        "source_revision": source_revision,
        "symbolic": {
            "evidence_kind": "symbolic",
            "scope": "arbitrary symmetric vertex and edge weights",
            "coefficients": coefficients,
            "conclusions": {
                "paper_samplewise_literal_equals_paper_mwcp": False,
                "half_corrected_samplewise_equals_paper_mwcp": True,
                "edge_coefficient_difference": "1/1",
            },
        },
        "search": {
            "evidence_kind": "exhaustive_finite",
            "domain": {
                "vertex_counts": [1, 2],
                "selected_sets": "all nonempty subsets",
                "vertex_weights": ["0/1", "1/1"],
                "symmetric_edge_weights": ["0/1", "1/1"],
            },
            "declared_ceiling": 26,
            "cases_examined": cases_examined,
            "completed": True,
            "case_ids": case_ids,
            "smallest_nonzero_edge_mismatch": smallest_mismatch,
        },
    }
