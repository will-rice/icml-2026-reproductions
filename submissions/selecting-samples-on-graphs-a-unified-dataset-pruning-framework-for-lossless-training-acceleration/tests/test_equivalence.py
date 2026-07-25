import json
from fractions import Fraction

import pytest

from graph_pruning_repro.equivalence import (
    compare_objectives,
    run_equivalence_audit,
    symbolic_coefficients,
)
from graph_pruning_repro.types import Instance


INSTANCE = Instance(
    vertices=("x", "y"),
    vertex_weights={"x": Fraction(1), "y": Fraction(2)},
    interactions={
        ("x", "y"): Fraction(-1),
        ("y", "x"): Fraction(-1),
    },
)

EXPECTED_SYMBOLIC_COEFFICIENTS = {
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


def test_two_vertex_literal_objective_mismatch() -> None:
    result = compare_objectives(INSTANCE, frozenset({"x", "y"}))

    assert result == {
        "paper_mwcp": "2/1",
        "paper_samplewise_literal": "1/1",
        "samplewise_minus_mwcp": "-1/1",
        "mwcp_edge_coefficient": 1,
        "samplewise_edge_coefficient": 2,
    }


def test_symbolic_coefficients_cover_all_named_variants_exactly() -> None:
    assert symbolic_coefficients() == EXPECTED_SYMBOLIC_COEFFICIENTS


def test_equivalence_audit_has_exact_26_case_accounting() -> None:
    audit = run_equivalence_audit("test-revision")
    search = audit["search"]

    assert audit["source_revision"] == "test-revision"
    assert search["evidence_kind"] == "exhaustive_finite"
    assert search["declared_ceiling"] == 26
    assert search["cases_examined"] == 26
    assert search["completed"] is True
    assert search["domain"] == {
        "vertex_counts": [1, 2],
        "selected_sets": "all nonempty subsets",
        "vertex_weights": ["0/1", "1/1"],
        "symmetric_edge_weights": ["0/1", "1/1"],
    }


def test_equivalence_audit_case_and_witness_order_is_deterministic() -> None:
    search = run_equivalence_audit("test-revision")["search"]
    case_ids = search["case_ids"]

    assert len(case_ids) == 26
    assert case_ids[:3] == [
        "n=1;selected=v0;vw=0/1;ew=-",
        "n=1;selected=v0;vw=1/1;ew=-",
        "n=2;selected=v0;vw=0/1,0/1;ew=0/1",
    ]
    assert case_ids[19] == (
        "n=2;selected=v0,v1;vw=0/1,0/1;ew=1/1"
    )
    assert case_ids[-1] == (
        "n=2;selected=v0,v1;vw=1/1,1/1;ew=1/1"
    )
    assert search["smallest_nonzero_edge_mismatch"] == {
        "id": "objective-equivalence-n2-edge1",
        "property": "paper_mwcp_vs_paper_samplewise_literal",
        "model_variant": "paper_samplewise_literal",
        "case_id": "n=2;selected=v0,v1;vw=0/1,0/1;ew=1/1",
        "vertices": ["v0", "v1"],
        "selected": ["v0", "v1"],
        "vertex_weights": [["v0", "0/1"], ["v1", "0/1"]],
        "interactions": [
            ["v0", "v1", "1/1"],
            ["v1", "v0", "1/1"],
        ],
        "comparison": {
            "paper_mwcp": "1/1",
            "paper_samplewise_literal": "2/1",
            "samplewise_minus_mwcp": "1/1",
            "mwcp_edge_coefficient": 1,
            "samplewise_edge_coefficient": 2,
        },
    }


def test_arbitrary_weight_conclusion_is_symbolic_not_finite() -> None:
    symbolic = run_equivalence_audit("test-revision")["symbolic"]

    assert symbolic == {
        "evidence_kind": "symbolic",
        "scope": "arbitrary symmetric vertex and edge weights",
        "coefficients": EXPECTED_SYMBOLIC_COEFFICIENTS,
        "conclusions": {
            "paper_samplewise_literal_equals_paper_mwcp": False,
            "half_corrected_samplewise_equals_paper_mwcp": True,
            "edge_coefficient_difference": "1/1",
        },
    }


def test_equivalence_audit_is_json_serializable_and_byte_deterministic() -> None:
    first = run_equivalence_audit("test-revision")
    second = run_equivalence_audit("test-revision")

    first_bytes = json.dumps(
        first,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    second_bytes = json.dumps(
        second,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    assert first == second
    assert first_bytes == second_bytes
    assert b"runtime" not in first_bytes


@pytest.mark.parametrize("source_revision", ("", 7))
def test_equivalence_audit_rejects_invalid_source_revision(
    source_revision: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        run_equivalence_audit(source_revision)  # type: ignore[arg-type]
