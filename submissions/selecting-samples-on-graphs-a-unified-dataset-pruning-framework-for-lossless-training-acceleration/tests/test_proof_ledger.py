from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import pytest

import graph_pruning_repro.proof_ledger as proof_ledger_module
from graph_pruning_repro.diminishing_returns import (
    appendix_shift_witness,
    canonical_parameterized_instance_id,
    canonical_variant_parameters,
)
from graph_pruning_repro.proof_ledger import (
    build_symbolic_ledger,
    equation_36,
    run_finite_ledger_control,
    run_symbolic_ledger_control,
    validate_prerequisite_graph,
)
from graph_pruning_repro.types import MODEL_VARIANTS, Instance


ROW_KEYS = {
    "row_id",
    "equation",
    "model_variant",
    "instance_id",
    "conclusions",
}
CONCLUSION_KEYS = {
    "conclusion_id",
    "statement",
    "required_premise_ids",
    "prerequisite_conclusion_refs",
    "check_id",
    "evidence_kind",
    "status",
    "blocked_by",
    "witness_ids",
}


def _conclusion(
    rows: tuple[dict[str, object], ...],
    equation: str,
    conclusion_id: str | None = None,
) -> dict[str, object]:
    row = next(row for row in rows if row["equation"] == equation)
    conclusions = row["conclusions"]
    assert isinstance(conclusions, list)
    if conclusion_id is None:
        assert len(conclusions) == 1
        return conclusions[0]
    return next(
        conclusion
        for conclusion in conclusions
        if conclusion["conclusion_id"] == conclusion_id
    )


def _conclusion_by_check(
    rows: tuple[dict[str, object], ...],
    check_id: str,
) -> dict[str, object]:
    return next(
        conclusion
        for row in rows
        for conclusion in row["conclusions"]
        if conclusion["check_id"] == check_id
    )


def test_symbolic_ledger_has_explicit_uniform_rows_and_conclusions() -> None:
    rows = build_symbolic_ledger("paper_samplewise_literal", {})

    assert [row["equation"] for row in rows] == [
        str(equation) for equation in range(28, 39)
    ]
    assert len(rows) == 11
    assert all(set(row) == ROW_KEYS for row in rows)
    assert sum(len(row["conclusions"]) for row in rows) == 12
    assert all(
        set(conclusion) == CONCLUSION_KEYS
        for row in rows
        for conclusion in row["conclusions"]
    )
    eq28 = next(row for row in rows if row["equation"] == "28")
    assert {
        conclusion["conclusion_id"]
        for conclusion in eq28["conclusions"]
    } == {
        "eq28_union_submodular_bound",
        "eq28_b_minus_t_bound",
    }
    validate_prerequisite_graph(rows)


def test_appendix_literal_ledger_links_shift_witness_everywhere() -> None:
    appendix_witness_id = appendix_shift_witness()["id"]
    rows = build_symbolic_ledger(
        "appendix_inline_shift_literal",
        {"appendix_shift": appendix_witness_id},
    )
    submodularity_conclusions = [
        conclusion
        for row in rows
        for conclusion in row["conclusions"]
        if "global_submodularity"
        in conclusion["required_premise_ids"]
    ]

    assert submodularity_conclusions
    assert all(
        conclusion["status"] in {"contradicted", "not_applicable"}
        for conclusion in submodularity_conclusions
    )
    assert all(
        appendix_witness_id in conclusion["witness_ids"]
        for conclusion in submodularity_conclusions
    )
    assert (
        _conclusion_by_check(
            rows,
            "ratio_transfer_from_repaired_objective",
        )["status"]
        == "contradicted"
    )


def test_cardinality_failure_blocks_downstream_without_relabeling_it() -> None:
    cardinality_witness = getattr(
        proof_ledger_module,
        "cardinality_b_minus_t_witness",
        lambda: None,
    )()
    assert cardinality_witness == {
        "id": "cardinality-b-minus-t",
        "property": "optimum_remainder_cardinality_exceeds_b_minus_t",
        "evidence_kind": "symbolic",
        "inputs": {
            "vertices": ["a", "b", "c"],
            "budget": 2,
            "iteration": 1,
            "s_t": ["a"],
            "s_star": ["b", "c"],
            "weight_assumptions": "none",
        },
        "intermediate_values": {
            "s_star_minus_s_t": ["b", "c"],
            "remainder_cardinality": 2,
            "b_minus_t": 1,
        },
        "classification": {
            "comparison": "2 > 1",
            "eq28b_cardinality_bound": "contradicted",
            "eq28a_counterexample": False,
            "theorem_counterexample": False,
        },
    }
    rows = build_symbolic_ledger("paper_samplewise_literal", {})
    cardinality = _conclusion_by_check(
        rows,
        "optimum_remainder_at_most_b_not_b_minus_t",
    )

    assert cardinality["status"] == "contradicted"
    assert cardinality["witness_ids"] == [cardinality_witness["id"]]
    eq30 = _conclusion(rows, "30")
    assert eq30["status"] == "not_applicable"
    assert (
        "paper_samplewise_literal/symbolic/eq28/"
        "eq28_b_minus_t_bound"
        in eq30["blocked_by"]
    )
    product = _conclusion_by_check(
        rows,
        "product_includes_k_equals_one_zero_factor",
    )
    assert product["status"] == "not_applicable"
    assert any("/eq33/" in blocker for blocker in product["blocked_by"])


def test_equation_36_separates_conclusion_from_log_domain() -> None:
    assert equation_36(0) == {
        "status": "not_applicable",
        "conclusion_status": "not_applicable",
        "log_derivation_status": "not_applicable",
    }
    assert equation_36(1) == {
        "status": "supported",
        "conclusion_status": "supported",
        "log_derivation_status": "not_applicable",
    }
    assert equation_36(2) == {
        "status": "supported",
        "conclusion_status": "supported",
        "log_derivation_status": "supported",
    }


@pytest.mark.parametrize(
    "mutation",
    ("missing", "duplicate", "replaced", "unknown", "cross-instance", "cycle"),
)
def test_prerequisite_graph_rejects_every_explicit_edge_mutation(
    mutation: str,
) -> None:
    rows = list(deepcopy(build_symbolic_ledger(
        "paper_samplewise_literal",
        {},
    )))
    eq28a = _conclusion(
        tuple(rows),
        "28",
        "eq28_union_submodular_bound",
    )
    eq30 = _conclusion(tuple(rows), "30")
    references = eq30["prerequisite_conclusion_refs"]
    assert isinstance(references, list)

    if mutation == "missing":
        references.pop()
    elif mutation == "duplicate":
        references.append(references[0])
    elif mutation == "replaced":
        references[0] = (
            "paper_samplewise_literal/symbolic/eq31/"
            "eq31_residual_identity"
        )
    elif mutation == "unknown":
        references[0] = (
            "paper_samplewise_literal/symbolic/eq99/"
            "unknown_conclusion"
        )
    elif mutation == "cross-instance":
        references[0] = references[0].replace(
            "/symbolic/",
            "/other-instance/",
        )
    else:
        cyclic = eq28a["prerequisite_conclusion_refs"]
        assert isinstance(cyclic, list)
        cyclic.append(
            "paper_samplewise_literal/symbolic/eq38/"
            "eq38_greedy_ratio"
        )

    with pytest.raises(ValueError):
        validate_prerequisite_graph(tuple(rows))


def test_score_only_variant_has_twelve_explicit_not_applicable_rows() -> None:
    rows = build_symbolic_ledger("appendix_eq26_score", {})

    assert len(rows) == 11
    assert sum(len(row["conclusions"]) for row in rows) == 12
    assert {
        conclusion["status"]
        for row in rows
        for conclusion in row["conclusions"]
    } == {"not_applicable"}

    symbolic = run_symbolic_ledger_control(
        {"appendix_shift": appendix_shift_witness()["id"]}
    )
    assert symbolic["model_variants"] == list(MODEL_VARIANTS)
    assert symbolic["actual_conclusion_operations"] == 84
    assert symbolic["declared_conclusion_ceiling"] == 84
    assert symbolic["completed"] is True


def test_finite_control_uses_task2_and_task3_canonical_oracles() -> None:
    graph = Instance(
        vertices=("v0", "v1"),
        vertex_weights={"v0": Fraction(), "v1": Fraction(1)},
        interactions={
            ("v0", "v1"): Fraction(-1),
            ("v1", "v0"): Fraction(-1),
        },
    )

    result = run_finite_ledger_control(((graph, 1),))

    assert result["weighted_cardinality_instances"] == 1
    assert result["actual_conclusion_operations"] == 72
    assert result["declared_conclusion_ceiling"] == 1_169_208
    assert result["evidence_kind"] == "non_exhaustive"
    assert result["completed"] is False
    examples = result["canonical_parameter_examples"]
    assert set(examples) == {
        "paper_mwcp",
        "paper_samplewise_literal",
        "single_counted_pairwise",
        "half_corrected_samplewise",
        "appendix_inline_shift_literal",
        "modular_shift_candidate",
    }
    for variant, example in examples.items():
        alpha, eta = canonical_variant_parameters(graph, variant)
        parameterized = Instance(
            graph.vertices,
            graph.vertex_weights,
            graph.interactions,
            alpha=alpha,
            eta=eta,
        )
        assert example["alpha"] == f"{alpha.numerator}/{alpha.denominator}"
        assert example["eta"] == f"{eta.numerator}/{eta.denominator}"
        assert example["base_instance_id"] == (
            canonical_parameterized_instance_id(parameterized, variant)
        )
        assert example["direct_marginal"] == example["closed_form_marginal"]

    modular = examples["modular_shift_candidate"]
    assert modular["eta"] == "2/1"
    assert modular["objective_selected"] == "0/1"
    assert modular["literal_base_objective_selected"] == "0/1"
    assert modular["direct_marginal"] == "2/1"
    assert modular["modular_formula_marginal"] == "2/1"
