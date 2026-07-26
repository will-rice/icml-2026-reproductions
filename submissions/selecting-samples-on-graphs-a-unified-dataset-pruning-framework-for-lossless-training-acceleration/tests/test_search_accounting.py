from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product

import graph_pruning_repro.proof_ledger as proof_ledger_module
import pytest

from graph_pruning_repro.greedy import (
    greedy_domain_formulas,
    premise_domain_formulas,
)
from graph_pruning_repro.proof_ledger import (
    declared_aggregate_ceiling,
    declared_component_ceilings,
    run_finite_ledger_control,
)
from graph_pruning_repro.types import Instance


def _symmetric_interactions(
    edges: tuple[tuple[str, str], ...],
    values: tuple[Fraction, ...],
) -> dict[tuple[str, str], Fraction]:
    interactions: dict[tuple[str, str], Fraction] = {}
    for (left, right), value in zip(edges, values, strict=True):
        interactions[(left, right)] = value
        interactions[(right, left)] = value
    return interactions


def _greedy_domain_instances() -> tuple[tuple[Instance, int], ...]:
    instances: list[tuple[Instance, int]] = []
    for n in range(1, 5):
        vertices = tuple(f"v{index}" for index in range(n))
        edges = tuple(combinations(vertices, 2))
        for vertex_values in product(
            (Fraction(), Fraction(1), Fraction(2)),
            repeat=n,
        ):
            for edge_values in product(
                (Fraction(-1), Fraction()),
                repeat=len(edges),
            ):
                graph = Instance(
                    vertices,
                    dict(zip(vertices, vertex_values, strict=True)),
                    _symmetric_interactions(edges, edge_values),
                )
                instances.extend(
                    (graph, budget)
                    for budget in range(1, min(3, n) + 1)
                )
    return tuple(instances)


def test_named_component_ceilings_and_aggregate_are_exact() -> None:
    expected = {
        "objective_equivalence_objective_values": 52,
        "symmetric_diminishing_return_primitives": 2_861_280,
        "asymmetric_literal_diagnostic_primitives": 118_428,
        "shift_marginal_score_values": 45_213,
        "rational_alpha_values": 1_792,
        "premise_subset_values": 508_500,
        "premise_marginal_values": 1_011_330,
        "premise_submodularity_comparisons": 3_394_890,
        "eq7_candidate_scores": 316_983,
        "eq7_terminal_paths": 210_675,
        "true_marginal_candidate_lookups": 1_901_898,
        "true_marginal_terminal_paths": 1_264_050,
        "optimum_subset_objective_values": 444_870,
        "greedy_summary_classifications": 584_604,
        "finite_appendix_f_conclusions": 1_169_208,
        "symbolic_appendix_f_conclusions": 84,
        "literal_algorithm1_audit": 1,
        "appendix_e_witness_marginals": 2,
    }

    assert declared_component_ceilings() == expected
    assert sum(expected.values()) == 13_833_860
    assert declared_aggregate_ceiling() == 13_833_860


def test_task4_formulas_match_every_named_greedy_component() -> None:
    assert premise_domain_formulas() == {
        "weighted_graphs": 5_421,
        "subset_objective_values_per_variant": 84_750,
        "marginal_values_per_variant": 168_555,
        "diminishing_return_comparisons_per_variant": 565_815,
        "premise_work_per_variant": 819_120,
        "subset_objective_values": 508_500,
        "marginal_values": 1_011_330,
        "diminishing_return_comparisons": 3_394_890,
        "premise_work": 4_914_720,
    }
    assert greedy_domain_formulas() == {
        "weighted_cardinality_instances": 16_239,
        "optimum_objective_values_per_variant": 74_145,
        "terminal_paths_per_selector": 210_675,
        "candidate_operations_per_selector": 316_983,
        "eq7_candidate_scores": 316_983,
        "eq7_terminal_paths": 210_675,
        "true_marginal_cache_lookups": 1_901_898,
        "true_marginal_terminal_paths": 1_264_050,
        "optimum_objective_values": 444_870,
        "classifications": 584_604,
    }


def test_full_finite_control_reuses_all_greedy_instances_exactly() -> None:
    instances = _greedy_domain_instances()
    assert len(instances) == 16_239

    result = run_finite_ledger_control(instances)

    assert result["weighted_cardinality_instances"] == 16_239
    assert result["model_variant_instances"] == 16_239 * 6
    assert result["conclusions_per_variant_instance"] == 12
    assert result["actual_conclusion_operations"] == 1_169_208
    assert result["declared_conclusion_ceiling"] == 1_169_208
    assert result["evidence_kind"] == "exhaustive_finite"
    assert result["completed"] is True
    assert (
        0
        <= result["actual_conclusion_operations"]
        <= result["declared_conclusion_ceiling"]
    )


def test_finite_ceiling_preflights_before_consuming_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterations = 0

    def forbidden_instances() -> object:
        nonlocal iterations
        iterations += 1
        raise AssertionError("finite instances consumed before preflight")
        yield

    monkeypatch.setattr(
        proof_ledger_module,
        "FINITE_LEDGER_CONCLUSION_CEILING",
        1,
    )

    with pytest.raises(ValueError, match="finite ledger ceiling"):
        run_finite_ledger_control(forbidden_instances())
    assert iterations == 0
