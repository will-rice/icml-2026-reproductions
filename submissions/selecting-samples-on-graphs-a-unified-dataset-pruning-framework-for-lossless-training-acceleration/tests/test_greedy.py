import json
from fractions import Fraction

import pytest

import graph_pruning_repro.greedy as greedy_module
from graph_pruning_repro.diminishing_returns import (
    canonical_parameterized_instance_id,
    canonical_variant_parameters,
)
from graph_pruning_repro.greedy import (
    classify_ratio,
    compare_ratio_to_one_minus_inverse_e,
    enumerate_eq7_paths,
    enumerate_true_marginal_paths,
    exhaustive_optima,
    greedy_domain_formulas,
    premise_domain_formulas,
    run_greedy_audit,
    summarize_guarantee,
)
from graph_pruning_repro.types import SET_FUNCTION_VARIANTS, Instance


def _symmetric_instance(
    vertices: tuple[str, ...],
    vertex_weights: tuple[Fraction, ...],
    edge_weights: dict[tuple[str, str], Fraction],
    *,
    alpha: Fraction = Fraction(1),
    eta: Fraction = Fraction(),
) -> Instance:
    interactions: dict[tuple[str, str], Fraction] = {}
    for (left, right), value in edge_weights.items():
        interactions[(left, right)] = value
        interactions[(right, left)] = value
    return Instance(
        vertices,
        dict(zip(vertices, vertex_weights, strict=True)),
        interactions,
        alpha=alpha,
        eta=eta,
    )


def test_all_eq7_ties_are_retained_in_lexicographic_order() -> None:
    instance = Instance(
        ("y", "x"),
        {"y": Fraction(), "x": Fraction()},
        {},
    )

    assert enumerate_eq7_paths(instance, budget=1) == (("x",), ("y",))


def test_eq7_is_independent_of_objective_and_marginal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = Instance(
        ("x", "y"),
        {"x": Fraction(2), "y": Fraction(1)},
        {},
    )

    def forbidden(*_args: object, **_kwargs: object) -> Fraction:
        raise AssertionError("Eq. (7) called an objective or marginal")

    monkeypatch.setattr(greedy_module, "evaluate_objective", forbidden)
    monkeypatch.setattr(greedy_module, "direct_marginal", forbidden)

    assert enumerate_eq7_paths(instance, budget=1) == (("x",),)


def test_eq7_and_true_literal_marginal_are_noninterchangeable() -> None:
    instance = _symmetric_instance(
        ("x", "y", "z"),
        (Fraction(3), Fraction(2), Fraction(1, 2)),
        {("x", "y"): Fraction(-1), ("x", "z"): Fraction()},
    )

    assert enumerate_eq7_paths(instance, budget=2) == (("x", "y"),)
    assert enumerate_true_marginal_paths(
        instance,
        budget=2,
        model_variant="paper_samplewise_literal",
    ) == (("x", "z"),)


def test_true_marginal_path_uses_objective_differences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = Instance(
        ("x", "y"),
        {"x": Fraction(2), "y": Fraction(1)},
        {},
    )
    calls: list[frozenset[str]] = []
    original = greedy_module.evaluate_objective

    def recording_objective(
        current: Instance,
        selected: frozenset[str],
        model_variant: str,
    ) -> Fraction:
        calls.append(selected)
        return original(current, selected, model_variant)

    monkeypatch.setattr(
        greedy_module,
        "evaluate_objective",
        recording_objective,
    )

    assert enumerate_true_marginal_paths(
        instance,
        budget=1,
        model_variant="paper_mwcp",
    ) == (("x",),)
    assert frozenset() in calls
    assert frozenset({"x"}) in calls
    assert frozenset({"y"}) in calls


def test_appendix_selection_relation_does_not_transfer_ratios() -> None:
    graph = _symmetric_instance(
        ("a", "b", "c"),
        (Fraction(3), Fraction(2), Fraction(2)),
        {
            ("a", "b"): Fraction(-1),
            ("a", "c"): Fraction(-1),
            ("b", "c"): Fraction(),
        },
    )
    literal = Instance(
        graph.vertices,
        graph.vertex_weights,
        graph.interactions,
        alpha=Fraction(1),
        eta=Fraction(),
    )
    appendix = Instance(
        graph.vertices,
        graph.vertex_weights,
        graph.interactions,
        alpha=Fraction(1),
        eta=Fraction(1),
    )

    literal_paths = enumerate_true_marginal_paths(
        literal,
        budget=2,
        model_variant="paper_samplewise_literal",
    )
    appendix_paths = enumerate_true_marginal_paths(
        appendix,
        budget=2,
        model_variant="appendix_inline_shift_literal",
    )
    assert appendix_paths == literal_paths == (("a", "b"), ("a", "c"))

    literal_optimum = exhaustive_optima(
        literal,
        2,
        "paper_samplewise_literal",
    )
    appendix_optimum = exhaustive_optima(
        appendix,
        2,
        "appendix_inline_shift_literal",
    )
    assert literal_optimum == appendix_optimum == (frozenset({"b", "c"}),)
    assert classify_ratio(Fraction(3), Fraction(4))["ratio"] == "3/4"
    assert classify_ratio(Fraction(7), Fraction(8))["ratio"] == "7/8"
    assert Fraction(3, 4) != Fraction(7, 8)


def test_exhaustive_optima_retains_all_ties() -> None:
    instance = Instance(
        ("y", "x"),
        {"y": Fraction(), "x": Fraction()},
        {},
    )

    assert exhaustive_optima(instance, 1, "paper_mwcp") == (
        frozenset({"x"}),
        frozenset({"y"}),
    )


def test_ratio_regimes_are_exact() -> None:
    assert classify_ratio(Fraction(), Fraction()) == {
        "status": "defined_zero_equality",
        "ratio": "1/1",
    }
    assert classify_ratio(Fraction(-1), Fraction()) == {
        "status": "undefined_zero_optimum",
        "ratio": None,
    }
    assert classify_ratio(Fraction(-1), Fraction(-1)) == {
        "status": "negative_objective_regime",
        "ratio": None,
    }
    assert classify_ratio(Fraction(3), Fraction(4)) == {
        "status": "defined_positive_optimum",
        "ratio": "3/4",
    }


def test_one_minus_inverse_e_comparison_uses_rational_enclosures() -> None:
    below = compare_ratio_to_one_minus_inverse_e(Fraction(1, 2))
    above = compare_ratio_to_one_minus_inverse_e(Fraction(2, 3))

    assert below["status"] == "below_bound"
    assert above["status"] == "meets_bound"
    assert below["arithmetic"] == above["arithmetic"] == "exact_rational"
    assert all(
        "/" in bound
        for result in (below, above)
        for bound in (
            result["threshold_lower"],
            result["threshold_upper"],
        )
    )
    assert not any(
        isinstance(value, float)
        for result in (below, above)
        for value in result.values()
    )


@pytest.mark.parametrize(
    "failed_premise",
    (
        "global_nonnegativity",
        "normalization",
        "global_monotonicity",
        "global_submodularity",
    ),
)
def test_each_failed_global_premise_excludes_guarantee_violation(
    failed_premise: str,
) -> None:
    premises = {
        "global_nonnegativity": True,
        "normalization": True,
        "global_monotonicity": True,
        "global_submodularity": True,
    }
    premises[failed_premise] = False
    record = {
        "id": f"case::{failed_premise}",
        "premise_evaluation": {
            **premises,
            "theorem_eligible": True,
            "failed_witness_ids": [f"witness::{failed_premise}"],
        },
        "ratio_classification": {
            "status": "defined_positive_optimum",
            "ratio": "1/2",
        },
    }

    result = summarize_guarantee([record])

    assert result["status"] == "not_evaluated"
    assert result["eligible_records"] == 0
    assert result["guarantee_violations"] == []
    assert result["out_of_premise_diagnostics"] == [
        {
            "id": f"case::{failed_premise}",
            "failed_premise_ids": [failed_premise],
            "failed_witness_ids": [f"witness::{failed_premise}"],
            "ratio_classification": {
                "status": "defined_positive_optimum",
                "ratio": "1/2",
            },
        }
    ]


def test_eligible_poor_ratio_is_the_only_guarantee_violation() -> None:
    record = {
        "id": "eligible-poor-ratio",
        "premise_evaluation": {
            "global_nonnegativity": True,
            "normalization": True,
            "global_monotonicity": True,
            "global_submodularity": True,
            "theorem_eligible": True,
            "failed_witness_ids": [],
        },
        "ratio_classification": {
            "status": "defined_positive_optimum",
            "ratio": "1/2",
        },
    }

    result = summarize_guarantee([record])

    assert result["status"] == "contradicted"
    assert result["eligible_records"] == 1
    assert result["out_of_premise_diagnostics"] == []
    assert result["guarantee_violations"][0]["id"] == (
        "eligible-poor-ratio"
    )
    assert result["guarantee_violations"][0]["premise_evaluation"] == (
        record["premise_evaluation"]
    )


def test_domain_formulas_are_exact_before_audit() -> None:
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


@pytest.fixture(scope="module")
def greedy_audit() -> dict[str, object]:
    return run_greedy_audit("task-4-test-revision")


def test_full_audit_has_exact_domain_and_bounded_accounting(
    greedy_audit: dict[str, object],
) -> None:
    premises = greedy_audit["premise_certification"]
    search = greedy_audit["greedy_search"]

    assert premises["weighted_graphs"] == 5_421
    assert premises["premise_records"] == 5_421 * 6
    assert premises["actual"] == {
        "subset_objective_values": 508_500,
        "marginal_values": 1_011_330,
        "diminishing_return_comparisons": 3_394_890,
        "premise_work": 4_914_720,
    }
    assert premises["declared_ceiling"] == premises["actual"]

    assert search["weighted_cardinality_instances"] == 16_239
    assert search["declared_ceiling"] == {
        "eq7_candidate_scores": 316_983,
        "eq7_terminal_paths": 210_675,
        "true_marginal_cache_lookups": 1_901_898,
        "true_marginal_terminal_paths": 1_264_050,
        "optimum_objective_values": 444_870,
        "classifications": 584_604,
    }
    assert search["actual"]["optimum_objective_values"] == 444_870
    assert search["actual"]["classifications"] == 584_604
    assert all(
        0 <= search["actual"][name] <= ceiling
        for name, ceiling in search["declared_ceiling"].items()
    )


def test_audit_uses_all_canonical_task3_parameter_tuples_and_ids(
    greedy_audit: dict[str, object],
) -> None:
    graph = _symmetric_instance(
        ("v0", "v1"),
        (Fraction(), Fraction()),
        {("v0", "v1"): Fraction(-1)},
    )
    expected_parameters = {
        variant: canonical_variant_parameters(graph, variant)
        for variant in SET_FUNCTION_VARIANTS
    }
    assert greedy_audit["canonical_parameter_examples"]["n=2;M=1/1"] == {
        variant: {
            "alpha": f"{alpha.numerator}/{alpha.denominator}",
            "eta": f"{eta.numerator}/{eta.denominator}",
            "instance_id": canonical_parameterized_instance_id(
                Instance(
                    graph.vertices,
                    graph.vertex_weights,
                    graph.interactions,
                    alpha=alpha,
                    eta=eta,
                ),
                variant,
            ),
        }
        for variant, (alpha, eta) in expected_parameters.items()
    }
    for record in greedy_audit["premise_certification"][
        "representative_records"
    ]:
        assert record["id"].startswith(record["base_instance_id"])


def test_score_only_premise_has_no_finite_work(
    greedy_audit: dict[str, object],
) -> None:
    assert greedy_audit["score_only_premise"] == {
        "model_variant": "appendix_eq26_score",
        "status": "not_applicable",
        "reason": "score_is_not_a_set_function",
        "finite_work_units": 0,
    }


def test_appendix_tie_relation_never_transfers_ratio(
    greedy_audit: dict[str, object],
) -> None:
    relation = greedy_audit["appendix_selection_relation"]

    assert relation["selection_ties_match_at_fixed_iteration"] is True
    assert relation["objective_ratios_recomputed"] is True
    assert relation["ratio_transfer"] is False
    assert "ratio_transfer" not in greedy_audit["guarantee_summary"]


def test_guarantee_arrays_are_independent_and_premise_gated(
    greedy_audit: dict[str, object],
) -> None:
    summary = greedy_audit["guarantee_summary"]

    assert summary["guarantee_violations"] is not (
        summary["out_of_premise_diagnostics"]
    )
    assert all(
        all(
            record["premise_evaluation"][premise]
            for premise in (
                "global_nonnegativity",
                "normalization",
                "global_monotonicity",
                "global_submodularity",
            )
        )
        for record in summary["guarantee_violations"]
    )
    assert all(
        record["failed_premise_ids"]
        for record in summary["out_of_premise_diagnostics"]
    )


def test_audit_is_byte_deterministic_and_has_no_runtime(
    greedy_audit: dict[str, object],
) -> None:
    first = json.dumps(
        greedy_audit,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    second = json.dumps(
        run_greedy_audit("task-4-test-revision"),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")

    assert first == second
    assert b"runtime" not in first


@pytest.mark.parametrize(
    ("ceiling_name", "too_small"),
    (
        ("PREMISE_SUBSET_CEILING", 508_499),
        ("PREMISE_MARGINAL_CEILING", 1_011_329),
        ("PREMISE_DIMINISHING_CEILING", 3_394_889),
        ("EQ7_SCORE_CEILING", 316_982),
        ("EQ7_PATH_CEILING", 210_674),
        ("TRUE_MARGINAL_LOOKUP_CEILING", 1_901_897),
        ("TRUE_MARGINAL_PATH_CEILING", 1_264_049),
        ("OPTIMUM_VALUE_CEILING", 444_869),
        ("CLASSIFICATION_CEILING", 584_603),
    ),
)
def test_all_formula_ceilings_preflight_before_iteration(
    monkeypatch: pytest.MonkeyPatch,
    ceiling_name: str,
    too_small: int,
) -> None:
    iterations = 0

    def forbidden_product(*_args: object, **_kwargs: object) -> object:
        nonlocal iterations
        iterations += 1
        raise AssertionError("greedy audit started traversal")

    monkeypatch.setattr(greedy_module, ceiling_name, too_small)
    monkeypatch.setattr(greedy_module, "product", forbidden_product)

    with pytest.raises(ValueError, match="ceiling"):
        run_greedy_audit("task-4-preflight-regression")
    assert iterations == 0


@pytest.mark.parametrize("source_revision", ("", 7))
def test_audit_rejects_invalid_source_revision(
    source_revision: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        run_greedy_audit(source_revision)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("function_name", "budget"),
    (
        ("eq7", 0),
        ("eq7", 3),
        ("true", 0),
        ("true", 3),
        ("optimum", 0),
        ("optimum", 3),
    ),
)
def test_greedy_helpers_reject_invalid_budgets(
    function_name: str,
    budget: int,
) -> None:
    instance = Instance(
        ("x", "y"),
        {"x": Fraction(), "y": Fraction()},
        {},
    )

    with pytest.raises(ValueError, match="budget"):
        if function_name == "eq7":
            enumerate_eq7_paths(instance, budget)
        elif function_name == "true":
            enumerate_true_marginal_paths(
                instance,
                budget,
                "paper_mwcp",
            )
        else:
            exhaustive_optima(instance, budget, "paper_mwcp")
