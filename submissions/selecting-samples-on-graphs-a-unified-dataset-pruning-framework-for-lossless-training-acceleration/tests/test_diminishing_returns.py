import json
from fractions import Fraction

import pytest

import graph_pruning_repro.diminishing_returns as diminishing_returns
from graph_pruning_repro.diminishing_returns import (
    SET_FUNCTION_VARIANTS,
    appendix_shift_witness,
    canonical_parameterized_instance_id,
    canonical_variant_parameters,
    closed_form_marginal,
    direct_marginal,
    enumerate_diminishing_returns,
    run_diminishing_returns_audit,
)
from graph_pruning_repro.types import Instance


@pytest.fixture(scope="module")
def diminishing_audit() -> dict[str, object]:
    return run_diminishing_returns_audit("task-3-test-revision")


def test_appendix_inline_shift_has_minimal_one_then_three_witness() -> None:
    instance = Instance(
        vertices=("x", "y"),
        vertex_weights={"x": Fraction(), "y": Fraction()},
        interactions={
            ("x", "y"): Fraction(),
            ("y", "x"): Fraction(),
        },
        alpha=Fraction(1),
        eta=Fraction(1),
    )

    assert (
        direct_marginal(
            instance,
            frozenset(),
            "x",
            "appendix_inline_shift_literal",
        )
        == 1
    )
    assert (
        direct_marginal(
            instance,
            frozenset({"y"}),
            "x",
            "appendix_inline_shift_literal",
        )
        == 3
    )
    witness = appendix_shift_witness()
    assert witness["id"].endswith(
        "alpha=1/1::eta=1/1::diagnostic=appendix-minimal"
    )
    assert witness["property"] == (
        "appendix_inline_shift_diminishing_returns"
    )
    assert witness["model_variant"] == "appendix_inline_shift_literal"
    assert witness["intermediate_values"] == {
        "marginal_empty": "1/1",
        "marginal_y": "3/1",
        "difference": "-2/1",
    }
    assert witness["minimality_checks"] == {
        "one_vertex_strict_chain_exists": False,
        "two_vertices_required": True,
    }


def test_direct_and_closed_forms_agree_on_all_six_exact_formulas() -> None:
    instance = Instance(
        ("x", "y"),
        {"x": Fraction(2), "y": Fraction()},
        {("x", "y"): Fraction(-1), ("y", "x"): Fraction(-2)},
        alpha=Fraction(1),
        eta=Fraction(3),
    )
    expected = {
        "paper_mwcp": Fraction(1),
        "paper_samplewise_literal": Fraction(-1),
        "single_counted_pairwise": Fraction(1),
        "half_corrected_samplewise": Fraction(1, 2),
        "appendix_inline_shift_literal": Fraction(8),
        "modular_shift_candidate": Fraction(2),
    }

    assert set(expected) == set(SET_FUNCTION_VARIANTS)
    for variant, value in expected.items():
        assert (
            direct_marginal(
                instance,
                frozenset({"y"}),
                "x",
                variant,
            )
            == value
        )
        assert (
            closed_form_marginal(
                instance,
                frozenset({"y"}),
                "x",
                variant,
            )
            == value
        )


def test_closed_form_does_not_call_objective_or_direct_marginal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = Instance(
        ("x", "y"),
        {"x": Fraction(2), "y": Fraction()},
        {("x", "y"): Fraction(-1), ("y", "x"): Fraction(-2)},
    )

    def forbidden(*_args: object, **_kwargs: object) -> Fraction:
        raise AssertionError("closed form reused another marginal implementation")

    monkeypatch.setattr(diminishing_returns, "evaluate_objective", forbidden)
    monkeypatch.setattr(diminishing_returns, "direct_marginal", forbidden)

    assert (
        closed_form_marginal(
            instance,
            frozenset({"y"}),
            "x",
            "paper_samplewise_literal",
        )
        == Fraction(-1)
    )


def test_direct_marginal_does_not_call_closed_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = Instance(
        ("x", "y"),
        {"x": Fraction(2), "y": Fraction()},
        {("x", "y"): Fraction(-1), ("y", "x"): Fraction(-2)},
    )

    def forbidden(*_args: object, **_kwargs: object) -> Fraction:
        raise AssertionError("direct marginal reused the closed form")

    monkeypatch.setattr(
        diminishing_returns,
        "closed_form_marginal",
        forbidden,
    )

    assert (
        direct_marginal(
            instance,
            frozenset({"y"}),
            "x",
            "paper_samplewise_literal",
        )
        == Fraction(-1)
    )


def test_canonical_variant_parameters_cover_all_six_without_outcomes() -> None:
    graph = Instance(
        vertices=("x", "y"),
        vertex_weights={"x": Fraction(), "y": Fraction()},
        interactions={
            ("x", "y"): Fraction(-1),
            ("y", "x"): Fraction(-1),
        },
    )

    assert {
        variant: canonical_variant_parameters(graph, variant)
        for variant in SET_FUNCTION_VARIANTS
    } == {
        "paper_mwcp": (Fraction(1), Fraction()),
        "paper_samplewise_literal": (Fraction(1), Fraction()),
        "single_counted_pairwise": (Fraction(1), Fraction()),
        "half_corrected_samplewise": (Fraction(1), Fraction()),
        "appendix_inline_shift_literal": (Fraction(1), Fraction(1)),
        "modular_shift_candidate": (Fraction(1), Fraction(2)),
    }
    changed_vertices = Instance(
        vertices=graph.vertices,
        vertex_weights={"x": Fraction(99), "y": Fraction(-99)},
        interactions=graph.interactions,
    )
    assert canonical_variant_parameters(
        changed_vertices,
        "modular_shift_candidate",
    ) == (Fraction(1), Fraction(2))
    edgeless = Instance(("x",), {"x": Fraction()}, {})
    assert canonical_variant_parameters(
        edgeless,
        "appendix_inline_shift_literal",
    ) == (Fraction(1), Fraction())
    assert canonical_variant_parameters(
        edgeless,
        "modular_shift_candidate",
    ) == (Fraction(1), Fraction())


def test_canonical_parameterized_ids_are_exact_ordered_and_collision_safe() -> None:
    graph = Instance(
        vertices=("y", "x"),
        vertex_weights={"y": Fraction(2), "x": Fraction(1, 2)},
        interactions={
            ("y", "x"): Fraction(-3),
            ("x", "y"): Fraction(-3),
        },
    )

    ids = {
        variant: canonical_parameterized_instance_id(graph, variant)
        for variant in SET_FUNCTION_VARIANTS
    }
    assert ids["appendix_inline_shift_literal"] == (
        "graph=n=2;vw=2/1,1/2;ew=-3/1"
        "::variant=appendix_inline_shift_literal::alpha=1/1::eta=3/1"
    )
    assert ids["modular_shift_candidate"].endswith(
        "::variant=modular_shift_candidate::alpha=1/1::eta=6/1"
    )
    assert len(set(ids.values())) == len(SET_FUNCTION_VARIANTS)

    renamed = Instance(
        vertices=("a", "b"),
        vertex_weights={"a": Fraction(2), "b": Fraction(1, 2)},
        interactions={
            ("a", "b"): Fraction(-3),
            ("b", "a"): Fraction(-3),
        },
    )
    assert canonical_parameterized_instance_id(
        renamed,
        "appendix_inline_shift_literal",
    ) == ids["appendix_inline_shift_literal"]


def test_canonical_id_rejects_asymmetric_interaction_collision() -> None:
    asymmetric = Instance(
        vertices=("x", "y"),
        vertex_weights={"x": Fraction(), "y": Fraction()},
        interactions={
            ("x", "y"): Fraction(-1),
            ("y", "x"): Fraction(-2),
        },
    )

    with pytest.raises(ValueError, match="symmetric interactions"):
        canonical_parameterized_instance_id(
            asymmetric,
            "paper_samplewise_literal",
        )


def test_modular_coefficient_is_fixed_from_graph_on_literal_base() -> None:
    vertices = ("x", "y", "z")
    eta_mod = 2 * (len(vertices) - 1) * Fraction(1)
    instance = Instance(
        vertices=vertices,
        vertex_weights={vertex: Fraction() for vertex in vertices},
        interactions={
            ("x", "y"): Fraction(-1),
            ("y", "x"): Fraction(-1),
        },
        eta=eta_mod,
    )
    selected = frozenset({"y"})

    literal = direct_marginal(
        instance,
        selected,
        "x",
        "paper_samplewise_literal",
    )
    modular = direct_marginal(
        instance,
        selected,
        "x",
        "modular_shift_candidate",
    )

    assert instance.eta == Fraction(4)
    assert literal == Fraction(-2)
    assert modular == Fraction(2)
    assert modular == literal + instance.eta


@pytest.mark.parametrize(
    ("selected", "candidate"),
    (
        ({"y"}, "x"),
        (frozenset({"missing"}), "x"),
        (frozenset({"y"}), "missing"),
        (frozenset({"x"}), "x"),
        (frozenset({1}), "x"),
    ),
)
def test_marginals_reject_invalid_selection_and_candidate(
    selected: object,
    candidate: object,
) -> None:
    instance = Instance(
        ("x", "y"),
        {"x": Fraction(), "y": Fraction()},
        {},
    )

    for marginal in (direct_marginal, closed_form_marginal):
        with pytest.raises((TypeError, ValueError)):
            marginal(
                instance,
                selected,  # type: ignore[arg-type]
                candidate,  # type: ignore[arg-type]
                "paper_mwcp",
            )


def test_marginals_and_enumerator_reject_score_only_variant() -> None:
    instance = Instance(("x",), {"x": Fraction()}, {})

    for marginal in (direct_marginal, closed_form_marginal):
        with pytest.raises(ValueError, match="not a set function"):
            marginal(
                instance,
                frozenset(),
                "x",
                "appendix_eq26_score",
            )
    with pytest.raises(ValueError, match="not a set function"):
        enumerate_diminishing_returns(instance, "appendix_eq26_score")


def test_enumerator_rejects_stored_parameter_mismatch() -> None:
    graph = Instance(
        ("x", "y"),
        {"x": Fraction(), "y": Fraction()},
        {
            ("x", "y"): Fraction(-1),
            ("y", "x"): Fraction(-1),
        },
    )

    with pytest.raises(ValueError, match="stored canonical parameters"):
        enumerate_diminishing_returns(
            graph,
            "appendix_inline_shift_literal",
        )


def test_enumerator_has_exact_triples_and_unique_parameterized_ids() -> None:
    instance = Instance(
        ("x", "y"),
        {"x": Fraction(), "y": Fraction()},
        {
            ("x", "y"): Fraction(1),
            ("y", "x"): Fraction(1),
        },
    )

    records = enumerate_diminishing_returns(instance, "paper_mwcp")

    assert len(records) == 6
    assert len({record["id"] for record in records}) == 6
    assert all(
        record["id"].startswith(
            "graph=n=2;vw=0/1,0/1;ew=1/1"
            "::variant=paper_mwcp::alpha=1/1::eta=0/1::"
        )
        for record in records
    )
    assert all(
        record["direct_a"] == record["closed_a"]
        and record["direct_b"] == record["closed_b"]
        for record in records
    )
    violation = next(
        record
        for record in records
        if record["a"] == []
        and record["b"] == ["v1"]
        and record["candidate"] == "v0"
    )
    assert violation["direct_a"] == "0/1"
    assert violation["direct_b"] == "1/1"
    assert violation["diminishing_returns_holds"] is False


def test_enumerator_rejects_duplicate_case_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = Instance(
        ("x", "y"),
        {"x": Fraction(), "y": Fraction()},
        {},
    )

    monkeypatch.setattr(
        diminishing_returns,
        "_diminishing_case_id",
        lambda *_args, **_kwargs: "duplicate",
    )

    with pytest.raises(ValueError, match="duplicate diminishing-returns ID"):
        enumerate_diminishing_returns(instance, "paper_mwcp")


def test_diminishing_audit_has_exact_domain_and_primitive_ceilings(
    diminishing_audit: dict[str, object],
) -> None:
    symmetric = diminishing_audit["symmetric_search"]
    asymmetric = diminishing_audit["asymmetric_diagnostic"]

    assert diminishing_audit["source_revision"] == "task-3-test-revision"
    assert symmetric["evidence_kind"] == "exhaustive_finite"
    assert symmetric["declared_case_ceiling"] == 79_480
    assert symmetric["cases_examined"] == 79_480
    assert symmetric["variant_cases_examined"] == 476_880
    assert symmetric["subset_objective_evaluations"] == 1_907_520
    assert symmetric["closed_form_marginal_evaluations"] == 953_760
    assert symmetric["primitive_evaluations"] == 2_861_280
    assert symmetric["completed"] is True
    assert tuple(symmetric["model_variants"]) == SET_FUNCTION_VARIANTS

    assert asymmetric["evidence_kind"] == "exhaustive_finite"
    assert asymmetric["declared_case_ceiling"] == 19_738
    assert asymmetric["cases_examined"] == 19_738
    assert asymmetric["subset_objective_evaluations"] == 78_952
    assert asymmetric["closed_form_marginal_evaluations"] == 39_476
    assert asymmetric["primitive_evaluations"] == 118_428
    assert asymmetric["model_variants"] == ["paper_samplewise_literal"]
    assert asymmetric["completed"] is True


def test_diminishing_audit_is_deterministic_and_json_exact(
    diminishing_audit: dict[str, object],
) -> None:
    encoded = json.dumps(
        diminishing_audit,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")

    assert b"runtime" not in encoded
    assert len(diminishing_audit["symmetric_search"]["case_digest"]) == 64
    assert len(
        diminishing_audit["asymmetric_diagnostic"]["case_digest"]
    ) == 64
    assert diminishing_audit["appendix_minimal_witness"] == (
        appendix_shift_witness()
    )


@pytest.mark.parametrize("source_revision", ("", 7))
def test_diminishing_audit_rejects_invalid_source_revision(
    source_revision: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        run_diminishing_returns_audit(
            source_revision,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("symmetric_max_vertices", "asymmetric_max_vertices"),
    ((5, 3), (4, 4), (3, 3)),
)
def test_diminishing_audit_rejects_undeclared_domains_before_iteration(
    symmetric_max_vertices: int,
    asymmetric_max_vertices: int,
) -> None:
    with pytest.raises(ValueError, match="approved diminishing-returns domain"):
        run_diminishing_returns_audit(
            "task-3-test-revision",
            symmetric_max_vertices=symmetric_max_vertices,
            asymmetric_max_vertices=asymmetric_max_vertices,
        )
