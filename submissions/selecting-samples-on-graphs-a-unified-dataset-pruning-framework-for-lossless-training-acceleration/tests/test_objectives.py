from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

import graph_pruning_repro.objectives as objectives
from graph_pruning_repro.objectives import evaluate_objective
from graph_pruning_repro.types import MODEL_VARIANTS, Instance


INSTANCE = Instance(
    vertices=("x", "y"),
    vertex_weights={"x": Fraction(1), "y": Fraction(2)},
    interactions={
        ("x", "y"): Fraction(-1),
        ("y", "x"): Fraction(-1),
    },
)


def test_half_correction_matches_mwcp() -> None:
    selected = frozenset({"x", "y"})

    assert (
        evaluate_objective(INSTANCE, selected, "half_corrected_samplewise")
        == Fraction(2)
    )


def test_eq26_is_not_misrepresented_as_objective() -> None:
    with pytest.raises(ValueError, match="score, not a set function"):
        evaluate_objective(INSTANCE, frozenset(), "appendix_eq26_score")


def test_all_seven_variant_names_are_exact_and_dispatch_is_exact() -> None:
    assert MODEL_VARIANTS == (
        "paper_mwcp",
        "paper_samplewise_literal",
        "single_counted_pairwise",
        "half_corrected_samplewise",
        "appendix_inline_shift_literal",
        "appendix_eq26_score",
        "modular_shift_candidate",
    )
    shifted = Instance(
        vertices=INSTANCE.vertices,
        vertex_weights=INSTANCE.vertex_weights,
        interactions=INSTANCE.interactions,
        alpha=Fraction(2),
        eta=Fraction(3),
    )
    selected = frozenset({"x", "y"})
    expected = {
        "paper_mwcp": Fraction(2),
        "paper_samplewise_literal": Fraction(1),
        "single_counted_pairwise": Fraction(2),
        "half_corrected_samplewise": Fraction(2),
        "appendix_inline_shift_literal": Fraction(25),
        "modular_shift_candidate": Fraction(7),
    }
    actual = {
        variant: evaluate_objective(shifted, selected, variant)
        for variant in expected
    }

    assert actual == expected
    assert all(type(value) is Fraction for value in actual.values())


def test_modular_shift_always_uses_literal_double_counted_base() -> None:
    shifted = Instance(
        vertices=INSTANCE.vertices,
        vertex_weights=INSTANCE.vertex_weights,
        interactions=INSTANCE.interactions,
        eta=Fraction(3),
    )

    result = evaluate_objective(
        shifted,
        frozenset({"x", "y"}),
        "modular_shift_candidate",
    )

    assert result == Fraction(7)
    assert result != Fraction(8)


def test_independent_traversals_handle_asymmetric_interactions_exactly() -> None:
    instance = Instance(
        vertices=("x", "y"),
        vertex_weights={"x": Fraction(), "y": Fraction()},
        interactions={
            ("x", "y"): Fraction(2),
            ("y", "x"): Fraction(-5),
        },
    )
    selected = frozenset({"x", "y"})

    assert objectives.evaluate_mwcp_edges(instance, selected) == Fraction(2)
    assert objectives.evaluate_samplewise_literal(instance, selected) == Fraction(-3)


def test_mwcp_traversal_does_not_call_samplewise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> Fraction:
        raise AssertionError("MWCP traversal called samplewise traversal")

    monkeypatch.setattr(objectives, "evaluate_samplewise_literal", forbidden)

    assert (
        objectives.evaluate_mwcp_edges(INSTANCE, frozenset({"x", "y"}))
        == Fraction(2)
    )


def test_samplewise_traversal_does_not_call_mwcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> Fraction:
        raise AssertionError("samplewise traversal called MWCP traversal")

    monkeypatch.setattr(objectives, "evaluate_mwcp_edges", forbidden)

    assert (
        objectives.evaluate_samplewise_literal(
            INSTANCE,
            frozenset({"x", "y"}),
        )
        == Fraction(1)
    )


def test_instance_is_ordered_and_deeply_frozen() -> None:
    vertex_weights = {"x": Fraction(1), "y": Fraction(2)}
    interactions = {
        ("x", "y"): Fraction(-1),
        ("y", "x"): Fraction(-1),
    }
    instance = Instance(("x", "y"), vertex_weights, interactions)
    vertex_weights["x"] = Fraction(99)
    interactions[("x", "y")] = Fraction(99)

    assert instance.vertices == ("x", "y")
    assert instance.vertex_weights["x"] == Fraction(1)
    assert instance.interactions[("x", "y")] == Fraction(-1)
    with pytest.raises(TypeError):
        instance.vertex_weights["x"] = Fraction(3)  # type: ignore[index]
    with pytest.raises(TypeError):
        instance.interactions[("x", "y")] = Fraction(3)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        instance.vertices = ("y", "x")  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "vertices": ["x"],
            "vertex_weights": {"x": Fraction()},
            "interactions": {},
        },
        {
            "vertices": (),
            "vertex_weights": {},
            "interactions": {},
        },
        {
            "vertices": ("x", "x"),
            "vertex_weights": {"x": Fraction()},
            "interactions": {},
        },
        {
            "vertices": ("",),
            "vertex_weights": {"": Fraction()},
            "interactions": {},
        },
        {
            "vertices": ("x", "y"),
            "vertex_weights": {"x": Fraction()},
            "interactions": {},
        },
        {
            "vertices": ("x",),
            "vertex_weights": {"x": 0},
            "interactions": {},
        },
        {
            "vertices": ("x",),
            "vertex_weights": {"x": Fraction(), "z": Fraction()},
            "interactions": {},
        },
        {
            "vertices": ("x", "y"),
            "vertex_weights": {"x": Fraction(), "y": Fraction()},
            "interactions": {("x", "x"): Fraction()},
        },
        {
            "vertices": ("x", "y"),
            "vertex_weights": {"x": Fraction(), "y": Fraction()},
            "interactions": {("x", "z"): Fraction()},
        },
        {
            "vertices": ("x", "y"),
            "vertex_weights": {"x": Fraction(), "y": Fraction()},
            "interactions": {"x-y": Fraction()},
        },
        {
            "vertices": ("x", "y"),
            "vertex_weights": {"x": Fraction(), "y": Fraction()},
            "interactions": {("x", "y"): 1},
        },
        {
            "vertices": ("x",),
            "vertex_weights": {"x": Fraction()},
            "interactions": {},
            "alpha": 1,
        },
        {
            "vertices": ("x",),
            "vertex_weights": {"x": Fraction()},
            "interactions": {},
            "eta": 0,
        },
    ),
)
def test_instance_rejects_invalid_vertices_weights_and_interactions(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        Instance(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "selected",
    (
        {"x"},
        frozenset({"unknown"}),
        frozenset({1}),
    ),
)
def test_objectives_reject_invalid_selections(selected: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_objective(
            INSTANCE,
            selected,  # type: ignore[arg-type]
            "paper_mwcp",
        )


def test_objectives_reject_unknown_variant() -> None:
    with pytest.raises(ValueError, match="unknown model variant"):
        evaluate_objective(
            INSTANCE,
            frozenset({"x"}),
            "unknown",  # type: ignore[arg-type]
        )
