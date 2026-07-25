from fractions import Fraction

import pytest

from graph_pruning_repro.diminishing_returns import direct_marginal
from graph_pruning_repro.minimize import minimize_witness
from graph_pruning_repro.types import Instance


def _negative_literal_marginal(witness: dict[str, object]) -> bool:
    instance = Instance(
        witness["vertices"],
        witness["vertex_weights"],
        witness["interactions"],
    )
    return (
        direct_marginal(
            instance,
            witness["selected"],
            witness["candidate"],
            "paper_samplewise_literal",
        )
        < 0
    )


def _unminimized_witness() -> dict[str, object]:
    return {
        "property": "negative_literal_marginal",
        "vertices": ("z", "x", "y"),
        "selected": frozenset({"y"}),
        "candidate": "x",
        "vertex_weights": {
            "z": Fraction(5),
            "x": Fraction(2),
            "y": Fraction(7),
        },
        "interactions": {
            ("z", "x"): Fraction(9),
            ("x", "z"): Fraction(9),
            ("x", "y"): Fraction(-3),
            ("y", "x"): Fraction(-3),
        },
    }


def test_minimizer_preserves_property_in_approved_reduction_order() -> None:
    minimized = minimize_witness(
        _negative_literal_marginal,
        _unminimized_witness(),
    )

    assert _negative_literal_marginal(minimized)
    assert minimized["property"] == "negative_literal_marginal"
    assert minimized["vertices"] == ("v0", "v1")
    assert minimized["candidate"] == "v0"
    assert minimized["selected"] == frozenset({"v1"})
    assert minimized["vertex_weights"] == {
        "v0": Fraction(),
        "v1": Fraction(),
    }
    assert minimized["interactions"] == {
        ("v0", "v1"): Fraction(),
        ("v1", "v0"): Fraction(-1),
    }
    assert len(minimized["id"]) == 16
    assert set(minimized["id"]) <= set("0123456789abcdef")


def test_minimizer_is_idempotent_and_ignores_mapping_insertion_order() -> None:
    original = _unminimized_witness()
    reordered = {
        "interactions": dict(reversed(tuple(original["interactions"].items()))),
        "vertex_weights": dict(
            reversed(tuple(original["vertex_weights"].items()))
        ),
        "candidate": original["candidate"],
        "selected": original["selected"],
        "vertices": original["vertices"],
        "property": original["property"],
    }

    first = minimize_witness(_negative_literal_marginal, original)
    second = minimize_witness(_negative_literal_marginal, reordered)
    third = minimize_witness(_negative_literal_marginal, first)

    assert first == second == third
    assert first["id"] == second["id"] == third["id"]


def test_minimized_ids_distinguish_preserved_property_metadata() -> None:
    first = minimize_witness(
        _negative_literal_marginal,
        _unminimized_witness(),
    )
    changed = _unminimized_witness()
    changed["property"] = "different_property"
    second = minimize_witness(_negative_literal_marginal, changed)

    assert first["id"] != second["id"]


def test_minimizer_rejects_original_that_does_not_preserve_property() -> None:
    with pytest.raises(ValueError, match="original witness"):
        minimize_witness(lambda _witness: False, _unminimized_witness())


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_vertices",
        "candidate_selected",
        "unknown_selected",
        "missing_weight",
        "non_fraction",
        "bad_interaction",
    ),
)
def test_minimizer_rejects_invalid_witness_domain(mutation: str) -> None:
    witness = _unminimized_witness()
    if mutation == "duplicate_vertices":
        witness["vertices"] = ("x", "x")
    elif mutation == "candidate_selected":
        witness["selected"] = frozenset({"x"})
    elif mutation == "unknown_selected":
        witness["selected"] = frozenset({"missing"})
    elif mutation == "missing_weight":
        witness["vertex_weights"] = {"x": Fraction()}
    elif mutation == "non_fraction":
        witness["vertex_weights"] = {
            "z": Fraction(),
            "x": 0,
            "y": Fraction(),
        }
    elif mutation == "bad_interaction":
        witness["interactions"] = {("x", "missing"): Fraction(-1)}

    with pytest.raises((TypeError, ValueError)):
        minimize_witness(_negative_literal_marginal, witness)


def test_minimizer_rejects_non_callable_predicate() -> None:
    with pytest.raises(TypeError, match="predicate must be callable"):
        minimize_witness(None, _unminimized_witness())  # type: ignore[arg-type]
