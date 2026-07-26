import json
from fractions import Fraction

import pytest

import graph_pruning_repro.shifts as shifts_module
from graph_pruning_repro.diminishing_returns import (
    canonical_parameterized_instance_id,
)
from graph_pruning_repro.shifts import (
    eq26_score_marginal,
    fixed_literal_shift_marginal,
    run_shift_audit,
    shift_boundary_case_count,
    shift_boundary_ceiling,
)
from graph_pruning_repro.types import Instance


@pytest.fixture(scope="module")
def shift_audit() -> dict[str, object]:
    return run_shift_audit("task-3-test-revision")


def test_eq27_boundary_values_are_exact_below_at_and_above(
    shift_audit: dict[str, object],
) -> None:
    diagnostic = shift_audit["representative_eq27_boundary"]

    assert diagnostic["threshold"] == "1/1"
    assert [record["position"] for record in diagnostic["values"]] == [
        "below",
        "at",
        "above",
    ]
    assert [record["eta"] for record in diagnostic["values"]] == [
        "0/1",
        "1/1",
        "2/1",
    ]
    assert [
        record["channels"]["appendix_eq26_score"]
        for record in diagnostic["values"]
    ] == ["-1/1", "0/1", "1/1"]
    assert [
        record["channels"]["appendix_inline_shift_literal"]
        for record in diagnostic["values"]
    ] == ["-2/1", "1/1", "4/1"]
    assert [
        record["channels"]["modular_shift_candidate"]
        for record in diagnostic["values"]
    ] == ["0/1", "0/1", "0/1"]


def test_modular_channel_uses_exact_literal_base_graph_coefficient(
    shift_audit: dict[str, object],
) -> None:
    diagnostic = shift_audit["representative_eq27_boundary"]
    modular = diagnostic["canonical_modular"]

    graph = Instance(
        ("x", "y"),
        {"x": Fraction(), "y": Fraction()},
        {
            ("x", "y"): Fraction(-1),
            ("y", "x"): Fraction(-1),
        },
        eta=Fraction(2),
    )
    assert modular == {
        "instance_id": canonical_parameterized_instance_id(
            graph,
            "modular_shift_candidate",
        ),
        "coefficient": "2/1",
        "marginal": "0/1",
    }
    assert modular["instance_id"].endswith(
        "::variant=modular_shift_candidate::alpha=1/1::eta=2/1"
    )

    fixed = diagnostic["fixed_literal_shift_diagnostics"]
    assert [record["coefficient"] for record in fixed] == [
        "1/1",
        "2/1",
        "3/1",
    ]
    assert [record["marginal"] for record in fixed] == [
        "-1/1",
        "0/1",
        "1/1",
    ]
    assert all(
        "::diagnostic=fixed-literal-shift::" in record["id"]
        for record in fixed
    )
    assert all(
        "variant=modular_shift_candidate" not in record["id"]
        for record in fixed
    )


def test_edgeless_m_zero_deduplicates_boundary_without_off_by_one(
    shift_audit: dict[str, object],
) -> None:
    boundary = shift_audit["boundary_search"]

    assert shift_boundary_ceiling() == 6_459
    assert shift_boundary_case_count() == 6_410
    assert boundary["declared_case_ceiling"] == 6_459
    assert boundary["cases_examined"] == 6_410
    assert boundary["channels_per_case"] == 7
    assert boundary["values_evaluated"] == 44_870
    assert boundary["declared_value_ceiling"] == 45_213
    assert boundary["completed"] is True
    assert boundary["deduplication"] == (
        "nonnegative below/at values coincide when M=0"
    )


def test_rational_alpha_controls_are_labeled_non_exhaustive(
    shift_audit: dict[str, object],
) -> None:
    controls = shift_audit["rational_alpha_controls"]

    assert controls["evidence_kind"] == "non_exhaustive"
    assert controls["declared_case_ceiling"] == 256
    assert controls["cases_examined"] == 256
    assert controls["channels_per_case"] == 7
    assert controls["values_evaluated"] == 1_792
    assert controls["completed"] is True
    assert controls["supports_universal_conclusion"] is False


def test_shift_formula_helpers_reject_candidate_and_selection_violations() -> None:
    instance = Instance(
        ("x", "y"),
        {"x": Fraction(), "y": Fraction()},
        {},
    )

    for function in (eq26_score_marginal, fixed_literal_shift_marginal):
        with pytest.raises(ValueError, match="candidate must not be selected"):
            if function is fixed_literal_shift_marginal:
                function(
                    instance,
                    frozenset({"x"}),
                    "x",
                    Fraction(),
                )
            else:
                function(instance, frozenset({"x"}), "x")


def test_shift_formula_helpers_use_exact_fractions() -> None:
    instance = Instance(
        ("x", "y"),
        {"x": Fraction(1, 2), "y": Fraction()},
        {
            ("x", "y"): Fraction(-1),
            ("y", "x"): Fraction(-1),
        },
        alpha=Fraction(3, 2),
        eta=Fraction(2, 3),
    )

    assert (
        eq26_score_marginal(instance, frozenset({"y"}), "x")
        == Fraction(3, 4)
    )
    assert (
        fixed_literal_shift_marginal(
            instance,
            frozenset({"y"}),
            "x",
            Fraction(5, 2),
        )
        == Fraction(1)
    )


def test_shift_audit_is_byte_deterministic_and_has_no_runtime(
    shift_audit: dict[str, object],
) -> None:
    first = json.dumps(
        shift_audit,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    second = json.dumps(
        run_shift_audit("task-3-test-revision"),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")

    assert first == second
    assert b"runtime" not in first


@pytest.mark.parametrize("source_revision", ("", 7))
def test_shift_audit_rejects_invalid_source_revision(
    source_revision: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        run_shift_audit(source_revision)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("max_vertices", "rational_controls"),
    ((5, 256), (3, 256), (4, 257), (4, 255)),
)
def test_shift_audit_rejects_undeclared_domain_before_iteration(
    max_vertices: int,
    rational_controls: int,
) -> None:
    with pytest.raises(ValueError, match="approved shift domain"):
        run_shift_audit(
            "task-3-test-revision",
            max_vertices=max_vertices,
            rational_controls=rational_controls,
        )


@pytest.mark.parametrize(
    ("ceiling_name", "too_small"),
    (
        ("SHIFT_CASE_CEILING", 6_458),
        ("SHIFT_VALUE_CEILING", 45_212),
        ("RATIONAL_ALPHA_CASE_CEILING", 255),
        ("RATIONAL_ALPHA_VALUE_CEILING", 1_791),
    ),
)
def test_all_shift_ceilings_preflight_before_iteration(
    monkeypatch: pytest.MonkeyPatch,
    ceiling_name: str,
    too_small: int,
) -> None:
    iterations = 0

    def forbidden_product(*_args: object, **_kwargs: object) -> object:
        nonlocal iterations
        iterations += 1
        raise AssertionError("shift audit started iteration")

    monkeypatch.setattr(shifts_module, ceiling_name, too_small)
    monkeypatch.setattr(shifts_module, "product", forbidden_product)

    with pytest.raises(ValueError, match="ceiling"):
        run_shift_audit("task-3-preflight-regression")
    assert iterations == 0
