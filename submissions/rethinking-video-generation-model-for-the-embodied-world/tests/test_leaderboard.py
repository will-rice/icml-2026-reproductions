import pytest

from conftest import LEADERBOARD_DIMENSIONS, leaderboard_row
from rbench_repro.leaderboard import (
    Formula,
    audit_leaderboard,
    compare_cohorts,
    derive_formula,
    derive_groups,
    infer_displayed_mean_formula,
)


def test_formula_is_recovered_from_inert_python_source(formula_sources):
    formula = derive_formula(*formula_sources)
    assert formula == Formula(
        columns=LEADERBOARD_DIMENSIONS,
        operation="mean",
        source_precision=4,
        output_precision=3,
        rounding="half_even",
        absolute_tolerance="0.0005",
    )


def test_untraceable_formula_is_rejected():
    with pytest.raises(ValueError, match="untraceable leaderboard formula"):
        derive_formula(b"DIMENSIONS = ['Common Manipulation']\n", b"score = 1\n")


def test_formula_can_be_inferred_from_consistent_displayed_records():
    rows = [leaderboard_row("Model-1", "0.1005")]
    rows[0]["avg"] = "0.100"
    formula = infer_displayed_mean_formula(
        rows,
        LEADERBOARD_DIMENSIONS,
    )
    assert formula == Formula(
        columns=LEADERBOARD_DIMENSIONS,
        operation="mean",
        source_precision=3,
        output_precision=3,
        rounding="half_even",
        absolute_tolerance="0.001",
    )


def test_formula_inference_rejects_material_average_discrepancy():
    rows = [leaderboard_row("Model-1", "0.100")]
    rows[0]["avg"] = "0.106"
    with pytest.raises(ValueError, match="inconsistent displayed mean"):
        infer_displayed_mean_formula(rows, LEADERBOARD_DIMENSIONS)


def test_formula_rejects_disconnected_rounding_precision(formula_sources):
    utils_source, app_source = formula_sources
    disconnected = utils_source.replace(
        b"values = [round(row[column], 4) for column in DIMENSION_COLUMNS]",
        b"values = [row[column] for column in DIMENSION_COLUMNS]",
    ) + b"\ndef unrelated(row):\n    return round(row['other'], 4)\n"
    with pytest.raises(ValueError, match="untraceable leaderboard formula"):
        derive_formula(disconnected, app_source)


def test_formula_rejects_disconnected_category_columns(formula_sources):
    utils_source, app_source = formula_sources
    disconnected = utils_source.replace(
        b"for column in DIMENSION_COLUMNS",
        b"for column in OTHER_COLUMNS",
    ).replace(
        b"def leaderboard_average(row):",
        b"OTHER_COLUMNS = ['other']\ndef leaderboard_average(row):",
    )
    with pytest.raises(ValueError, match="untraceable leaderboard formula"):
        derive_formula(disconnected, app_source)


def test_groups_are_recovered_without_importing_app_source():
    source = (
        b'OPEN_SOURCE_MODELS = ["Model-1", "Model-2"]\n'
        b'COMMERCIAL_MODELS = ("Model-3",)\n'
        b'IGNORED_LABEL = "not a group"\n'
        b'raise RuntimeError("must remain inert")\n'
    )
    assert derive_groups(source) == {
        "commercial": ("Model-3",),
        "open_source": ("Model-1", "Model-2"),
    }


def test_groups_are_recovered_from_current_space_mapping():
    source = (
        b'GROUPS = {"Open-source": ["Model-1", "Model-2"], '
        b'"Robotics-specific": ["Model-3"]}\n'
        b'raise RuntimeError("must remain inert")\n'
    )
    assert derive_groups(source) == {
        "open-source": ("Model-1", "Model-2"),
        "robotics-specific": ("Model-3",),
    }


def test_paper_and_later_cohorts_remain_separate(
    leaderboard_sources, recovered_formula, groups
):
    paper = audit_leaderboard(
        leaderboard_sources["paper"],
        "paper-era",
        "leaderboard.json",
        recovered_formula,
        groups,
    )
    later = audit_leaderboard(
        leaderboard_sources["later"],
        "later",
        "leaderboard.json",
        recovered_formula,
        groups,
    )
    comparison = compare_cohorts(paper, later).to_dict()
    assert (paper.raw_count, paper.valid_count, paper.unique_normalized_count) == (25, 25, 25)
    assert (later.raw_count, later.valid_count, later.unique_normalized_count) == (28, 28, 28)
    assert paper.ordered_name_hash == "0997c2cd82bb96e065cb3f1f3606f451859491845bbca7492dce1ade10a8c9aa"
    assert later.ordered_name_hash == "ed6242ddf922b260dfb5b96762444107dab0975626f0110b8e45d7f03924fba1"
    assert comparison["prepended_models"] == ["LingBot-Video", "Cosmos3-Nano", "Cosmos3-Super"]
    assert comparison["shared_records_field_equal"] is True


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01, 100.01, True])
def test_invalid_numeric_values_are_retained_as_schema_errors(
    one_row_source, recovered_formula, groups, bad
):
    row = leaderboard_row("Model-1")
    row["avg"] = bad
    one_row_source.replace_json("leaderboard.json", [row])
    result = audit_leaderboard(
        one_row_source, "paper-era", "leaderboard.json", recovered_formula, groups
    )
    assert result.valid_count == 0
    assert result.unique_exact_count == 0
    assert result.invalid_rows[0]["errors"]


def test_duplicate_collision_missing_dimension_and_group_orphan_are_reported(
    one_row_source, recovered_formula
):
    rows = [
        leaderboard_row("Model-1"),
        leaderboard_row(" MODEL-1 "),
        {"model": "incomplete", "avg": 0.0},
    ]
    one_row_source.replace_json("leaderboard.json", rows)
    result = audit_leaderboard(
        one_row_source,
        "paper-era",
        "leaderboard.json",
        recovered_formula,
        {"open": ("missing-model",)},
    )
    assert result.normalization_collisions == ((" MODEL-1 ", "Model-1"),)
    assert result.invalid_rows[-1]["errors"] == ["missing dimension columns"]
    assert result.group_orphans == ("missing-model",)


def test_decimal_mean_rounding_boundary_and_wrong_average(one_row_source, groups):
    formula = Formula(LEADERBOARD_DIMENSIONS, "mean", 4, 3, "half_even", "0.0005")
    row = leaderboard_row("Model-1", "0.1005")
    row["avg"] = "0.101"
    one_row_source.replace_json("leaderboard.json", [row])
    result = audit_leaderboard(one_row_source, "paper-era", "leaderboard.json", formula, groups)
    assert result.rows[0]["recomputed_avg"] == "0.100"
    assert result.discrepancies[0]["absolute_error"] == "0.001"


def test_supplementary_file_cannot_change_main_cohort_count(
    leaderboard_sources, recovered_formula, groups
):
    result = audit_leaderboard(
        leaderboard_sources["later"],
        "later",
        "leaderboard.json",
        recovered_formula,
        groups,
    )
    assert result.raw_count == 28


def test_duplicate_names_cannot_hide_unequal_shared_records(
    leaderboard_sources, recovered_formula, groups
):
    paper_rows = [leaderboard_row("duplicate", "1"), leaderboard_row("duplicate", "2")]
    later_rows = [leaderboard_row("duplicate", "9"), leaderboard_row("duplicate", "2")]
    leaderboard_sources["paper"].replace_json("leaderboard.json", paper_rows)
    leaderboard_sources["later"].replace_json("leaderboard.json", later_rows)
    paper = audit_leaderboard(
        leaderboard_sources["paper"], "paper-era", "leaderboard.json", recovered_formula, groups
    )
    later = audit_leaderboard(
        leaderboard_sources["later"], "later", "leaderboard.json", recovered_formula, groups
    )
    assert compare_cohorts(paper, later).shared_records_field_equal is False
