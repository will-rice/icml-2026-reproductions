"""Tests for delta-based precision in StepEvaluator.

Delta precision only counts properties the prediction actually changed
(vs pre-state), preventing history TPs from inflating precision.
"""

import copy

import pytest

from next_action_pred_eval.core.state import StateBuilder
from next_action_pred_eval.core.symbolic import symbolic_to_operations
from next_action_pred_eval.evaluation.evaluator import StepEvaluator


def _build_state(symbolic_ops):
    """Build workbook state from symbolic operation strings."""
    ops = symbolic_to_operations(symbolic_ops)
    builder = StateBuilder()
    builder.apply_operations(ops)
    return builder.state


def _eval(history, prediction, future_gt):
    """Run StepEvaluator.evaluate with standard setup."""
    initial_state = _build_state(history)
    target_builder = StateBuilder(copy.deepcopy(initial_state))
    target_builder.apply_operations(symbolic_to_operations(future_gt))
    target_state = target_builder.state

    evaluator = StepEvaluator()
    return evaluator.evaluate(
        ground_truth_operations=future_gt[:len(prediction)],
        predicted_operations=prediction,
        lookahead_window=None,
        all_future_operations=future_gt,
        initial_state_cache=initial_state,
        lookahead_state_cache=target_state,
        skip_ops_diff=True,
    )


class TestDeltaPrecision:
    """Verify delta-based precision only reflects prediction's own contribution."""

    def test_bad_prediction_history_tp_no_longer_inflates(self):
        """
        History builds A1=10, A2=20, A3=30.
        Prediction changes A3 to 999 (wrong — target wants A3=30).

        OLD precision: TP=2(A1,A2) + MM=1(A3) → 2/3 ≈ 0.667
        DELTA precision: only A3 changed → delta_mm=1 → 0/1 = 0.0
        """
        result = _eval(
            history=["VALUE | Sheet1!A1 | 10", "VALUE | Sheet1!A2 | 20",
                     "VALUE | Sheet1!A3 | 30"],
            prediction=["VALUE | Sheet1!A3 | 999"],
            future_gt=["VALUE | Sheet1!A3 | 30", "VALUE | Sheet1!A4 | 40"],
        )
        assert result.metrics.final_state_precision == 0.0
        assert result.metrics.final_state_mm >= 1

    def test_good_prediction_gets_full_credit(self):
        """
        History: A1=10, A2=20. Prediction: A3=30 (correct).
        Delta: A3 added, matches target → 1/1 = 1.0.
        """
        result = _eval(
            history=["VALUE | Sheet1!A1 | 10", "VALUE | Sheet1!A2 | 20"],
            prediction=["VALUE | Sheet1!A3 | 30"],
            future_gt=["VALUE | Sheet1!A3 | 30", "VALUE | Sheet1!A4 | 40"],
        )
        assert result.metrics.final_state_precision == 1.0
        assert result.metrics.final_state_tp >= 1

    def test_clear_matching_target_is_tp(self):
        """
        History sets bold on A1. Prediction clears it. Target also has no bold.
        Both post and target are default → delta TP (absent from diffs = TP).
        """
        result = _eval(
            history=["VALUE | Sheet1!A1 | Hello", "FONT_BOLD | Sheet1!A1 | True"],
            prediction=["FONT_BOLD | Sheet1!A1 | False"],
            future_gt=["FONT_BOLD | Sheet1!A1 | False", "VALUE | Sheet1!A2 | World"],
        )
        assert result.metrics.final_state_precision == 1.0
        assert result.metrics.final_state_tp >= 1
        assert result.metrics.final_state_fp == 0
        assert result.metrics.final_state_mm == 0

    def test_unchanged_property_not_counted(self):
        """
        Prediction sets A1=10, same as history. No change → empty delta_props.
        """
        result = _eval(
            history=["VALUE | Sheet1!A1 | 10", "VALUE | Sheet1!A2 | 20"],
            prediction=["VALUE | Sheet1!A1 | 10"],
            future_gt=["VALUE | Sheet1!A1 | 10", "VALUE | Sheet1!A3 | 30"],
        )
        assert result.metrics.final_state_tp == 0
        assert result.metrics.final_state_fp == 0
        assert result.metrics.final_state_mm == 0

    def test_multi_property_mixed(self):
        """
        Prediction: A1=99 (wrong, was 10), A3=30 (correct, new).
        Delta: A1.value MM, A3.value TP → precision 0.5.
        """
        result = _eval(
            history=["VALUE | Sheet1!A1 | 10", "VALUE | Sheet1!A2 | 20"],
            prediction=["VALUE | Sheet1!A1 | 99", "VALUE | Sheet1!A3 | 30"],
            future_gt=["VALUE | Sheet1!A3 | 30", "VALUE | Sheet1!A4 | 40"],
        )
        assert result.metrics.final_state_tp == 1
        assert result.metrics.final_state_mm == 1
        assert result.metrics.final_state_fp == 0
        assert result.metrics.final_state_precision == pytest.approx(0.5)

    # ---------- Format operations ----------

    def test_format_bold_correct(self):
        """Prediction bolds A1, target also bolds A1 → TP."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | Hello"],
            prediction=["FONT_BOLD | Sheet1!A1 | True"],
            future_gt=["FONT_BOLD | Sheet1!A1 | True"],
        )
        assert result.metrics.final_state_tp >= 1
        assert result.metrics.final_state_precision == 1.0

    def test_format_bold_wrong(self):
        """Prediction bolds A1, target doesn't bold A1 → FP."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | Hello"],
            prediction=["FONT_BOLD | Sheet1!A1 | True"],
            future_gt=["VALUE | Sheet1!A2 | World"],
        )
        assert result.metrics.final_state_fp >= 1
        assert result.metrics.final_state_precision == 0.0

    def test_format_italic_with_history_bold(self):
        """
        History has bold on A1. Prediction adds italic on A1 (wrong, target
        doesn't need italic). Bold is NOT in delta → not counted.
        """
        result = _eval(
            history=["VALUE | Sheet1!A1 | Hi", "FONT_BOLD | Sheet1!A1 | True"],
            prediction=["FONT_ITALIC | Sheet1!A1 | True"],
            future_gt=["VALUE | Sheet1!A2 | World"],
        )
        # Only italic changed — it's an FP (target doesn't want it)
        assert result.metrics.final_state_fp >= 1
        assert result.metrics.final_state_tp == 0
        assert result.metrics.final_state_precision == 0.0

    def test_fill_color_correct(self):
        """Prediction sets fill color, target also sets same fill color."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | Data"],
            prediction=["FILL_COLOR | Sheet1!A1 | #FF0000"],
            future_gt=["FILL_COLOR | Sheet1!A1 | #FF0000"],
        )
        assert result.metrics.final_state_precision == 1.0

    def test_fill_color_wrong(self):
        """Prediction sets red fill, target wants blue fill → MM."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | Data"],
            prediction=["FILL_COLOR | Sheet1!A1 | #FF0000"],
            future_gt=["FILL_COLOR | Sheet1!A1 | #0000FF"],
        )
        assert result.metrics.final_state_mm >= 1
        assert result.metrics.final_state_precision == 0.0

    # ---------- Multi-cell ranges (INPUT) ----------

    def test_input_range_all_correct(self):
        """INPUT on A1:B2 with correct values → all TPs."""
        result = _eval(
            history=[],
            prediction=["INPUT | Sheet1!A1:B2 | [[1, 2], [3, 4]]"],
            future_gt=["INPUT | Sheet1!A1:B2 | [[1, 2], [3, 4]]"],
        )
        assert result.metrics.final_state_tp >= 4
        assert result.metrics.final_state_fp == 0
        assert result.metrics.final_state_mm == 0
        assert result.metrics.final_state_precision == 1.0

    def test_input_range_partially_wrong(self):
        """INPUT on A1:B1 — prediction [1, 99], target [1, 2]."""
        result = _eval(
            history=[],
            prediction=["INPUT | Sheet1!A1:B1 | [[1, 99]]"],
            future_gt=["INPUT | Sheet1!A1:B1 | [[1, 2]]"],
        )
        # A1=1 correct (TP), B1=99 vs 2 (MM)
        assert result.metrics.final_state_tp >= 1
        assert result.metrics.final_state_mm >= 1
        assert result.metrics.final_state_precision == pytest.approx(0.5)

    def test_input_overwrites_history_correctly(self):
        """
        History: A1=old, A2=old2. Prediction: INPUT A1:A2 = [new1, new2].
        Target also has new1, new2. Both changed → both TP.
        """
        result = _eval(
            history=["VALUE | Sheet1!A1 | old", "VALUE | Sheet1!A2 | old2"],
            prediction=["INPUT | Sheet1!A1:A2 | [[\"new1\"], [\"new2\"]]"],
            future_gt=["INPUT | Sheet1!A1:A2 | [[\"new1\"], [\"new2\"]]"],
        )
        assert result.metrics.final_state_tp >= 2
        assert result.metrics.final_state_precision == 1.0

    # ---------- Fresh cell (no history) ----------

    def test_prediction_on_fresh_cell_fp(self):
        """Prediction writes to cell with no history and not in target → FP."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | 10"],
            prediction=["VALUE | Sheet1!Z99 | junk"],
            future_gt=["VALUE | Sheet1!A2 | 20"],
        )
        assert result.metrics.final_state_fp >= 1
        assert result.metrics.final_state_tp == 0
        assert result.metrics.final_state_precision == 0.0

    def test_prediction_on_fresh_cell_correct(self):
        """Prediction writes to new cell matching target → TP."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | 10"],
            prediction=["VALUE | Sheet1!A2 | 20"],
            future_gt=["VALUE | Sheet1!A2 | 20"],
        )
        assert result.metrics.final_state_tp >= 1
        assert result.metrics.final_state_precision == 1.0

    # ---------- Overwrites history with correct value ----------

    def test_overwrite_wrong_history_correctly(self):
        """
        History set A1=wrong. Prediction corrects A1=right (target wants right).
        Delta: A1 changed → post_vs_target TP → precision 1.0.
        """
        result = _eval(
            history=["VALUE | Sheet1!A1 | wrong"],
            prediction=["VALUE | Sheet1!A1 | right"],
            future_gt=["VALUE | Sheet1!A1 | right"],
        )
        assert result.metrics.final_state_tp >= 1
        assert result.metrics.final_state_precision == 1.0

    def test_overwrite_history_with_wrong_value(self):
        """History A1=10, prediction A1=99, target wants A1=10 → MM."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | 10"],
            prediction=["VALUE | Sheet1!A1 | 99"],
            future_gt=["VALUE | Sheet1!A1 | 10"],
        )
        assert result.metrics.final_state_mm >= 1
        assert result.metrics.final_state_precision == 0.0

    # ---------- No initial_state_cache (None) ----------

    def test_no_initial_state_cache(self):
        """Evaluator builds empty pre-state when cache is None."""
        future_gt = ["VALUE | Sheet1!A1 | 10", "VALUE | Sheet1!A2 | 20"]
        target_state = _build_state(future_gt)

        evaluator = StepEvaluator()
        result = evaluator.evaluate(
            ground_truth_operations=future_gt[:1],
            predicted_operations=["VALUE | Sheet1!A1 | 10"],
            lookahead_window=None,
            all_future_operations=future_gt,
            initial_state_cache=None,
            lookahead_state_cache=target_state,
            skip_ops_diff=True,
        )
        # Fresh cell, correct value → TP
        assert result.metrics.final_state_tp >= 1
        assert result.metrics.final_state_precision == 1.0

    # ---------- Multiple properties on same cell ----------

    def test_value_and_format_same_cell(self):
        """
        History: A1=Hello. Prediction: A1=World (correct) + bold A1 (correct).
        Both changed, both match target → precision 1.0.
        """
        result = _eval(
            history=["VALUE | Sheet1!A1 | Hello"],
            prediction=[
                "VALUE | Sheet1!A1 | World",
                "FONT_BOLD | Sheet1!A1 | True",
            ],
            future_gt=[
                "VALUE | Sheet1!A1 | World",
                "FONT_BOLD | Sheet1!A1 | True",
            ],
        )
        assert result.metrics.final_state_tp >= 2
        assert result.metrics.final_state_precision == 1.0

    def test_value_correct_format_wrong_same_cell(self):
        """
        Prediction: A1=World (correct) + italic A1 (wrong, target doesn't want it).
        value → TP, italic → FP. Precision = 1/2 = 0.5.
        """
        result = _eval(
            history=["VALUE | Sheet1!A1 | Hello"],
            prediction=[
                "VALUE | Sheet1!A1 | World",
                "FONT_ITALIC | Sheet1!A1 | True",
            ],
            future_gt=[
                "VALUE | Sheet1!A1 | World",
                "VALUE | Sheet1!A2 | Next",
            ],
        )
        assert result.metrics.final_state_tp >= 1
        assert result.metrics.final_state_fp >= 1
        assert result.metrics.final_state_precision == pytest.approx(0.5)

    # ---------- Formula operations ----------

    def test_formula_correct(self):
        """Prediction adds formula that matches target."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | 10", "VALUE | Sheet1!A2 | 20"],
            prediction=["FORMULA | Sheet1!A3 | =SUM(A1:A2)"],
            future_gt=["FORMULA | Sheet1!A3 | =SUM(A1:A2)"],
        )
        assert result.metrics.final_state_tp >= 1
        assert result.metrics.final_state_precision == 1.0

    def test_formula_wrong(self):
        """Prediction adds wrong formula → MM."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | 10", "VALUE | Sheet1!A2 | 20"],
            prediction=["FORMULA | Sheet1!A3 | =SUM(A1:A2)"],
            future_gt=["FORMULA | Sheet1!A3 | =AVERAGE(A1:A2)"],
        )
        assert result.metrics.final_state_mm >= 1
        assert result.metrics.final_state_precision == 0.0

    # ---------- Prediction removes value target needs ----------

    def test_clear_value_target_needs(self):
        """
        History: A1=Hello. Prediction clears A1. Target needs A1=Hello.
        The prediction removed something the target needs → counts as MM.
        """
        result = _eval(
            history=["VALUE | Sheet1!A1 | Hello"],
            prediction=["VALUE | Sheet1!A1 | "],
            future_gt=["VALUE | Sheet1!A2 | World"],
        )
        m = result.metrics
        # A1 changed (Hello→empty). Target still has A1=Hello from history.
        # post has empty/default, target has Hello → FN in diffs → delta_mm
        assert m.final_state_precision == 0.0
        assert m.final_state_tp == 0

    # ---------- Large history, tiny prediction ----------

    def test_large_history_single_pred_wrong(self):
        """
        20 history cells, prediction touches 1 cell wrongly.
        Old precision would be ~19/20 = 0.95.
        Delta precision should be 0.0.
        """
        history = [f"VALUE | Sheet1!A{i} | {i}" for i in range(1, 21)]
        prediction = ["VALUE | Sheet1!A1 | WRONG"]
        future_gt = [f"VALUE | Sheet1!A{i} | {i}" for i in range(1, 21)]

        result = _eval(history=history, prediction=prediction, future_gt=future_gt)
        assert result.metrics.final_state_mm >= 1
        assert result.metrics.final_state_tp == 0
        assert result.metrics.final_state_precision == 0.0

    def test_large_history_single_pred_correct(self):
        """
        20 history cells, prediction correctly adds cell 21.
        Delta precision = 1.0 (only cell 21 counted).
        """
        history = [f"VALUE | Sheet1!A{i} | {i}" for i in range(1, 21)]
        prediction = ["VALUE | Sheet1!A21 | 21"]
        future_gt = ["VALUE | Sheet1!A21 | 21"]

        result = _eval(history=history, prediction=prediction, future_gt=future_gt)
        assert result.metrics.final_state_tp >= 1
        assert result.metrics.final_state_precision == 1.0

    # ---------- Multiple predictions, some correct some not ----------

    def test_three_preds_two_correct_one_wrong(self):
        """
        Prediction: A1=1 (correct), A2=2 (correct), A3=999 (wrong).
        All fresh cells. Precision = 2/3 ≈ 0.667.
        """
        result = _eval(
            history=[],
            prediction=[
                "VALUE | Sheet1!A1 | 1",
                "VALUE | Sheet1!A2 | 2",
                "VALUE | Sheet1!A3 | 999",
            ],
            future_gt=[
                "VALUE | Sheet1!A1 | 1",
                "VALUE | Sheet1!A2 | 2",
                "VALUE | Sheet1!A3 | 3",
            ],
        )
        assert result.metrics.final_state_tp >= 2
        assert result.metrics.final_state_mm >= 1
        assert result.metrics.final_state_precision == pytest.approx(2.0 / 3.0)

    # ---------- FN preserved from broad comparison ----------

    def test_fn_from_future_preserved(self):
        """
        Prediction adds A3=30 (correct). Future also has A4=40 (not predicted).
        FN should still reflect that A4 is missing (from broad comparison).
        """
        result = _eval(
            history=["VALUE | Sheet1!A1 | 10"],
            prediction=["VALUE | Sheet1!A3 | 30"],
            future_gt=["VALUE | Sheet1!A3 | 30", "VALUE | Sheet1!A4 | 40"],
        )
        assert result.metrics.final_state_tp >= 1
        assert result.metrics.final_state_fn >= 1  # A4 missing
        assert result.metrics.final_state_precision == 1.0

    # ---------- Border operations ----------

    def test_border_correct(self):
        """Prediction adds bottom border, target also adds it."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | Data"],
            prediction=["BORDER_BOTTOM | Sheet1!A1 | Thin, Continuous, #000000"],
            future_gt=["BORDER_BOTTOM | Sheet1!A1 | Thin, Continuous, #000000"],
        )
        assert result.metrics.final_state_tp >= 1
        assert result.metrics.final_state_precision == 1.0

    def test_border_wrong_color(self):
        """Prediction adds black border, target wants red border → MM."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | Data"],
            prediction=["BORDER_BOTTOM | Sheet1!A1 | Thin, Continuous, #000000"],
            future_gt=["BORDER_BOTTOM | Sheet1!A1 | Thin, Continuous, #FF0000"],
        )
        assert result.metrics.final_state_mm >= 1
        assert result.metrics.final_state_precision == 0.0

    # ---------- Merge operations ----------

    def test_merge_correct(self):
        """Prediction merges cells, target also merges same range."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | Title"],
            prediction=["MERGE | Sheet1!A1:C1 | true"],
            future_gt=["MERGE | Sheet1!A1:C1 | true"],
        )
        assert result.metrics.final_state_tp >= 1
        assert result.metrics.final_state_precision == 1.0

    # ---------- Number format ----------

    def test_number_format_correct(self):
        """Prediction sets number format matching target."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | 0.5"],
            prediction=["NUMBER_FORMAT | Sheet1!A1 | 0.00%"],
            future_gt=["NUMBER_FORMAT | Sheet1!A1 | 0.00%"],
        )
        assert result.metrics.final_state_tp >= 1
        assert result.metrics.final_state_precision == 1.0

    def test_number_format_wrong(self):
        """Prediction sets wrong number format → MM."""
        result = _eval(
            history=["VALUE | Sheet1!A1 | 0.5"],
            prediction=["NUMBER_FORMAT | Sheet1!A1 | 0.00%"],
            future_gt=["NUMBER_FORMAT | Sheet1!A1 | #,##0"],
        )
        assert result.metrics.final_state_mm >= 1
        assert result.metrics.final_state_precision == 0.0
