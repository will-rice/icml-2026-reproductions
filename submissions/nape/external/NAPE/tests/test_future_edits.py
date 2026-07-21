"""
Tests for FutureEditsManager — ground truth updates when predictions are accepted.

Tests cover:
1. Fingerprint building (property types, cell expansion, border side precision)
2. Deduplication (overlapping ops removed, unaffected ops kept)
3. Inverse operations (pure FPs get inverse ops)
4. Merge handling (duplicate merges, format propagation)
5. End-to-end with real trajectory data
"""

import copy
import json
import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional

from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.operations import (
    SetValue, SetFormula, SetInput, SetFillColor, SetFontProperty,
    SetAlignment, SetBorder, MergeCells, SetNumberFormat,
    SetWrapText, SetTextOrientation,
)
from next_action_pred_eval.core.symbolic import symbolic_to_operations, operations_to_symbolic
from next_action_pred_eval.core.state import StateBuilder
from next_action_pred_eval.evaluation.future_edits import (
    FutureEditsManager,
    FutureEditsConfig,
)
from next_action_pred_eval.evaluation.evaluator import StepEvaluator
from next_action_pred_eval.evaluation.state_comparator import StateComparator


# ============================================================================
# Helpers
# ============================================================================

def _cr(sheet: str, range_str: str) -> CellRange:
    """Shorthand for creating CellRange."""
    return CellRange(sheet=sheet, range=range_str)


def _build_eval_result(
    predicted_ops: List[Operation],
    gt_ops: List[Operation],
    initial_state: Optional[Dict] = None,
):
    """Build an EvaluationResult by running the evaluator."""
    evaluator = StepEvaluator()

    # Build initial state
    if initial_state is None:
        initial_state = {"worksheets": {}}

    # Build pred state
    pred_builder = StateBuilder(copy.deepcopy(initial_state))
    pred_builder.apply_operations(predicted_ops)

    # Build true state (initial + all GT ops)
    true_builder = StateBuilder(copy.deepcopy(initial_state))
    true_builder.apply_operations(gt_ops)

    return evaluator.evaluate(
        ground_truth_operations=gt_ops,
        predicted_operations=predicted_ops,
        initial_state_cache=initial_state,
        lookahead_state_cache=true_builder.state,
    )


# ============================================================================
# 1. Fingerprint Building
# ============================================================================

class TestFingerprintBuilding:
    """Test that _build_fingerprint produces correct (sheet, cell, property) tuples."""

    def setup_method(self):
        self.manager = FutureEditsManager()

    def test_set_value_fingerprint(self):
        """SetValue on A1 produces {(Sheet1, A1, value)}."""
        ops = [SetValue.from_symbolic('VALUE | Sheet1!A1 | 42')]
        fp = self.manager._build_fingerprint(ops)
        assert ("Sheet1", "A1", "value") in fp
        assert len(fp) == 1

    def test_set_input_range_fingerprint(self):
        """SetInput on A1:B2 produces fingerprints for all 4 cells."""
        ops = [SetInput.from_symbolic('INPUT | Sheet1!A1:B2 | [[1,2],[3,4]]')]
        fp = self.manager._build_fingerprint(ops)
        assert ("Sheet1", "A1", "value") in fp
        assert ("Sheet1", "B1", "value") in fp
        assert ("Sheet1", "A2", "value") in fp
        assert ("Sheet1", "B2", "value") in fp
        assert len(fp) == 4

    def test_set_font_bold_fingerprint(self):
        """SetFontProperty bold produces Format.font.bold property."""
        ops = [SetFontProperty.from_symbolic('FONT_BOLD | Sheet1!A1 | True')]
        fp = self.manager._build_fingerprint(ops)
        assert ("Sheet1", "A1", "Format.font.bold") in fp

    def test_set_fill_color_fingerprint(self):
        """SetFillColor produces Format.fill.fgColor property."""
        ops = [SetFillColor.from_symbolic('FILL_COLOR | Sheet1!A1 | #FF0000')]
        fp = self.manager._build_fingerprint(ops)
        assert ("Sheet1", "A1", "Format.fill.fgColor") in fp

    def test_set_border_all_fingerprint(self):
        """SetBorder ALL produces all 4 border sides for all cells."""
        ops = [SetBorder.from_symbolic('BORDER_ALL | Sheet1!A1:B2 | Thin, Continuous, #000000')]
        fp = self.manager._build_fingerprint(ops)
        # BORDER_ALL generates left, right, top, bottom for each cell
        for cell in ["A1", "A2", "B1", "B2"]:
            for side in ["left", "right", "top", "bottom"]:
                assert ("Sheet1", cell, f"Format.borders.{side}") in fp

    def test_merge_cells_fingerprint(self):
        """MergeCells produces merged_cells property for all cells."""
        ops = [MergeCells.from_symbolic('MERGE | Sheet1!A1:B2 | true')]
        fp = self.manager._build_fingerprint(ops)
        assert ("Sheet1", "A1", "merged_cells") in fp
        assert ("Sheet1", "B2", "merged_cells") in fp

    def test_number_format_fingerprint(self):
        ops = [SetNumberFormat.from_symbolic('NUMBER_FORMAT | Sheet1!A1 | #,##0.00')]
        fp = self.manager._build_fingerprint(ops)
        assert ("Sheet1", "A1", "number_format") in fp

    def test_alignment_fingerprint(self):
        ops = [SetAlignment.from_symbolic('ALIGN_HORIZONTAL | Sheet1!A1 | center')]
        fp = self.manager._build_fingerprint(ops)
        assert ("Sheet1", "A1", "Format.horizontalAlignment") in fp

    def test_wrap_text_fingerprint(self):
        ops = [SetWrapText.from_symbolic('WRAP_TEXT | Sheet1!A1 | True')]
        fp = self.manager._build_fingerprint(ops)
        assert ("Sheet1", "A1", "Format.wrapText") in fp

    def test_multi_op_fingerprint(self):
        """Multiple operations produce union of fingerprints."""
        ops = [
            SetValue.from_symbolic('VALUE | Sheet1!A1 | 42'),
            SetFontProperty.from_symbolic('FONT_BOLD | Sheet1!A1 | True'),
            SetFillColor.from_symbolic('FILL_COLOR | Sheet1!B1 | #FF0000'),
        ]
        fp = self.manager._build_fingerprint(ops)
        assert ("Sheet1", "A1", "value") in fp
        assert ("Sheet1", "A1", "Format.font.bold") in fp
        assert ("Sheet1", "B1", "Format.fill.fgColor") in fp

    def test_multi_sheet_fingerprint(self):
        """Fingerprints include sheet name."""
        ops = [
            SetValue.from_symbolic('VALUE | Sheet1!A1 | 1'),
            SetValue.from_symbolic('VALUE | Sheet2!A1 | 2'),
        ]
        fp = self.manager._build_fingerprint(ops)
        assert ("Sheet1", "A1", "value") in fp
        assert ("Sheet2", "A1", "value") in fp
        assert len(fp) == 2


# ============================================================================
# 2. Deduplication
# ============================================================================

class TestFingerprintOverlap:
    """Test fingerprint overlap detection (used in v3 deduplication)."""

    def setup_method(self):
        self.manager = FutureEditsManager()

    def test_exact_overlap_detected(self):
        """Operation on same cell+property overlaps with fingerprint."""
        pred_ops = [SetValue.from_symbolic('VALUE | Sheet1!A1 | 42')]
        fp = self.manager._build_fingerprint(pred_ops)

        future_op = SetValue.from_symbolic('VALUE | Sheet1!A1 | 100')
        future_fp = self.manager._build_fingerprint([future_op])

        assert self.manager._fingerprints_overlap(fp, future_fp)

    def test_different_cell_no_overlap(self):
        """Operation on different cell doesn't overlap."""
        pred_ops = [SetValue.from_symbolic('VALUE | Sheet1!A1 | 42')]
        fp = self.manager._build_fingerprint(pred_ops)

        future_op = SetValue.from_symbolic('VALUE | Sheet1!B1 | 100')
        future_fp = self.manager._build_fingerprint([future_op])

        assert not self.manager._fingerprints_overlap(fp, future_fp)

    def test_different_property_no_overlap(self):
        """Operation on same cell but different property doesn't overlap."""
        pred_ops = [SetValue.from_symbolic('VALUE | Sheet1!A1 | 42')]
        fp = self.manager._build_fingerprint(pred_ops)

        future_op = SetFontProperty.from_symbolic('FONT_BOLD | Sheet1!A1 | True')
        future_fp = self.manager._build_fingerprint([future_op])

        assert not self.manager._fingerprints_overlap(fp, future_fp)

    def test_partial_range_overlap(self):
        """Fingerprints with overlapping cell ranges overlap."""
        pred_ops = [SetValue.from_symbolic('VALUE | Sheet1!A1 | 42')]
        fp = self.manager._build_fingerprint(pred_ops)

        future_op = SetInput.from_symbolic('INPUT | Sheet1!A1:B2 | [[1,2],[3,4]]')
        future_fp = self.manager._build_fingerprint([future_op])

        assert self.manager._fingerprints_overlap(fp, future_fp)

    def test_hierarchical_property_overlap(self):
        """Parent property overlaps with child property."""
        # Format.font.bold should match Format.font (hierarchical)
        pred_fp = {("Sheet1", "A1", "Format.font.bold")}
        future_fp = {("Sheet1", "A1", "Format.font")}

        # Check if properties are related (parent-child relationship)
        assert self.manager._properties_are_related("Format.font.bold", "Format.font")
        assert self.manager._properties_are_related("Format.font", "Format.font.bold")


# ============================================================================
# 3. End-to-End apply_future_edits
# ============================================================================

class TestApplyFutureEdits:
    """Test the full apply_future_edits flow."""

    def setup_method(self):
        self.manager = FutureEditsManager()

    def test_perfect_prediction_keeps_future(self):
        """Perfect prediction (matching GT) keeps unaffected future ops."""
        gt_ops = symbolic_to_operations([
            'VALUE | Sheet1!A1 | 42',
            'VALUE | Sheet1!A2 | 100',
            'FONT_BOLD | Sheet1!B1 | True',
        ])

        predicted = symbolic_to_operations(['VALUE | Sheet1!A1 | 42'])

        # Build states
        initial_state = {"worksheets": {}}
        target_builder = StateBuilder()
        target_builder.apply_operations(gt_ops)
        final_target_state = target_builder.state

        # Build eval result
        eval_result = _build_eval_result(
            predicted_ops=predicted,
            gt_ops=gt_ops[1:],  # GT ops at this step (A2, B1)
            initial_state=initial_state,
        )

        new_gt, changes = self.manager.apply_future_edits(
            current_gt=gt_ops,
            start_idx=0,
            end_idx=1,
            predicted_ops=predicted,
            eval_result=eval_result,
            initial_state=initial_state,
            final_target_state=final_target_state,
        )

        # Prediction replaces first op, rest kept
        assert len(new_gt) >= 2  # At least prediction + unaffected ops
        # B1 FONT_BOLD should be kept (different cell+property)
        symbolic = operations_to_symbolic(new_gt)
        assert any('FONT_BOLD' in s and 'B1' in s for s in symbolic)

    def test_wrong_prediction_gets_inverse(self):
        """Wrong prediction generates inverse ops to fix FPs."""
        gt_ops = symbolic_to_operations([
            'VALUE | Sheet1!A1 | 42',
            'VALUE | Sheet1!A2 | 100',
        ])

        # Predict wrong value + extra bold
        predicted = symbolic_to_operations([
            'VALUE | Sheet1!A1 | 999',
            'FONT_BOLD | Sheet1!A1 | True',
        ])

        initial_state = {"worksheets": {}}
        target_builder = StateBuilder()
        target_builder.apply_operations(gt_ops)
        final_target_state = target_builder.state

        eval_result = _build_eval_result(
            predicted_ops=predicted,
            gt_ops=gt_ops[1:],
            initial_state=initial_state,
        )

        new_gt, changes = self.manager.apply_future_edits(
            current_gt=gt_ops,
            start_idx=0,
            end_idx=1,
            predicted_ops=predicted,
            eval_result=eval_result,
            initial_state=initial_state,
            final_target_state=final_target_state,
        )

        # Should have inverse ops to undo the bold FP
        assert changes["new_length"] >= changes["old_length"] or len(changes.get("inverse_ops_added", [])) > 0

    def test_requires_states(self):
        """Raises ValueError if states not provided."""
        gt_ops = symbolic_to_operations(['VALUE | Sheet1!A1 | 42'])
        predicted = symbolic_to_operations(['VALUE | Sheet1!A1 | 42'])
        eval_result = _build_eval_result(predicted, gt_ops)

        with pytest.raises(ValueError, match="initial_state"):
            self.manager.apply_future_edits(
                current_gt=gt_ops,
                start_idx=0,
                end_idx=1,
                predicted_ops=predicted,
                eval_result=eval_result,
                initial_state=None,
                final_target_state=None,
            )


# ============================================================================
# 4. Deduplication with Known Reference Data
# ============================================================================

class TestDeduplicationWithTrajectory:
    """Test dedup using real trajectory data from 0000afae."""

    def setup_method(self):
        self.manager = FutureEditsManager()


# ============================================================================
# 5. Property Type Mapping
# ============================================================================

class TestPropertyTypeMapping:
    """Test _get_property_type returns correct property strings."""

    def setup_method(self):
        self.manager = FutureEditsManager()

    @pytest.mark.parametrize("symbolic,expected_prop", [
        ('VALUE | Sheet1!A1 | 42', 'value'),
        ('FORMULA | Sheet1!A1 | =SUM(B1:B10)', 'formula'),
        ('INPUT | Sheet1!A1 | "hello"', 'value'),
        ('FILL_COLOR | Sheet1!A1 | #FF0000', 'Format.fill.fgColor'),
        ('FONT_BOLD | Sheet1!A1 | True', 'Format.font.bold'),
        ('FONT_SIZE | Sheet1!A1 | 12', 'Format.font.size'),
        ('FONT_COLOR | Sheet1!A1 | #000000', 'Format.font.color'),
        ('BORDER_ALL | Sheet1!A1 | Thin, Continuous, #000000', 'Format.borders'),
        ('ALIGN_HORIZONTAL | Sheet1!A1 | center', 'Format.horizontalAlignment'),
        ('NUMBER_FORMAT | Sheet1!A1 | #,##0.00', 'number_format'),
        ('WRAP_TEXT | Sheet1!A1 | True', 'Format.wrapText'),
        ('MERGE | Sheet1!A1:B2 | true', 'merged_cells'),
    ])
    def test_property_types(self, symbolic, expected_prop):
        ops = symbolic_to_operations([symbolic])
        prop = self.manager._get_property_type(ops[0])
        assert prop == expected_prop


# ============================================================================
# 6. Config
# ============================================================================

class TestFutureEditsConfig:
    """Test configuration options."""

    def test_default_config(self):
        config = FutureEditsConfig()
        assert config.max_inverse_ops == 250
        assert config.track_metadata is True
        assert config.max_fingerprint_cells == 1000

    def test_custom_config(self):
        config = FutureEditsConfig(max_inverse_ops=10, track_metadata=False)
        manager = FutureEditsManager(config=config)
        assert manager.config.max_inverse_ops == 10
        assert manager.config.track_metadata is False
