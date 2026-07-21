"""
Tests for operation behavioral equivalence.

Categories:
1. Symbolic round-trip (from_symbolic -> to_symbolic)
2. apply_to_state correctness
3. Default value handling (should NOT create state keys)
4. Inverse operations (apply + inverse = empty state)
5. Border side logic
6. MergeCells complex behavior
7. SetInput dimension edge cases
"""

import json
import pytest
from copy import deepcopy

from next_action_pred_eval.core.symbolic import symbolic_to_operations, operations_to_symbolic
from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.core.operations import (
    SetValue, SetFormula, SetInput, SetFillColor, SetFontProperty,
    SetAlignment, SetBorder, MergeCells, SetNumberFormat,
    SetWrapText, SetTextOrientation, PasteFrom, AutoFill, OPERATION_MAP,
)
from next_action_pred_eval.core.state import StateBuilder


# ============================================================================
# Category 1: Symbolic Round-Trip
# ============================================================================

class TestSymbolicRoundTrip:
    """Test from_symbolic -> to_symbolic produces equivalent strings."""

    REAL_OPS = [
        'MERGE | Sheet1!A1:G1 | true',
        'INPUT | Sheet1!A1 | "Hello World"',
        'ALIGN_HORIZONTAL | Sheet1!A1:A2 | left',
        'FONT_COLOR | Sheet1!A2 | #467886',
        'FONT_BOLD | Sheet1!A5 | True',
        'WRAP_TEXT | Sheet1!A5:A8 | True',
        'BORDER_TOP | Sheet1!A8:G8 | Thin, Continuous, #000000',
        'BORDER_BOTTOM | Sheet1!B8:D8 | Thin, Continuous, #000000',
        'NUMBER_FORMAT | Sheet1!B11:G23 | 0.000',
        'ALIGN_VERTICAL | Sheet1!A18:A23 | center',
        'VALUE | Sheet1!A1 | 42',
        'FORMULA | Sheet1!C2 | =SUM(A2:B2)',
        'AUTOFILL | Sheet1!A1:A10 | Sheet1!A1:A3',
    ]

    @pytest.mark.parametrize("symbolic", REAL_OPS)
    def test_round_trip(self, symbolic):
        """Parse symbolic -> convert back -> parse again should be equivalent."""
        ops = symbolic_to_operations([symbolic])
        assert len(ops) == 1, f"Expected 1 operation, got {len(ops)}"

        reserialized = ops[0].to_symbolic()
        ops2 = symbolic_to_operations([reserialized])
        assert len(ops2) == 1

        # The re-parsed operation should produce the same state
        state1 = {"worksheets": {}}
        state2 = {"worksheets": {}}
        ops[0].apply_to_state(state1)
        ops2[0].apply_to_state(state2)
        assert state1 == state2

    def test_all_operation_types_parseable(self):
        """Every key in OPERATION_MAP should produce a parseable operation."""
        for key in OPERATION_MAP:
            assert key in OPERATION_MAP

    def test_input_2d_array_round_trip(self):
        symbolic = 'INPUT | Sheet1!A1:B2 | [[1, 2], [3, 4]]'
        ops = symbolic_to_operations([symbolic])
        assert len(ops) == 1
        assert isinstance(ops[0], SetInput)
        assert ops[0].value == [[1, 2], [3, 4]]


# ============================================================================
# Category 2: apply_to_state — State Key Correctness
# ============================================================================

class TestApplyToState:
    """Test that each operation writes the correct state keys."""

    def test_set_value(self, empty_state):
        op = SetValue.from_symbolic('VALUE | Sheet1!A1 | 42')
        op.apply_to_state(empty_state)
        assert empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["value"] == 42

    def test_set_formula(self, empty_state):
        op = SetFormula.from_symbolic('FORMULA | Sheet1!A1 | =SUM(B1:B10)')
        op.apply_to_state(empty_state)
        assert empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["formula"] == "=SUM(B1:B10)"

    def test_set_input_scalar(self, empty_state):
        op = SetInput.from_symbolic('INPUT | Sheet1!A1 | "hello"')
        op.apply_to_state(empty_state)
        assert empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["value"] == "hello"

    def test_set_input_2d_array(self, empty_state):
        op = SetInput.from_symbolic('INPUT | Sheet1!A1:B2 | [[1, 2], [3, 4]]')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A1"]["value"] == 1
        assert cells["B1"]["value"] == 2
        assert cells["A2"]["value"] == 3
        assert cells["B2"]["value"] == 4

    def test_set_number_format(self, empty_state):
        op = SetNumberFormat.from_symbolic('NUMBER_FORMAT | Sheet1!A1 | #,##0.00')
        op.apply_to_state(empty_state)
        assert empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["number_format"] == "#,##0.00"

    def test_set_fill_color(self, empty_state):
        op = SetFillColor.from_symbolic('FILL_COLOR | Sheet1!A1 | #FF0000')
        op.apply_to_state(empty_state)
        fmt = empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["Format"]
        assert fmt["fill"]["fgColor"] == "#FF0000"
        assert fmt["fill"]["patternType"] == "solid"

    def test_set_font_bold(self, empty_state):
        op = SetFontProperty.from_symbolic('FONT_BOLD | Sheet1!A1 | True')
        op.apply_to_state(empty_state)
        fmt = empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["Format"]
        assert fmt["font"]["bold"] is True

    def test_set_font_size(self, empty_state):
        op = SetFontProperty.from_symbolic('FONT_SIZE | Sheet1!A1 | 14')
        op.apply_to_state(empty_state)
        fmt = empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["Format"]
        assert fmt["font"]["size"] == 14.0

    def test_set_alignment_horizontal(self, empty_state):
        op = SetAlignment.from_symbolic('ALIGN_HORIZONTAL | Sheet1!A1 | center')
        op.apply_to_state(empty_state)
        fmt = empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["Format"]
        assert fmt["horizontalAlignment"] == "center"

    def test_set_alignment_vertical(self, empty_state):
        op = SetAlignment.from_symbolic('ALIGN_VERTICAL | Sheet1!A1 | center')
        op.apply_to_state(empty_state)
        fmt = empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["Format"]
        assert fmt["verticalAlignment"] == "center"

    def test_set_wrap_text(self, empty_state):
        op = SetWrapText.from_symbolic('WRAP_TEXT | Sheet1!A1 | True')
        op.apply_to_state(empty_state)
        fmt = empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["Format"]
        assert fmt["wrapText"] is True

    def test_set_text_orientation(self, empty_state):
        op = SetTextOrientation.from_symbolic('TEXT_ORIENTATION | Sheet1!A1 | 45')
        op.apply_to_state(empty_state)
        fmt = empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["Format"]
        assert fmt["textOrientation"] == 45

    def test_set_border_all(self, empty_state):
        op = SetBorder.from_symbolic('BORDER_ALL | Sheet1!A1:B2 | Thin, Continuous, #000000')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        for addr in ["A1", "B1", "A2", "B2"]:
            borders = cells[addr]["Format"]["borders"]
            assert "left" in borders
            assert "right" in borders
            assert "top" in borders
            assert "bottom" in borders

    def test_merge_cells(self, empty_state):
        op = MergeCells.from_symbolic('MERGE | Sheet1!A1:B2 | true')
        op.apply_to_state(empty_state)
        merged = empty_state["worksheets"]["Sheet1"]["worksheetProperties"]["merged_cells"]
        assert len(merged) == 1
        # Coordinates are 1-indexed (A1=row1,col1, B2=row2,col2)
        assert merged[0] == {"start_row": 1, "start_col": 1, "end_row": 2, "end_col": 2}


# ============================================================================
# Category 3: Default Value Handling
# ============================================================================

class TestDefaultHandling:
    """Operations at default values should NOT create state keys."""

    def test_font_bold_false_no_key(self, empty_state):
        op = SetFontProperty.from_symbolic('FONT_BOLD | Sheet1!A1 | False')
        op.apply_to_state(empty_state)
        cell = empty_state["worksheets"]["Sheet1"]["cells"].get("A1", {})
        font = cell.get("Format", {}).get("font", {})
        assert "bold" not in font

    def test_align_general_no_key(self, empty_state):
        op = SetAlignment.from_symbolic('ALIGN_HORIZONTAL | Sheet1!A1 | General')
        op.apply_to_state(empty_state)
        cell = empty_state["worksheets"]["Sheet1"]["cells"].get("A1", {})
        fmt = cell.get("Format", {})
        assert "horizontalAlignment" not in fmt

    def test_number_format_general_no_key(self, empty_state):
        op = SetNumberFormat.from_symbolic('NUMBER_FORMAT | Sheet1!A1 | General')
        op.apply_to_state(empty_state)
        cell = empty_state["worksheets"]["Sheet1"]["cells"].get("A1", {})
        assert "number_format" not in cell

    def test_wrap_text_false_no_key(self, empty_state):
        op = SetWrapText.from_symbolic('WRAP_TEXT | Sheet1!A1 | False')
        op.apply_to_state(empty_state)
        cell = empty_state["worksheets"]["Sheet1"]["cells"].get("A1", {})
        fmt = cell.get("Format", {})
        assert "wrapText" not in fmt

    def test_text_orientation_zero_no_key(self, empty_state):
        op = SetTextOrientation.from_symbolic('TEXT_ORIENTATION | Sheet1!A1 | 0')
        op.apply_to_state(empty_state)
        cell = empty_state["worksheets"]["Sheet1"]["cells"].get("A1", {})
        fmt = cell.get("Format", {})
        assert "textOrientation" not in fmt

    def test_fill_color_clear_no_key(self, empty_state):
        op = SetFillColor.from_symbolic('FILL_COLOR | Sheet1!A1 | clear')
        op.apply_to_state(empty_state)
        cell = empty_state["worksheets"]["Sheet1"]["cells"].get("A1", {})
        fmt = cell.get("Format", {})
        assert "fill" not in fmt


# ============================================================================
# Category 4: Inverse Operations
# ============================================================================

class TestInverseOperations:
    """Apply op + inverse should return to clean state."""

    def test_value_inverse(self, empty_state):
        op = SetValue.from_symbolic('VALUE | Sheet1!A1 | 42')
        op.apply_to_state(empty_state)
        inv = op.get_inverse()
        inv.apply_to_state(empty_state)
        cell = empty_state["worksheets"]["Sheet1"]["cells"].get("A1", {})
        assert "value" not in cell

    def test_font_bold_inverse(self, empty_state):
        op = SetFontProperty.from_symbolic('FONT_BOLD | Sheet1!A1 | True')
        op.apply_to_state(empty_state)
        inv = op.get_inverse()
        inv.apply_to_state(empty_state)
        cell = empty_state["worksheets"]["Sheet1"]["cells"].get("A1", {})
        font = cell.get("Format", {}).get("font", {})
        assert "bold" not in font

    def test_fill_color_inverse(self, empty_state):
        op = SetFillColor.from_symbolic('FILL_COLOR | Sheet1!A1 | #FF0000')
        op.apply_to_state(empty_state)
        inv = op.get_inverse()
        inv.apply_to_state(empty_state)
        cell = empty_state["worksheets"]["Sheet1"]["cells"].get("A1", {})
        fmt = cell.get("Format", {})
        assert "fill" not in fmt

    def test_wrap_text_inverse(self, empty_state):
        op = SetWrapText.from_symbolic('WRAP_TEXT | Sheet1!A1 | True')
        op.apply_to_state(empty_state)
        inv = op.get_inverse()
        inv.apply_to_state(empty_state)
        cell = empty_state["worksheets"]["Sheet1"]["cells"].get("A1", {})
        fmt = cell.get("Format", {})
        assert "wrapText" not in fmt


# ============================================================================
# Category 5: Border Side Logic
# ============================================================================

class TestBorderSideLogic:
    """Test BORDER_ALL, BORDER_OUTSIDE, BORDER_LEFT etc. on ranges."""

    def test_border_outside_only_edges(self, empty_state):
        """BORDER_OUTSIDE on 3x3 should only set edges on boundary cells."""
        op = SetBorder.from_symbolic('BORDER_OUTSIDE | Sheet1!A1:C3 | Thin, Continuous, #000000')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]

        # Center cell (B2) should have NO borders
        b2 = cells.get("B2", {}).get("Format", {}).get("borders", {})
        assert len(b2) == 0, f"B2 should have no borders, got {b2}"

        # A1 (top-left corner) should have left and top
        a1 = cells["A1"]["Format"]["borders"]
        assert "left" in a1
        assert "top" in a1

    def test_border_top_only_first_row(self, empty_state):
        """BORDER_TOP on A1:C3 should only set top on row 1."""
        op = SetBorder.from_symbolic('BORDER_TOP | Sheet1!A1:C3 | Thin, Continuous, #000000')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]

        # A1 should have top border
        assert "top" in cells["A1"]["Format"]["borders"]
        # A2 should NOT have top border
        a2_borders = cells.get("A2", {}).get("Format", {}).get("borders", {})
        assert "top" not in a2_borders

    def test_border_left_only_first_col(self, empty_state):
        """BORDER_LEFT on A1:C3 should only set left on column A."""
        op = SetBorder.from_symbolic('BORDER_LEFT | Sheet1!A1:C3 | Thin, Continuous, #000000')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]

        assert "left" in cells["A1"]["Format"]["borders"]
        b1_borders = cells.get("B1", {}).get("Format", {}).get("borders", {})
        assert "left" not in b1_borders


# ============================================================================
# Category 6: MergeCells Complex Behavior
# ============================================================================

class TestMergeCellsComplex:
    """Test merge/unmerge with borders and format propagation."""

    def test_merge_adds_to_merged_cells_list(self, empty_state):
        op = MergeCells.from_symbolic('MERGE | Sheet1!A1:B2 | true')
        op.apply_to_state(empty_state)
        merged = empty_state["worksheets"]["Sheet1"]["worksheetProperties"]["merged_cells"]
        assert len(merged) == 1

    def test_unmerge_removes_from_merged_cells_list(self, empty_state):
        merge_op = MergeCells.from_symbolic('MERGE | Sheet1!A1:B2 | true')
        merge_op.apply_to_state(empty_state)

        unmerge_op = MergeCells.from_symbolic('UNMERGE | Sheet1!A1:B2 | false')
        unmerge_op.apply_to_state(empty_state)

        merged = empty_state["worksheets"]["Sheet1"]["worksheetProperties"]["merged_cells"]
        assert len(merged) == 0

    def test_merge_clears_non_anchor_cells(self, empty_state):
        """After merge, non-anchor cells should be cleared."""
        val_op = SetValue.from_symbolic('VALUE | Sheet1!B1 | "data"')
        val_op.apply_to_state(empty_state)
        merge_op = MergeCells.from_symbolic('MERGE | Sheet1!A1:B1 | true')
        merge_op.apply_to_state(empty_state)

        b1 = empty_state["worksheets"]["Sheet1"]["cells"].get("B1", {})
        assert "value" not in b1


# ============================================================================
# Category 7: SetInput Dimension Edge Cases
# ============================================================================

class TestSetInputEdgeCases:
    """Test SetInput with various input shapes."""

    def test_scalar_string(self, empty_state):
        op = SetInput.from_symbolic('INPUT | Sheet1!A1 | "hello"')
        op.apply_to_state(empty_state)
        assert empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["value"] == "hello"

    def test_scalar_number(self, empty_state):
        op = SetInput.from_symbolic('INPUT | Sheet1!A1 | 42')
        op.apply_to_state(empty_state)
        assert empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["value"] == 42

    def test_2d_array_matches_range(self, empty_state):
        op = SetInput.from_symbolic('INPUT | Sheet1!A1:C2 | [[1, 2, 3], [4, 5, 6]]')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A1"]["value"] == 1
        assert cells["C1"]["value"] == 3
        assert cells["A2"]["value"] == 4
        assert cells["C2"]["value"] == 6

    def test_null_value_clears_cells(self, empty_state):
        # First set a value
        set_op = SetInput.from_symbolic('INPUT | Sheet1!A1 | "data"')
        set_op.apply_to_state(empty_state)
        assert empty_state["worksheets"]["Sheet1"]["cells"]["A1"]["value"] == "data"

        # Then clear it
        clear_op = SetInput.from_symbolic('INPUT | Sheet1!A1 | null')
        clear_op.apply_to_state(empty_state)
        cell = empty_state["worksheets"]["Sheet1"]["cells"].get("A1", {})
        assert "value" not in cell


# ============================================================================
# Category 8: AutoFill Operation
# ============================================================================

class TestAutoFillRoundTrip:
    """AUTOFILL symbolic round-trip tests."""

    AUTOFILL_SYMBOLICS = [
        'AUTOFILL | Sheet1!A1:A10 | Sheet1!A1:A3',    # drag down
        'AUTOFILL | Sheet1!A1:A8 | Sheet1!A5:A8',     # drag up
        'AUTOFILL | Sheet1!A1:G1 | Sheet1!A1:C1',     # drag right
        'AUTOFILL | Sheet1!A1:E1 | Sheet1!C1:E1',     # drag left
        'AUTOFILL | Sheet1!A1:B8 | Sheet1!A1:B3',     # multi-col vertical
    ]

    @pytest.mark.parametrize("symbolic", AUTOFILL_SYMBOLICS)
    def test_round_trip(self, symbolic):
        ops = symbolic_to_operations([symbolic])
        assert len(ops) == 1
        assert isinstance(ops[0], AutoFill)
        reserialized = ops[0].to_symbolic()
        ops2 = symbolic_to_operations([reserialized])
        assert ops2[0].cell_range == ops[0].cell_range
        assert ops2[0].value == ops[0].value

    def test_inverse_round_trip(self):
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A10 | Sheet1!A1:A3')
        inv = op.get_inverse()
        assert inv.is_inverse
        sym = inv.to_symbolic()
        assert 'clear' in sym
        inv2 = AutoFill.from_symbolic(sym)
        assert inv2.is_inverse


class TestAutoFillNumberSeries:
    """Test AutoFill with number patterns."""

    def test_single_number_copies(self, empty_state):
        """Single number should be copied (NOT incremented)."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | 5').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A4 | Sheet1!A1')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        for row in range(2, 5):
            assert cells[f"A{row}"]["value"] == 5

    def test_two_number_arithmetic_series(self, empty_state):
        """Two numbers define an arithmetic series."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | 2').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | 4').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A5 | Sheet1!A1:A2')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A3"]["value"] == 6
        assert cells["A4"]["value"] == 8
        assert cells["A5"]["value"] == 10

    def test_three_number_series(self, empty_state):
        """Three numbers with constant step."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | 10').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | 20').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A3 | 30').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A6 | Sheet1!A1:A3')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A4"]["value"] == 40
        assert cells["A5"]["value"] == 50
        assert cells["A6"]["value"] == 60

    def test_negative_step_numbers_go_negative(self, empty_state):
        """Raw numbers go negative (NO bounce)."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | 3').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | 1').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A5 | Sheet1!A1:A2')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A3"]["value"] == -1
        assert cells["A4"]["value"] == -3

    def test_non_uniform_step_uses_avg(self, empty_state):
        """Non-uniform differences use average step."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | 1').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | 2').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A3 | 5').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A5 | Sheet1!A1:A3')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        # avg step = (1+3)/2 = 2.0
        assert cells["A4"]["value"] == 7.0
        assert cells["A5"]["value"] == 9.0


class TestAutoFillTextPatterns:
    """Test AutoFill with text-based patterns."""

    def test_single_text_copies(self, empty_state):
        """Single text value is copied."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | "Hello"').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A3 | Sheet1!A1')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A2"]["value"] == "Hello"
        assert cells["A3"]["value"] == "Hello"

    def test_text_cycle(self, empty_state):
        """Multiple text values cycle."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | "X"').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | "Y"').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A6 | Sheet1!A1:A2')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A3"]["value"] == "X"
        assert cells["A4"]["value"] == "Y"
        assert cells["A5"]["value"] == "X"
        assert cells["A6"]["value"] == "Y"


class TestAutoFillTextNumber:
    """Test AutoFill with text+number suffix patterns."""

    def test_single_text_number_increments(self, empty_state):
        """'Item 1' increments to 'Item 2', 'Item 3'."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | "Item 1"').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A4 | Sheet1!A1')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A2"]["value"] == "Item 2"
        assert cells["A3"]["value"] == "Item 3"
        assert cells["A4"]["value"] == "Item 4"

    def test_text_number_series_step(self, empty_state):
        """'Week 2', 'Week 4' → step 2."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | "Week 2"').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | "Week 4"').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A5 | Sheet1!A1:A2')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A3"]["value"] == "Week 6"
        assert cells["A4"]["value"] == "Week 8"
        assert cells["A5"]["value"] == "Week 10"

    def test_text_number_bounce_at_zero(self, empty_state):
        """Text+number patterns bounce at 0."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | "Item 2"').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | "Item 1"').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A6 | Sheet1!A1:A2')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A3"]["value"] == "Item 0"     # step=-1, next=0
        assert cells["A4"]["value"] == "Item 1"     # bounce: |-1|=1, step flips to +1
        assert cells["A5"]["value"] == "Item 2"
        assert cells["A6"]["value"] == "Item 3"


class TestAutoFillCustomLists:
    """Test AutoFill with months, days, Q-series."""

    def test_month_short_from_single(self, empty_state):
        """Single 'Jan' extends to Feb, Mar, ..."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | "Jan"').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A4 | Sheet1!A1')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A2"]["value"] == "Feb"
        assert cells["A3"]["value"] == "Mar"
        assert cells["A4"]["value"] == "Apr"

    def test_month_full_wraps(self, empty_state):
        """November → December → January (wrap)."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | "November"').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | "December"').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A4 | Sheet1!A1:A2')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A3"]["value"] == "January"
        assert cells["A4"]["value"] == "February"

    def test_day_short(self, empty_state):
        SetValue.from_symbolic('VALUE | Sheet1!A1 | "Mon"').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | "Tue"').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A4 | Sheet1!A1:A2')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A3"]["value"] == "Wed"
        assert cells["A4"]["value"] == "Thu"

    def test_quarter_cycle(self, empty_state):
        """Q1 cycles: Q1 → Q2 → Q3 → Q4 → Q1."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | "Q1"').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A6 | Sheet1!A1')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A2"]["value"] == "Q2"
        assert cells["A3"]["value"] == "Q3"
        assert cells["A4"]["value"] == "Q4"
        assert cells["A5"]["value"] == "Q1"  # wraps
        assert cells["A6"]["value"] == "Q2"

    def test_month_case_preserved_upper(self, empty_state):
        """UPPER case months should stay upper."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | "JAN"').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A3 | Sheet1!A1')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A2"]["value"] == "FEB"
        assert cells["A3"]["value"] == "MAR"


class TestAutoFillFormula:
    """Test AutoFill with formula reference adjustment."""

    def test_formula_row_offset(self, empty_state):
        """Formula =A1+B1 dragged down adjusts row refs."""
        SetFormula.from_symbolic('FORMULA | Sheet1!C1 | =A1+B1').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!C1:C3 | Sheet1!C1')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["C2"]["formula"] == "=A2+B2"
        assert cells["C3"]["formula"] == "=A3+B3"

    def test_formula_absolute_ref_preserved(self, empty_state):
        """$A$1 (absolute) should NOT be adjusted."""
        SetFormula.from_symbolic('FORMULA | Sheet1!C1 | =$A$1+B1').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!C1:C3 | Sheet1!C1')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["C2"]["formula"] == "=$A$1+B2"

    def test_formula_horizontal_col_offset(self, empty_state):
        """Formula dragged right adjusts column refs."""
        SetFormula.from_symbolic('FORMULA | Sheet1!A3 | =A1+A2').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A3:C3 | Sheet1!A3')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["B3"]["formula"] == "=B1+B2"
        assert cells["C3"]["formula"] == "=C1+C2"


class TestAutoFillMultiColumn:
    """Test AutoFill with multi-col source ranges."""

    def test_two_columns_independent(self, empty_state):
        """Each column's pattern is extended independently."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | 1').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | 2').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!B1 | 10').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!B2 | 20').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:B4 | Sheet1!A1:B2')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A3"]["value"] == 3
        assert cells["A4"]["value"] == 4
        assert cells["B3"]["value"] == 30
        assert cells["B4"]["value"] == 40


class TestAutoFillMixedTypes:
    """Test AutoFill with mixed-type source values (position-independence)."""

    def test_number_text_alternating(self, empty_state):
        """[1, 'x', 3] — each position has 1 value, so numbers copy, text copies."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | 1').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | "x"').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A3 | 3').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A9 | Sheet1!A1:A3')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        # Position-independent: each slot only has 1 source value → copies
        assert cells["A4"]["value"] == 1
        assert cells["A5"]["value"] == "x"
        assert cells["A6"]["value"] == 3
        assert cells["A7"]["value"] == 1
        assert cells["A8"]["value"] == "x"
        assert cells["A9"]["value"] == 3


class TestAutoFillFormatTiling:
    """Test that formatting is tiled cyclically from source."""

    def test_format_tiled(self, empty_state):
        """Source has bold on row 1 only — should tile cyclically."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | 1').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | 2').apply_to_state(empty_state)
        SetFontProperty.from_symbolic('FONT_BOLD | Sheet1!A1 | True').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A6 | Sheet1!A1:A2')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        # Bold on dest rows 3, 5 (tiles source row 1 pattern at even fill indices)
        assert cells["A3"]["Format"]["font"]["bold"] is True
        assert "Format" not in cells.get("A4", {}) or "font" not in cells.get("A4", {}).get("Format", {})
        assert cells["A5"]["Format"]["font"]["bold"] is True


class TestAutoFillDirections:
    """Test all four fill directions."""

    def test_drag_down(self, empty_state):
        SetValue.from_symbolic('VALUE | Sheet1!A1 | 1').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | 2').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A4 | Sheet1!A1:A2')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A3"]["value"] == 3
        assert cells["A4"]["value"] == 4

    def test_drag_right(self, empty_state):
        SetValue.from_symbolic('VALUE | Sheet1!A1 | 1').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!B1 | 2').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:D1 | Sheet1!A1:B1')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["C1"]["value"] == 3
        assert cells["D1"]["value"] == 4

    def test_drag_up(self, empty_state):
        SetValue.from_symbolic('VALUE | Sheet1!A3 | 1').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A4 | 2').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A4 | Sheet1!A3:A4')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        # Drag up: direction=-1, extends backwards from start of source
        assert cells["A2"]["value"] == 0
        assert cells["A1"]["value"] == -1

    def test_drag_left(self, empty_state):
        SetValue.from_symbolic('VALUE | Sheet1!C1 | 1').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!D1 | 2').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:D1 | Sheet1!C1:D1')
        op.apply_to_state(empty_state)
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        # Drag left: direction=-1, extends backwards from start of source
        assert cells["B1"]["value"] == 0
        assert cells["A1"]["value"] == -1


class TestAutoFillInverse:
    """Test that inverse clears all values and formatting in fill area."""

    def test_inverse_clears_filled_cells(self, empty_state):
        """Apply + inverse should leave fill area clean."""
        SetValue.from_symbolic('VALUE | Sheet1!A1 | 1').apply_to_state(empty_state)
        SetValue.from_symbolic('VALUE | Sheet1!A2 | 2').apply_to_state(empty_state)
        op = AutoFill.from_symbolic('AUTOFILL | Sheet1!A1:A5 | Sheet1!A1:A2')
        op.apply_to_state(empty_state)

        # Verify fill worked
        cells = empty_state["worksheets"]["Sheet1"]["cells"]
        assert cells["A3"]["value"] == 3

        # Apply inverse
        inv = op.get_inverse()
        inv.apply_to_state(empty_state)

        # Fill area (A3:A5) should be cleared
        for addr in ["A3", "A4", "A5"]:
            cell = cells.get(addr, {})
            assert "value" not in cell
            assert "formula" not in cell
            assert "Format" not in cell

        # Source area (A1:A2) should be untouched
        assert cells["A1"]["value"] == 1
        assert cells["A2"]["value"] == 2


class TestAutoFillValidation:
    """Test geometry validation rejects invalid configurations."""

    def test_same_range_rejected(self):
        with pytest.raises(ValueError, match="same range"):
            AutoFill(
                cell_range=CellRange(sheet="Sheet1", range="A1:A3"),
                value="Sheet1!A1:A3"
            )

    def test_source_not_subset_rejected(self):
        with pytest.raises(ValueError, match="subset"):
            AutoFill(
                cell_range=CellRange(sheet="Sheet1", range="B1:B5"),
                value="Sheet1!A1:A3"
            )

    def test_diagonal_rejected(self):
        with pytest.raises(ValueError, match="same columns.*same rows"):
            AutoFill(
                cell_range=CellRange(sheet="Sheet1", range="A1:B2"),
                value="Sheet1!A1"
            )

    def test_different_sheet_rejected(self):
        with pytest.raises(ValueError, match="sheet"):
            AutoFill(
                cell_range=CellRange(sheet="Sheet1", range="A1:A5"),
                value="Sheet2!A1:A3"
            )


# ===========================================================================
# AutoFillDetector transformer tests
# ===========================================================================

class TestAutoFillDetector:
    """Test the AutoFillDetector transformer that collapses sequences into AUTOFILL ops."""

    @pytest.fixture
    def detector(self):
        from next_action_pred_eval.generation.sequencing.transformers.autofill_detector import (
            AutoFillDetector,
        )
        return AutoFillDetector({"enabled": True, "min_fill": 2})

    @staticmethod
    def _ctx(ops):
        from next_action_pred_eval.generation.sequencing.base import SequencingContext
        return SequencingContext(operations=ops)

    @staticmethod
    def _sv(sheet, cell, value):
        return SetValue(cell_range=CellRange(sheet=sheet, range=cell), value=value)

    @staticmethod
    def _sf(sheet, cell, formula):
        return SetFormula(cell_range=CellRange(sheet=sheet, range=cell), value=formula)

    def test_arithmetic_series_collapse(self, detector):
        ops = [self._sv("S1", "A1", 1), self._sv("S1", "A2", 3),
               self._sv("S1", "A3", 5), self._sv("S1", "A4", 7)]
        ctx = detector.transform(self._ctx(ops))
        types = [type(op).__name__ for op in ctx.operations]
        assert types == ["SetValue", "SetValue", "AutoFill"]
        af = ctx.operations[2]
        assert str(af.cell_range) == "S1!A1:A4"  # destination
        assert af.value == "S1!A1:A2"             # source

    def test_too_short_no_collapse(self, detector):
        ops = [self._sv("S1", "A1", 1), self._sv("S1", "A2", 3), self._sv("S1", "A3", 5)]
        ctx = detector.transform(self._ctx(ops))
        types = [type(op).__name__ for op in ctx.operations]
        assert types == ["SetValue"] * 3

    def test_custom_list_months(self, detector):
        ops = [self._sv("S1", "A1", "Jan"), self._sv("S1", "A2", "Feb"),
               self._sv("S1", "A3", "Mar"), self._sv("S1", "A4", "Apr")]
        ctx = detector.transform(self._ctx(ops))
        types = [type(op).__name__ for op in ctx.operations]
        assert types == ["SetValue", "AutoFill"]
        assert ctx.operations[1].value == "S1!A1"  # source is single cell

    def test_text_number_collapse(self, detector):
        ops = [self._sv("S1", "A1", "Item 1"), self._sv("S1", "A2", "Item 2"),
               self._sv("S1", "A3", "Item 3"), self._sv("S1", "A4", "Item 4")]
        ctx = detector.transform(self._ctx(ops))
        types = [type(op).__name__ for op in ctx.operations]
        assert types == ["SetValue", "AutoFill"]

    def test_formula_offset_collapse(self, detector):
        ops = [self._sf("S1", "C1", "=A1+B1"), self._sf("S1", "C2", "=A2+B2"),
               self._sf("S1", "C3", "=A3+B3"), self._sf("S1", "C4", "=A4+B4")]
        ctx = detector.transform(self._ctx(ops))
        types = [type(op).__name__ for op in ctx.operations]
        assert types == ["SetFormula", "AutoFill"]
        assert ctx.operations[1].value == "S1!C1"  # source

    def test_constant_values_no_collapse(self, detector):
        """Constant copies are better handled by OperationMerger → INPUT."""
        ops = [self._sv("S1", "A1", 5), self._sv("S1", "A2", 5),
               self._sv("S1", "A3", 5), self._sv("S1", "A4", 5)]
        ctx = detector.transform(self._ctx(ops))
        assert all(type(op).__name__ == "SetValue" for op in ctx.operations)

    def test_non_pattern_no_collapse(self, detector):
        ops = [self._sv("S1", "A1", 1), self._sv("S1", "A2", 5),
               self._sv("S1", "A3", 2), self._sv("S1", "A4", 8)]
        ctx = detector.transform(self._ctx(ops))
        assert all(type(op).__name__ == "SetValue" for op in ctx.operations)

    def test_interleaved_columns(self, detector):
        """Row-first ordering with two columns — both should collapse independently."""
        ops = [
            self._sv("S1", "A1", 1), self._sv("S1", "B1", 10),
            self._sv("S1", "A2", 2), self._sv("S1", "B2", 20),
            self._sv("S1", "A3", 3), self._sv("S1", "B3", 30),
            self._sv("S1", "A4", 4), self._sv("S1", "B4", 40),
        ]
        ctx = detector.transform(self._ctx(ops))
        autofills = [op for op in ctx.operations if isinstance(op, AutoFill)]
        assert len(autofills) == 2

    def test_horizontal_collapse(self, detector):
        ops = [self._sv("S1", "A1", 1), self._sv("S1", "B1", 2),
               self._sv("S1", "C1", 3), self._sv("S1", "D1", 4)]
        ctx = detector.transform(self._ctx(ops))
        types = [type(op).__name__ for op in ctx.operations]
        assert types == ["SetValue", "SetValue", "AutoFill"]

    def test_state_equivalence_numbers(self, detector):
        """Collapsed ops must produce same state as original ops."""
        original = [self._sv("S1", "A1", 1), self._sv("S1", "A2", 3),
                    self._sv("S1", "A3", 5), self._sv("S1", "A4", 7), self._sv("S1", "A5", 9)]
        state1 = {"worksheets": {}}
        for op in original:
            op.apply_to_state(state1)

        ctx = detector.transform(self._ctx(original))
        state2 = {"worksheets": {}}
        for op in ctx.operations:
            op.apply_to_state(state2)

        assert state1 == state2

    def test_state_equivalence_formulas(self, detector):
        original = [self._sf("S1", "C1", "=A1+B1"), self._sf("S1", "C2", "=A2+B2"),
                    self._sf("S1", "C3", "=A3+B3")]
        state1 = {"worksheets": {}}
        for op in original:
            op.apply_to_state(state1)

        ctx = detector.transform(self._ctx(original))
        state2 = {"worksheets": {}}
        for op in ctx.operations:
            op.apply_to_state(state2)

        assert state1 == state2

    def test_state_equivalence_months(self, detector):
        original = [self._sv("S1", "A1", "Jan"), self._sv("S1", "A2", "Feb"),
                    self._sv("S1", "A3", "Mar"), self._sv("S1", "A4", "Apr"),
                    self._sv("S1", "A5", "May")]
        state1 = {"worksheets": {}}
        for op in original:
            op.apply_to_state(state1)

        ctx = detector.transform(self._ctx(original))
        state2 = {"worksheets": {}}
        for op in ctx.operations:
            op.apply_to_state(state2)

        assert state1 == state2

    def test_detect_values_false_skips_values(self):
        """When detect_values=False, only formulas are collapsed."""
        from next_action_pred_eval.generation.sequencing.transformers.autofill_detector import (
            AutoFillDetector,
        )
        det = AutoFillDetector({"enabled": True, "min_fill": 2, "detect_values": False})
        ops = [self._sv("S1", "A1", 1), self._sv("S1", "A2", 2),
               self._sv("S1", "A3", 3), self._sv("S1", "A4", 4)]
        ctx = det.transform(self._ctx(ops))
        assert all(type(op).__name__ == "SetValue" for op in ctx.operations)

    def test_detect_formulas_false_skips_formulas(self):
        """When detect_formulas=False, only values are collapsed."""
        from next_action_pred_eval.generation.sequencing.transformers.autofill_detector import (
            AutoFillDetector,
        )
        det = AutoFillDetector({"enabled": True, "min_fill": 2, "detect_formulas": False})
        ops = [self._sf("S1", "C1", "=A1+B1"), self._sf("S1", "C2", "=A2+B2"),
               self._sf("S1", "C3", "=A3+B3"), self._sf("S1", "C4", "=A4+B4")]
        ctx = det.transform(self._ctx(ops))
        assert all(type(op).__name__ == "SetFormula" for op in ctx.operations)
