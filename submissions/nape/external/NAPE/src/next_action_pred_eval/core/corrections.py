"""
Correction Generator Module.

Generates inverse and correction operations to transform one state into another.
Used primarily in the evaluation loop to correct prediction divergence.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy

from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.defaults import EXCEL_DEFAULTS
from next_action_pred_eval.core.operations import (
    SetValue, SetFormula, SetInput,
    SetFillColor, SetFontProperty, SetAlignment,
    SetBorder, MergeCells, SetNumberFormat,
    SetWrapText, SetTextOrientation
)

logger = logging.getLogger(__name__)


@dataclass
class PropertyDifference:
    """Represents a difference in a single property between two states."""
    sheet: str
    cell: str
    property_path: str  # e.g., "value", "Format.font.bold", "Format.borders.left.lineStyle"
    predicted_value: Any
    true_value: Any
    match_type: str  # "TP", "FP", "FN", "MISMATCH"

    def __hash__(self):
        return hash((self.sheet, self.cell, self.property_path, self.match_type))


class CorrectionGenerator:
    """
    Generates inverse and correction operations.

    Responsibilities:
    - Generate inverse ops for FPs (properties in predicted but not in target)
    - Synthesize ops for FNs (properties missing from prediction)
    - Handle MISMATCH values (properties with wrong values)
    - Merge correction ops efficiently

    This class provides methods to convert property differences into concrete
    Operation objects that can be applied to transform one state into another.
    """

    def __init__(self):
        """Initialize CorrectionGenerator."""
        # Mapping of property paths to operation generators
        self._operation_generators = {
            'value': self._gen_value_op,
            'formula': self._gen_formula_op,
            'number_format': self._gen_number_format_op,
            'Format.font.bold': lambda s, c, v: self._gen_font_op(s, c, 'bold', v),
            'Format.font.italic': lambda s, c, v: self._gen_font_op(s, c, 'italic', v),
            'Format.font.size': lambda s, c, v: self._gen_font_op(s, c, 'size', v),
            'Format.font.color': lambda s, c, v: self._gen_font_op(s, c, 'color', v),
            'Format.font.name': lambda s, c, v: self._gen_font_op(s, c, 'name', v),
            'Format.font.underline': lambda s, c, v: self._gen_font_op(s, c, 'underline', v),
            'Format.fill.fgColor': self._gen_fill_op,
            'Format.horizontalAlignment': lambda s, c, v: self._gen_alignment_op(s, c, 'horizontal', v),
            'Format.verticalAlignment': lambda s, c, v: self._gen_alignment_op(s, c, 'vertical', v),
            'Format.wrapText': self._gen_wrap_text_op,
            'Format.textOrientation': self._gen_text_orientation_op,
        }

        # Border operations have dynamic paths
        for side in ['left', 'right', 'top', 'bottom']:
            for prop in ['lineStyle', 'color']:
                path = f'Format.borders.{side}.{prop}'
                self._operation_generators[path] = lambda s, c, v, sd=side, p=prop: self._gen_border_op(s, c, sd, p, v)

    def generate_inverse_ops(
        self,
        fp_diffs: List[PropertyDifference],
        merge: bool = True
    ) -> List[Operation]:
        """
        Generate inverse operations for false positives.

        FPs are properties that exist in the predicted state but not in the
        true/target state. Inverse operations reset these to default values.

        Args:
            fp_diffs: List of PropertyDifference objects with match_type="FP"
            merge: If True, attempt to merge adjacent operations

        Returns:
            List of Operation objects that will clear/reset the FP properties
        """
        ops: List[Operation] = []

        for diff in fp_diffs:
            if diff.match_type not in ("FP", "MISMATCH"):
                continue

            op = self._generate_inverse_for_property(diff)
            if op:
                ops.append(op)

        if merge:
            ops = self._merge_operations(ops)

        return ops

    def synthesize_fn_ops(
        self,
        fn_diffs: List[PropertyDifference],
        merge: bool = True
    ) -> List[Operation]:
        """
        Synthesize operations for false negatives.

        FNs are properties that exist in the true/target state but not in the
        predicted state. Synthesized operations will add these properties.

        Args:
            fn_diffs: List of PropertyDifference objects with match_type="FN"
            merge: If True, attempt to merge adjacent operations

        Returns:
            List of Operation objects that will add the missing properties
        """
        ops: List[Operation] = []

        for diff in fn_diffs:
            if diff.match_type not in ("FN", "MISMATCH"):
                continue

            op = self._synthesize_operation_for_property(diff)
            if op:
                ops.append(op)

        if merge:
            ops = self._merge_operations(ops)

        return ops

    def generate_all_corrections(
        self,
        differences: List[PropertyDifference],
        include_fp_inverse: bool = True,
        include_fn_synthesis: bool = True,
        include_mismatch_fixes: bool = True,
        merge: bool = True
    ) -> List[Operation]:
        """
        Generate all correction operations from a list of differences.

        This is the main entry point that handles:
        1. FPs -> inverse operations to clear
        2. FNs -> synthesize operations to add
        3. MISMATCH -> handled as FP (clear) + FN (add correct)

        Args:
            differences: List of PropertyDifference from comparison
            include_fp_inverse: Include inverse ops for FPs
            include_fn_synthesis: Include synthesized ops for FNs
            include_mismatch_fixes: Handle MISMATCH matches
            merge: Merge resulting operations

        Returns:
            List of correction Operation objects
        """
        correction_ops: List[Operation] = []

        # Separate differences by type
        fp_diffs = [d for d in differences if d.match_type in ("FP", "MISMATCH")]
        fn_diffs = [d for d in differences if d.match_type in ("FN", "MISMATCH")]
        mismatch_diffs = [d for d in differences if d.match_type == "MISMATCH"]

        # 1. Handle FPs
        if include_fp_inverse:
            inverse_ops = self.generate_inverse_ops(fp_diffs, merge=False)
            correction_ops.extend(inverse_ops)

        # 2. Handle MISMATCH values
        if include_mismatch_fixes:
            # For MISMATCH, the FN synthesis will overwrite with correct value
            # But for merged_cells, we need explicit UNMERGE first
            merge_fp_diffs = [d for d in fp_diffs if d.property_path == "merged_cells"]
            for diff in merge_fp_diffs:
                if any(
                    inc.sheet == diff.sheet and inc.cell == diff.cell
                    for inc in mismatch_diffs
                ):
                    # This is a mismatched merge, need UNMERGE first
                    unmerge_op = self._gen_unmerge_op(diff)
                    if unmerge_op:
                        correction_ops.append(unmerge_op)

        # 3. Handle FNs (including MISMATCH values which need new values)
        if include_fn_synthesis:
            fn_ops = self.synthesize_fn_ops(fn_diffs, merge=False)
            correction_ops.extend(fn_ops)

        # 4. Merge if requested
        if merge and correction_ops:
            correction_ops = self._merge_operations(correction_ops)

        return correction_ops

    def _generate_inverse_for_property(self, diff: PropertyDifference) -> Optional[Operation]:
        """Generate an inverse operation for a single FP property."""
        cell_range = CellRange(sheet=diff.sheet, range=diff.cell)

        # Handle merged_cells specially
        if diff.property_path == "merged_cells":
            return self._gen_unmerge_op(diff)

        # Get the default value for this property
        default_value = self._get_default_value(diff.property_path)

        # Use the operation generator if available
        generator = self._operation_generators.get(diff.property_path)
        if generator:
            return generator(diff.sheet, diff.cell, default_value)

        logger.warning(f"No inverse generator for property: {diff.property_path}")
        return None

    def _synthesize_operation_for_property(self, diff: PropertyDifference) -> Optional[Operation]:
        """Synthesize an operation to add a missing FN property."""
        cell_range = CellRange(sheet=diff.sheet, range=diff.cell)

        # Handle merged_cells specially
        if diff.property_path == "merged_cells":
            return self._gen_merge_op(diff)

        # Use the true value
        generator = self._operation_generators.get(diff.property_path)
        if generator:
            return generator(diff.sheet, diff.cell, diff.true_value)

        logger.warning(f"No synthesize generator for property: {diff.property_path}")
        return None

    def _get_default_value(self, property_path: str) -> Any:
        """Get the default value for a property path."""
        defaults_map = {
            'value': None,
            'formula': None,
            'number_format': EXCEL_DEFAULTS.get('number_format', 'General'),
            'Format.font.bold': EXCEL_DEFAULTS.get('font_bold', False),
            'Format.font.italic': EXCEL_DEFAULTS.get('font_italic', False),
            'Format.font.size': EXCEL_DEFAULTS.get('font_size', 11),
            'Format.font.color': EXCEL_DEFAULTS.get('font_color', '#000000'),
            'Format.font.name': EXCEL_DEFAULTS.get('font_name', 'Calibri'),
            'Format.font.underline': EXCEL_DEFAULTS.get('font_underline', 'none'),
            'Format.fill.fgColor': None,
            'Format.horizontalAlignment': EXCEL_DEFAULTS.get('horizontal_alignment', 'General'),
            'Format.verticalAlignment': EXCEL_DEFAULTS.get('vertical_alignment', 'Bottom'),
            'Format.wrapText': EXCEL_DEFAULTS.get('wrap_text', False),
            'Format.textOrientation': EXCEL_DEFAULTS.get('text_orientation', 0),
        }

        # Handle border properties
        if 'borders' in property_path:
            return None  # Default is no border

        return defaults_map.get(property_path)

    # ============= Operation Generators =============

    def _gen_value_op(self, sheet: str, cell: str, value: Any) -> Operation:
        return SetValue(
            cell_range=CellRange(sheet=sheet, range=cell),
            value=value,
            is_inverse=(value is None)
        )

    def _gen_formula_op(self, sheet: str, cell: str, value: Any) -> Operation:
        return SetFormula(
            cell_range=CellRange(sheet=sheet, range=cell),
            value=value,
            is_inverse=(value is None)
        )

    def _gen_number_format_op(self, sheet: str, cell: str, value: Any) -> Operation:
        return SetNumberFormat(
            cell_range=CellRange(sheet=sheet, range=cell),
            value=value or 'General',
            is_inverse=(value == 'General')
        )

    def _gen_font_op(self, sheet: str, cell: str, prop: str, value: Any) -> Operation:
        default_key = f'font_{prop}'
        is_inverse = value == EXCEL_DEFAULTS.get(default_key)
        return SetFontProperty(
            cell_range=CellRange(sheet=sheet, range=cell),
            property=prop,
            value=value,
            is_inverse=is_inverse
        )

    def _gen_fill_op(self, sheet: str, cell: str, value: Any) -> Operation:
        return SetFillColor(
            cell_range=CellRange(sheet=sheet, range=cell),
            value=value,
            is_inverse=(value is None)
        )

    def _gen_alignment_op(self, sheet: str, cell: str, align_type: str, value: Any) -> Operation:
        default_key = f'{align_type}_alignment'
        is_inverse = value == EXCEL_DEFAULTS.get(default_key)
        return SetAlignment(
            cell_range=CellRange(sheet=sheet, range=cell),
            alignment_type=align_type,
            value=value,
            is_inverse=is_inverse
        )

    def _gen_wrap_text_op(self, sheet: str, cell: str, value: Any) -> Operation:
        is_inverse = value == EXCEL_DEFAULTS.get('wrap_text', False)
        return SetWrapText(
            cell_range=CellRange(sheet=sheet, range=cell),
            value=value,
            is_inverse=is_inverse
        )

    def _gen_text_orientation_op(self, sheet: str, cell: str, value: Any) -> Operation:
        is_inverse = value == EXCEL_DEFAULTS.get('text_orientation', 0)
        return SetTextOrientation(
            cell_range=CellRange(sheet=sheet, range=cell),
            value=value or 0,
            is_inverse=is_inverse
        )

    def _gen_border_op(self, sheet: str, cell: str, side: str, prop: str, value: Any) -> Operation:
        # For borders, we need to handle style and color together
        # This is simplified - full implementation would need more context
        border_value = {
            'weight': 'Thin' if value else None,
            'style': value if prop == 'lineStyle' else 'Continuous',
            'color': value if prop == 'color' else None
        }
        return SetBorder(
            cell_range=CellRange(sheet=sheet, range=cell),
            side=side,
            value=border_value,
            is_inverse=(value is None)
        )

    def _gen_merge_op(self, diff: PropertyDifference) -> Optional[Operation]:
        """Generate a merge operation from a difference."""
        # The diff.true_value should contain merge info
        if not diff.true_value:
            return None

        merge_range = diff.cell  # Should be the merge range like "A1:B2"
        return MergeCells(
            cell_range=CellRange(sheet=diff.sheet, range=merge_range),
            value=True,
            is_inverse=False
        )

    def _gen_unmerge_op(self, diff: PropertyDifference) -> Optional[Operation]:
        """Generate an unmerge operation from a difference."""
        merge_range = diff.cell
        return MergeCells(
            cell_range=CellRange(sheet=diff.sheet, range=merge_range),
            value=False,
            is_inverse=True
        )

    def _merge_operations(self, ops: List[Operation]) -> List[Operation]:
        """
        Merge operations where possible.

        This is a simplified implementation - for full merging capability,
        use the ExcelToOfficeJS.merge_operations method.
        """
        if not ops:
            return ops

        # Group by operation type and try to merge adjacent cells
        # For now, return as-is - full merging requires more complex logic
        return ops
