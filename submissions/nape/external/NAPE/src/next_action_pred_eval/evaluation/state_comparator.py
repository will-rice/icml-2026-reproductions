"""
State Comparator Module
Compares two workbook states cell-by-cell and property-by-property.
Calculates TP, FP, FN metrics for evaluation.
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.operations import SetBorder
from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.core.defaults import EXCEL_DEFAULTS
from next_action_pred_eval.core.operation_merger import RectangleMerger, BorderMerger
from next_action_pred_eval.utils.cell_utils import get_cell_address, get_range_string
from next_action_pred_eval.utils.workbook.excel_parser import ExcelParser

logger = logging.getLogger(__name__)


# ============= Helpers =============

def _set_nested_value(obj: Dict[str, Any], path: str, value: Any) -> None:
    """Set a nested dict value using a dot-separated path.

    Examples:
        _set_nested_value(cell, "value", 42)
        _set_nested_value(cell, "Format.font.bold", True)
        _set_nested_value(cell, "Format.borders.left", {"lineStyle": "thin"})
    """
    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        elif not isinstance(current[part], dict):
            return
        current = current[part]
    final_key = parts[-1]
    if isinstance(value, dict):
        if final_key not in current or not isinstance(current.get(final_key), dict):
            current[final_key] = {}
        for k, v in value.items():
            current[final_key][k] = v
    else:
        current[final_key] = value


# ============= Border Style Normalization =============

BORDER_STYLE_CANONICAL: Dict[str, Optional[str]] = {
    # Standard cell border style names
    "hair": "hair",
    "thin": "thin",
    "medium": "medium",
    "thick": "thick",
    "dotted": "dotted",
    "dashed": "dashed",
    "dashdot": "dashdot",
    "dashdotdot": "dashdotdot",
    "double": "double",
    "mediumdashed": "mediumdashed",
    "mediumdashdot": "mediumdashdot",
    "mediumdashdotdot": "mediumdashdotdot",
    "slantdashdot": "slantdashdot",
    # Office.js equivalents
    "continuous": "thin",
    "hairline": "hair",
    # None values
    "none": None,
    "": None,
}


@dataclass
class PropertyDifference:
    """Represents a difference in a single property."""

    sheet: str
    """Worksheet name."""

    cell: str
    """Cell address (e.g., 'A1' or 'A1:B2' for merged cells)."""

    property_path: str
    """Property path (e.g., 'value', 'Format.font.bold')."""

    predicted_value: Any
    """Value in predicted state."""

    true_value: Any
    """Value in true/target state."""

    match_type: str
    """Match type: 'TP', 'FP', 'FN', or 'MISMATCH'."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sheet": self.sheet,
            "cell": self.cell,
            "property_path": self.property_path,
            "predicted_value": str(self.predicted_value),
            "true_value": str(self.true_value),
            "match_type": self.match_type,
        }


@dataclass
class ComparisonResult:
    """Result of comparing two states."""

    true_positives: int
    """Properties correctly predicted."""

    false_positives: int
    """Properties in predicted but not in true (pure over-predictions)."""

    false_negatives: int
    """Properties in true but not in predicted (pure under-predictions)."""

    mismatches: int
    """Properties in both states but with different values."""

    ops_diff: int
    """Simplified operation count to go from predicted to true state."""

    total_properties_predicted: int
    """Total properties in predicted state (TP + FP + MM)."""

    total_properties_true: int
    """Total properties in true state (TP + FN + MM)."""

    differences: List[PropertyDifference]
    """Detailed list of all differences."""

    property_type_stats: Dict[str, Dict[str, int]]
    """Statistics by property type: {prop_type: {TP: X, FP: Y, FN: Z, MISMATCH: N}}."""

    inverse_ops_merged: List[Operation] = field(default_factory=list)
    """Operations to cancel FP properties."""

    def precision(self) -> float:
        """Calculate precision: TP / (TP + FP + MM)."""
        total = self.true_positives + self.false_positives + self.mismatches
        return self.true_positives / total if total > 0 else 0.0

    def recall(self) -> float:
        """Calculate recall: TP / (TP + FN + MM)."""
        total = self.true_positives + self.false_negatives + self.mismatches
        return self.true_positives / total if total > 0 else 0.0

    def f1_score(self) -> float:
        """Calculate F1 score."""
        denom = (2 * self.true_positives + self.false_positives
                 + self.false_negatives + 2 * self.mismatches)
        return 2 * self.true_positives / denom if denom > 0 else 0.0


class StateComparator:
    """
    Compares two workbook states and calculates metrics.

    Properties compared per cell:
    - value, formula, number_format
    - Format.font.* (name, size, bold, italic, color, underline)
    - Format.fill.fgColor
    - Format.horizontalAlignment, verticalAlignment
    - Format.textOrientation, wrapText
    - Format.borders.{left,right,top,bottom}.{lineStyle,color}
    - merged_cells (at worksheet level)
    """

    # Property categories
    SIMPLE_PROPERTIES = ["value", "formula", "number_format"]
    FONT_PROPERTIES = ["name", "size", "bold", "italic", "color", "underline"]
    FILL_PROPERTIES = ["fgColor", "patternType"]
    ALIGNMENT_PROPERTIES = ["horizontalAlignment", "verticalAlignment"]
    OTHER_FORMAT_PROPERTIES = ["textOrientation", "wrapText"]
    BORDER_SIDES = ["left", "right", "top", "bottom"]
    BORDER_PROPERTIES = ["lineStyle", "color"]

    def __init__(self, ignore_defaults: bool = True):
        """
        Initialize StateComparator.

        Args:
            ignore_defaults: If True, ignore properties at default values
                           (e.g., don't count missing bold=False as FN).
        """
        self.ignore_defaults = ignore_defaults

    def compare(
        self,
        predicted_state: Dict[str, Any],
        true_state: Dict[str, Any],
        skip_ops_diff: bool = False,
    ) -> ComparisonResult:
        """
        Compare two states and calculate TP, FP, FN.

        Args:
            predicted_state: State from predicted operations.
            true_state: State from ground truth operations.
            skip_ops_diff: If True, skip expensive ops_diff calculation
                (ExcelParser + symmetric_diff + merge) and only compute
                inverse_ops. The caller is responsible for computing
                ops_diff via another mechanism (e.g. _apply_undo_summary).

        Returns:
            ComparisonResult with metrics and detailed differences.
        """
        logger.debug("Starting state comparison")

        differences: List[PropertyDifference] = []

        # Get all sheets from both states
        pred_sheets = set(predicted_state.get("worksheets", {}).keys())
        true_sheets = set(true_state.get("worksheets", {}).keys())
        all_sheets = pred_sheets | true_sheets

        for sheet in all_sheets:
            sheet_diffs = self._compare_sheet(
                sheet,
                predicted_state.get("worksheets", {}).get(sheet),
                true_state.get("worksheets", {}).get(sheet),
            )
            differences.extend(sheet_diffs)

        # Calculate totals (non-overlapping: FP and FN exclude mismatches)
        tp = sum(1 for d in differences if d.match_type == "TP")
        fp = sum(1 for d in differences if d.match_type == "FP")
        fn = sum(1 for d in differences if d.match_type == "FN")
        mm = sum(1 for d in differences if d.match_type == "MISMATCH")

        # Calculate property type statistics
        property_type_stats = self._calculate_property_type_stats(differences)

        # ops_diff + inverse_ops: compute via state→operations→symmetric_diff→merge
        # This matches the reference implementation's _calculate_ops_to_reach_target.
        # When skip_ops_diff=True, only compute inverse_ops (cheap) and set
        # ops_diff=0 as placeholder — the orchestrator overwrites it anyway via
        # _apply_undo_summary.
        if skip_ops_diff:
            inverse_ops_merged = self._convert_fp_to_inverse_ops(differences)
            ops_diff = 0
        else:
            ops_diff, inverse_ops_merged = self._calculate_ops_to_reach_target(
                predicted_state, true_state, differences
            )

        result = ComparisonResult(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            mismatches=mm,
            total_properties_predicted=tp + fp + mm,
            total_properties_true=tp + fn + mm,
            differences=differences,
            property_type_stats=property_type_stats,
            ops_diff=ops_diff,
            inverse_ops_merged=inverse_ops_merged,
        )

        logger.debug(
            f"Comparison complete: TP={tp}, FP={fp}, FN={fn}, "
            f"MM={mm}, inverse_ops={len(inverse_ops_merged)}"
        )
        return result

    def _calculate_ops_to_reach_target(
        self,
        pred_state: Dict[str, Any],
        true_state: Dict[str, Any],
        differences: List["PropertyDifference"],
    ) -> Tuple[int, List[Operation]]:
        """
        Compute ops_diff by parsing both states into operations, taking the
        symmetric difference, and merging adjacent ops into ranges.

        This matches the reference implementation which counts actual
        spreadsheet operations (merged ranges) rather than individual
        property-level differences.

        Returns:
            Tuple of (ops_count, inverse_ops_merged).
        """
        try:
            parser = ExcelParser()
            pred_ops = parser.parse(state=pred_state)
            true_ops = parser.parse(state=true_state)

            # Symmetric difference: ops in one state but not the other
            unique_ops = list(set(true_ops).symmetric_difference(set(pred_ops)))

            # Extract inverse ops for pure FPs
            inverse_ops = self._convert_fp_to_inverse_ops(differences)

            # Add inverse ops before merging
            unique_ops.extend(inverse_ops)

            # Merge: separate borders from non-borders
            border_ops = [op for op in unique_ops if isinstance(op, SetBorder)]
            non_border_ops = [op for op in unique_ops if not isinstance(op, SetBorder)]

            merged: List[Operation] = []
            if non_border_ops:
                merged.extend(
                    RectangleMerger().merge(
                        non_border_ops, row_first=True, merge_inputs=False
                    )
                )
            if border_ops:
                merged.extend(BorderMerger().merge(border_ops))

            # Also merge just inverse ops for returning
            inv_border = [op for op in inverse_ops if isinstance(op, SetBorder)]
            inv_non_border = [op for op in inverse_ops if not isinstance(op, SetBorder)]
            inverse_merged: List[Operation] = []
            if inv_non_border:
                inverse_merged.extend(
                    RectangleMerger().merge(inv_non_border, row_first=True)
                )
            if inv_border:
                inverse_merged.extend(BorderMerger().merge(inv_border))

            return len(merged), inverse_merged

        except Exception as e:
            # Fallback to simple count if parsing/merging fails
            logger.warning("_calculate_ops_to_reach_target failed: %s, falling back to simple count", e)
            fn = sum(1 for d in differences if d.match_type == "FN")
            fp = sum(1 for d in differences if d.match_type == "FP")
            mm = sum(1 for d in differences if d.match_type == "MISMATCH")
            inverse_ops = self._convert_fp_to_inverse_ops(differences)
            return fn + fp + mm, inverse_ops

    def _convert_fp_to_inverse_ops(
        self, differences: List[PropertyDifference]
    ) -> List[Operation]:
        """
        Convert pure FP PropertyDifferences to inverse operations.

        Builds a state from pure FP properties, parses it into operations
        via ExcelParser, then calls get_inverse() on each.
        """
        from next_action_pred_eval.utils.workbook.excel_parser import ExcelParser
        from next_action_pred_eval.core.operation_merger import RectangleMerger, BorderMerger
        from next_action_pred_eval.core.operations import SetBorder

        # Filter to FP diffs (pure over-predictions, no overlap with MISMATCH)
        fp_diffs = [d for d in differences if d.match_type == "FP"]

        if not fp_diffs:
            return []

        # Build state dict from FP properties
        state = {"worksheets": {}}
        for diff in fp_diffs:
            sheet_name = diff.sheet
            cell_addr = diff.cell
            prop_path = diff.property_path
            value = diff.predicted_value

            if sheet_name not in state["worksheets"]:
                state["worksheets"][sheet_name] = {
                    "cells": {},
                    "worksheetProperties": {"merged_cells": []}
                }

            # Handle merged_cells (worksheet-level)
            if prop_path == "merged_cells":
                if isinstance(value, tuple) and len(value) == 4:
                    state["worksheets"][sheet_name]["worksheetProperties"]["merged_cells"].append({
                        "start_row": value[0], "start_col": value[1],
                        "end_row": value[2], "end_col": value[3]
                    })
                continue

            if cell_addr not in state["worksheets"][sheet_name]["cells"]:
                state["worksheets"][sheet_name]["cells"][cell_addr] = {}

            _set_nested_value(
                state["worksheets"][sheet_name]["cells"][cell_addr],
                prop_path,
                value
            )

            # ExcelParser needs patternType for fill colors
            if prop_path == "Format.fill.fgColor":
                _set_nested_value(
                    state["worksheets"][sheet_name]["cells"][cell_addr],
                    "Format.fill.patternType",
                    "solid"
                )

        # Parse state into operations
        try:
            parser = ExcelParser()
            ops = parser.parse(state=state)
        except Exception as e:
            logger.warning(f"Failed to parse pure FP state: {e}")
            return []

        # Get inverse of each
        inverse_ops = []
        for op in ops:
            try:
                inverse_ops.append(op.get_inverse())
            except Exception:
                pass

        # Merge all inverse ops (rectangle merge for non-borders, border merge for borders)
        try:
            border_ops = [op for op in inverse_ops if isinstance(op, SetBorder)]
            non_border_ops = [op for op in inverse_ops if not isinstance(op, SetBorder)]

            if non_border_ops:
                rect_merger = RectangleMerger()
                non_border_ops = rect_merger.merge(non_border_ops, row_first=True)

            if border_ops:
                border_merger = BorderMerger()
                border_ops = border_merger.merge(border_ops)

            inverse_ops = non_border_ops + border_ops
        except Exception as e:
            logger.debug(f"Failed to merge inverse ops: {e}")

        logger.debug(f"Created {len(inverse_ops)} inverse ops from {len(fp_diffs)} FPs")
        return inverse_ops

    def _compare_sheet(
        self,
        sheet_name: str,
        predicted_sheet: Optional[Dict[str, Any]],
        true_sheet: Optional[Dict[str, Any]],
    ) -> List[PropertyDifference]:
        """Compare a single sheet."""
        differences = []

        if predicted_sheet is None and true_sheet is None:
            return differences

        if predicted_sheet is None:
            predicted_sheet = {"cells": {}, "worksheetProperties": {"merged_cells": []}}

        if true_sheet is None:
            true_sheet = {"cells": {}, "worksheetProperties": {"merged_cells": []}}

        # Compare cells
        pred_cells = predicted_sheet.get("cells", {})
        true_cells = true_sheet.get("cells", {})
        all_cells = set(pred_cells.keys()) | set(true_cells.keys())

        for cell in all_cells:
            cell_diffs = self._compare_cell(
                sheet_name,
                cell,
                pred_cells.get(cell),
                true_cells.get(cell),
            )
            differences.extend(cell_diffs)

        # Compare merged cells
        merge_diffs = self._compare_merged_cells(
            sheet_name,
            predicted_sheet.get("worksheetProperties", {}).get("merged_cells", []),
            true_sheet.get("worksheetProperties", {}).get("merged_cells", []),
        )
        differences.extend(merge_diffs)

        return differences

    def _compare_cell(
        self,
        sheet_name: str,
        cell_addr: str,
        predicted_cell: Optional[Dict[str, Any]],
        true_cell: Optional[Dict[str, Any]],
    ) -> List[PropertyDifference]:
        """Compare a single cell."""
        differences = []

        if predicted_cell is None:
            predicted_cell = {}
        if true_cell is None:
            true_cell = {}

        # Simple properties
        for prop in self.SIMPLE_PROPERTIES:
            diffs = self._compare_property(
                sheet_name,
                cell_addr,
                prop,
                predicted_cell.get(prop),
                true_cell.get(prop),
            )
            differences.extend(diffs)

        # Format properties
        pred_format = predicted_cell.get("Format", {})
        true_format = true_cell.get("Format", {})

        # Font properties
        pred_font = pred_format.get("font", {})
        true_font = true_format.get("font", {})
        for prop in self.FONT_PROPERTIES:
            diffs = self._compare_property(
                sheet_name,
                cell_addr,
                f"Format.font.{prop}",
                pred_font.get(prop),
                true_font.get(prop),
            )
            differences.extend(diffs)

        # Fill properties (composite)
        pred_fill = pred_format.get("fill", {})
        true_fill = true_format.get("fill", {})
        fill_diffs = self._compare_composite_property(
            sheet_name,
            cell_addr,
            "Format.fill",
            pred_fill,
            true_fill,
            self.FILL_PROPERTIES,
        )
        differences.extend(fill_diffs)

        # Alignment and other format properties
        for prop in self.ALIGNMENT_PROPERTIES + self.OTHER_FORMAT_PROPERTIES:
            diffs = self._compare_property(
                sheet_name,
                cell_addr,
                f"Format.{prop}",
                pred_format.get(prop),
                true_format.get(prop),
            )
            differences.extend(diffs)

        # Border properties (composite per side)
        pred_borders = pred_format.get("borders", {})
        true_borders = true_format.get("borders", {})
        for side in self.BORDER_SIDES:
            pred_border = pred_borders.get(side, {})
            true_border = true_borders.get(side, {})
            border_diffs = self._compare_composite_property(
                sheet_name,
                cell_addr,
                f"Format.borders.{side}",
                pred_border,
                true_border,
                self.BORDER_PROPERTIES,
            )
            differences.extend(border_diffs)

        return differences

    def _compare_property(
        self,
        sheet: str,
        cell: str,
        property_path: str,
        predicted_value: Any,
        true_value: Any,
    ) -> List[PropertyDifference]:
        """
        Compare a single property.

        Returns list of PropertyDifference (0-3 items depending on match result).
        """
        pred_val = self._normalize_value(predicted_value, property_path)
        true_val = self._normalize_value(true_value, property_path)

        pred_is_default = self._is_none_or_default(pred_val, property_path)
        true_is_default = self._is_none_or_default(true_val, property_path)

        # Both missing/default
        if pred_is_default and true_is_default:
            return []

        # Check if values match
        if self._values_equal(predicted_value, true_value, property_path):
            return [
                PropertyDifference(
                    sheet=sheet,
                    cell=cell,
                    property_path=property_path,
                    predicted_value=predicted_value,
                    true_value=true_value,
                    match_type="TP",
                )
            ]

        # Values differ
        has_pred = not pred_is_default
        has_true = not true_is_default

        if has_pred and has_true:
            # Both exist but differ — MISMATCH
            return [
                PropertyDifference(
                    sheet=sheet,
                    cell=cell,
                    property_path=property_path,
                    predicted_value=predicted_value,
                    true_value=true_value,
                    match_type="MISMATCH",
                ),
            ]
        elif has_pred:
            # Only predicted has value (over-prediction)
            return [
                PropertyDifference(
                    sheet=sheet,
                    cell=cell,
                    property_path=property_path,
                    predicted_value=predicted_value,
                    true_value=true_value,
                    match_type="FP",
                )
            ]
        elif has_true:
            # Only true has value (under-prediction)
            return [
                PropertyDifference(
                    sheet=sheet,
                    cell=cell,
                    property_path=property_path,
                    predicted_value=predicted_value,
                    true_value=true_value,
                    match_type="FN",
                )
            ]

        return []

    def _compare_composite_property(
        self,
        sheet: str,
        cell: str,
        property_path: str,
        predicted_dict: Dict[str, Any],
        true_dict: Dict[str, Any],
        sub_properties: List[str],
    ) -> List[PropertyDifference]:
        """Compare a composite property (e.g., fill, border side)."""
        all_match = True
        has_any_pred = False
        has_any_true = False

        for sub_prop in sub_properties:
            pred_val = predicted_dict.get(sub_prop)
            true_val = true_dict.get(sub_prop)

            full_path = f"{property_path}.{sub_prop}"
            pred_norm = self._normalize_value(pred_val, full_path)
            true_norm = self._normalize_value(true_val, full_path)

            pred_is_default = self._is_none_or_default(pred_norm, full_path)
            true_is_default = self._is_none_or_default(true_norm, full_path)

            if not pred_is_default:
                has_any_pred = True
            if not true_is_default:
                has_any_true = True

            if not self._values_equal(pred_val, true_val, full_path):
                if not (pred_is_default and true_is_default):
                    all_match = False

        if not has_any_pred and not has_any_true:
            return []

        pred_composite = {
            p: predicted_dict.get(p)
            for p in sub_properties
            if predicted_dict.get(p) is not None
        }
        true_composite = {
            p: true_dict.get(p)
            for p in sub_properties
            if true_dict.get(p) is not None
        }

        if all_match:
            return [
                PropertyDifference(
                    sheet=sheet,
                    cell=cell,
                    property_path=property_path,
                    predicted_value=pred_composite if pred_composite else None,
                    true_value=true_composite if true_composite else None,
                    match_type="TP",
                )
            ]

        if has_any_pred and has_any_true:
            # Both have values but differ — MISMATCH
            return [
                PropertyDifference(
                    sheet=sheet,
                    cell=cell,
                    property_path=property_path,
                    predicted_value=pred_composite,
                    true_value=true_composite,
                    match_type="MISMATCH",
                ),
            ]
        elif has_any_pred:
            return [
                PropertyDifference(
                    sheet=sheet,
                    cell=cell,
                    property_path=property_path,
                    predicted_value=pred_composite,
                    true_value=None,
                    match_type="FP",
                )
            ]
        else:
            return [
                PropertyDifference(
                    sheet=sheet,
                    cell=cell,
                    property_path=property_path,
                    predicted_value=None,
                    true_value=true_composite,
                    match_type="FN",
                )
            ]

    def _compare_merged_cells(
        self,
        sheet_name: str,
        predicted_merges: List[Dict[str, int]],
        true_merges: List[Dict[str, int]],
    ) -> List[PropertyDifference]:
        """Compare merged cells."""
        differences = []

        def merge_to_tuple(m: Dict[str, int]) -> Tuple[int, int, int, int]:
            return (m["start_row"], m["start_col"], m["end_row"], m["end_col"])

        def tuple_to_range(t: Tuple[int, int, int, int]) -> str:
            return f"{get_cell_address(t[0], t[1])}:{get_cell_address(t[2], t[3])}"

        pred_set = {merge_to_tuple(m) for m in predicted_merges}
        true_set = {merge_to_tuple(m) for m in true_merges}

        # True Positives
        for merge in pred_set & true_set:
            differences.append(
                PropertyDifference(
                    sheet=sheet_name,
                    cell=tuple_to_range(merge),
                    property_path="merged_cells",
                    predicted_value=merge,
                    true_value=merge,
                    match_type="TP",
                )
            )

        # False Positives
        for merge in pred_set - true_set:
            differences.append(
                PropertyDifference(
                    sheet=sheet_name,
                    cell=tuple_to_range(merge),
                    property_path="merged_cells",
                    predicted_value=merge,
                    true_value=None,
                    match_type="FP",
                )
            )

        # False Negatives
        for merge in true_set - pred_set:
            differences.append(
                PropertyDifference(
                    sheet=sheet_name,
                    cell=tuple_to_range(merge),
                    property_path="merged_cells",
                    predicted_value=None,
                    true_value=merge,
                    match_type="FN",
                )
            )

        return differences

    def _extract_color(self, value: Any) -> Optional[str]:
        """Extract RGB color string from various formats."""
        if value is None:
            return None

        if isinstance(value, str):
            return value.upper() if value.startswith("#") else value

        if isinstance(value, dict):
            if value.get("meta") == "auto":
                return None
            if "rgb" in value:
                rgb = value["rgb"]
                return rgb.upper() if isinstance(rgb, str) and rgb.startswith("#") else rgb
            if "meta" in value and "rgb" not in value:
                return None

        return None

    def _normalize_border_style(self, style: Any) -> Optional[str]:
        """Normalize border line style to canonical form."""
        if style is None:
            return None

        if isinstance(style, str):
            style_lower = style.lower()
            return BORDER_STYLE_CANONICAL.get(style_lower, style_lower)

        return None

    def _normalize_value(self, value: Any, property_path: str) -> Any:
        """Normalize a property value for comparison."""
        if value is None:
            return None

        # Color properties
        color_properties = [
            "Format.font.color",
            "Format.fill.fgColor",
            "Format.fill.bgColor",
        ]
        border_color_props = [
            f"Format.borders.{side}.color" for side in self.BORDER_SIDES
        ]

        if property_path in color_properties or property_path in border_color_props:
            return self._extract_color(value)

        # Border style properties
        border_style_props = [
            f"Format.borders.{side}.lineStyle" for side in self.BORDER_SIDES
        ]
        if property_path in border_style_props:
            return self._normalize_border_style(value)

        return value

    def _is_default_value(self, value: Any, property_path: str) -> bool:
        """Check if a value is the default for its property type."""
        if value is None:
            return True

        # Font color: black is default
        if property_path == "Format.font.color":
            normalized = self._extract_color(value)
            return normalized is None or normalized.upper() == "#000000"

        # Fill color: None or black is default
        if property_path == "Format.fill.fgColor":
            normalized = self._extract_color(value)
            return normalized is None or normalized.upper() == "#000000"

        # Border color: None is default
        border_color_props = [
            f"Format.borders.{side}.color" for side in self.BORDER_SIDES
        ]
        if property_path in border_color_props:
            return self._extract_color(value) is None

        # Border style: None is default
        border_style_props = [
            f"Format.borders.{side}.lineStyle" for side in self.BORDER_SIDES
        ]
        if property_path in border_style_props:
            return value is None or str(value).lower() == "none"

        # Font properties
        if property_path == "Format.font.bold":
            return value is False or value is None
        if property_path == "Format.font.italic":
            return value is False or value is None
        if property_path == "Format.font.underline":
            return value is None or value == "none" or value is False
        if property_path == "Format.font.size":
            return value is None or value == EXCEL_DEFAULTS.get("font_size", 11)
        if property_path == "Format.font.name":
            return value is None or value == EXCEL_DEFAULTS.get("font_name", "Calibri")

        # Alignment defaults
        if property_path == "Format.horizontalAlignment":
            return value is None or str(value).lower() == "general"
        if property_path == "Format.verticalAlignment":
            return value is None or str(value).lower() == "bottom"

        # Other defaults
        if property_path == "Format.wrapText":
            return value is False or value is None
        if property_path == "Format.textOrientation":
            return value is None or value == 0
        if property_path == "Format.fill.patternType":
            return value is None or value == "none"
        if property_path == "number_format":
            return value is None or value == "General"

        # Generic defaults
        if value in [False, 0, "", [], "none", "None"]:
            return True

        return False

    def _is_none_or_default(self, value: Any, property_path: str = "") -> bool:
        """Check if value is None or a default value."""
        if value is None:
            return True

        if self.ignore_defaults and property_path:
            return self._is_default_value(value, property_path)

        if value in [False, 0, "", []]:
            return True

        return False

    def _values_equal(self, val1: Any, val2: Any, property_path: str = "") -> bool:
        """Check if two values are equal after normalization."""
        norm1 = self._normalize_value(val1, property_path)
        norm2 = self._normalize_value(val2, property_path)

        if norm1 is None and norm2 is None:
            return True
        if norm1 is None or norm2 is None:
            if norm1 is None and self._is_default_value(norm2, property_path):
                return True
            if norm2 is None and self._is_default_value(norm1, property_path):
                return True
            return False

        # Numeric comparison
        if isinstance(norm1, (int, float)) and isinstance(norm2, (int, float)):
            return abs(norm1 - norm2) < 1e-9

        # String comparison
        if isinstance(norm1, str) and isinstance(norm2, str):
            if "color" in property_path.lower():
                return norm1.upper() == norm2.upper()
            if "style" in property_path.lower() or "alignment" in property_path.lower():
                return norm1.lower() == norm2.lower()

        return norm1 == norm2

    def _calculate_property_type_stats(
        self, differences: List[PropertyDifference]
    ) -> Dict[str, Dict[str, int]]:
        """Calculate statistics by property type."""
        stats: Dict[str, Dict[str, int]] = {}

        for diff in differences:
            prop_parts = diff.property_path.split(".")
            if len(prop_parts) > 1 and prop_parts[0] == "Format":
                prop_type = f"Format.{prop_parts[1]}"
            else:
                prop_type = prop_parts[0]

            if prop_type not in stats:
                stats[prop_type] = {"TP": 0, "FP": 0, "FN": 0, "MISMATCH": 0}

            stats[prop_type][diff.match_type] += 1

        return stats

    def generate_report(self, result: ComparisonResult) -> str:
        """Generate a human-readable comparison report."""
        lines = [
            "=" * 80,
            "STATE COMPARISON REPORT",
            "=" * 80,
            "",
            f"True Positives (TP):     {result.true_positives:4d}",
            f"False Positives (FP):    {result.false_positives:4d}",
            f"False Negatives (FN):    {result.false_negatives:4d}",
            f"Mismatches (MM):         {result.mismatches:4d}",
            "",
            f"Precision: {result.precision():.2%}",
            f"Recall:    {result.recall():.2%}",
            f"F1 Score:  {result.f1_score():.2%}",
            "",
            "PROPERTY TYPE BREAKDOWN:",
            "-" * 40,
        ]

        for prop_type, stats in sorted(result.property_type_stats.items()):
            lines.append(
                f"  {prop_type}: TP={stats['TP']} FP={stats['FP']} FN={stats['FN']}"
            )

        lines.extend(["", "=" * 80])
        return "\n".join(lines)


__all__ = [
    "StateComparator",
    "ComparisonResult",
    "PropertyDifference",
    "BORDER_STYLE_CANONICAL",
]
