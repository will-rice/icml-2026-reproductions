"""
Default Value Filter - Removes operations that set values to Excel defaults

Handles:
- Known Excel defaults (Calibri, size 11, General alignment, etc.)
- Frequency-based default detection (if 95%+ of operations use the same value, it's likely default)
- Smart filtering for alignments and number formats
"""

from typing import List, Dict, Any, Optional, Set, Counter
from collections import Counter, defaultdict

from next_action_pred_eval.generation.sequencing.base import BaseTransformer, SequencingContext
from next_action_pred_eval.core.operation import Operation


# Known Excel default values by operation type and property
KNOWN_DEFAULTS = {
    # Font properties (SetFontProperty)
    "font_name": {"Calibri", "Arial"},  # Common default fonts
    "font_size": {11, 11.0, "11", "11.0", 10, 10.0, "10", "10.0", 12, 12.0, "12", "12.0"},
    "font_color": {"#000000", "#000", "000000", "black", None},
    "font_bold": {False, "false", 0, None},
    "font_italic": {False, "false", 0, None},
    "font_underline": {"none", "None", None, False},

    # Alignment (SetAlignment)
    "horizontal": {"General", "general", None},
    "vertical": {"Bottom", "bottom", None},

    # Number format (SetNumberFormat)
    "number_format": {"General", "general", "@", None},

    # Fill color (SetFillColor) — #FFFFFF not always default (could be intentional)
    "fill_color": {None, "transparent"},

    # Wrap text (SetWrapText)
    "wrap_text": {False, "false", 0, None},

    # Text orientation (SetTextOrientation)
    "text_orientation": {0, "0", None},
}


def is_known_default(op_type: str, property_name: Optional[str], value: Any) -> bool:
    """
    Check if a value is a known Excel default.

    Args:
        op_type: Operation type name (e.g., "SetFontProperty", "SetAlignment")
        property_name: Property name for property-specific ops (e.g., "name" for font,
                       or "horizontal"/"vertical" for alignment)
        value: The value to check

    Returns:
        True if the value is a known default
    """
    # Handle SetFontProperty with specific properties
    if op_type == "SetFontProperty" and property_name:
        prop_map = {
            "name": "font_name",
            "size": "font_size",
            "color": "font_color",
            "bold": "font_bold",
            "italic": "font_italic",
            "underline": "font_underline",
        }
        default_key = prop_map.get(property_name)
        if default_key and default_key in KNOWN_DEFAULTS:
            return value in KNOWN_DEFAULTS[default_key]

    # Handle SetAlignment with alignment_type (horizontal/vertical)
    if op_type == "SetAlignment" and property_name:
        # property_name here is actually alignment_type (horizontal or vertical)
        if property_name in KNOWN_DEFAULTS:
            return value in KNOWN_DEFAULTS[property_name]

    # Handle SetNumberFormat
    if op_type == "SetNumberFormat":
        return value in KNOWN_DEFAULTS.get("number_format", set())

    # Handle SetFillColor
    if op_type == "SetFillColor":
        return value in KNOWN_DEFAULTS.get("fill_color", set())

    # Handle SetWrapText
    if op_type == "SetWrapText":
        return value in KNOWN_DEFAULTS.get("wrap_text", set())

    # Handle SetTextOrientation
    if op_type == "SetTextOrientation":
        return value in KNOWN_DEFAULTS.get("text_orientation", set())

    return False


def get_value_key(op: Operation) -> Optional[str]:
    """
    Get a hashable key representing the operation's value.

    Args:
        op: The operation

    Returns:
        A string key representing the value, or None if not applicable
    """
    try:
        if hasattr(op, 'property'):
            # SetFontProperty
            return f"{op.property}:{op.value}"
        elif hasattr(op, 'alignment_type'):
            # SetAlignment uses alignment_type instead of property
            return f"{op.alignment_type}:{op.value}"
        elif hasattr(op, 'value'):
            return str(op.value)
        return None
    except:
        return None


def _classify_value(value) -> str:
    """Classify a cell value as text, number, or formula."""
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        if value.startswith("="):
            return "formula"
        try:
            float(value)
            return "number"
        except ValueError:
            return "text"
    return "text"


def _build_cell_content_types(operations: List) -> Dict:
    """Build a map of (sheet, row, col) -> content type from value operations."""
    from next_action_pred_eval.core.operations import SetValue, SetInput, SetFormula
    from openpyxl.utils import range_boundaries

    cell_types: Dict = {}

    for op in operations:
        try:
            sheet = op.cell_range.sheet
            min_col, min_row, max_col, max_row = range_boundaries(op.cell_range.range)
        except Exception:
            continue

        if isinstance(op, SetFormula) and not op.is_inverse:
            for r in range(min_row, max_row + 1):
                for c in range(min_col, max_col + 1):
                    cell_types[(sheet, r, c)] = "formula"

        elif isinstance(op, SetValue) and not op.is_inverse:
            vtype = _classify_value(op.value)
            for r in range(min_row, max_row + 1):
                for c in range(min_col, max_col + 1):
                    cell_types[(sheet, r, c)] = vtype

        elif isinstance(op, SetInput) and not op.is_inverse:
            values = op.value
            if isinstance(values, list):
                for ri, row_vals in enumerate(values):
                    if isinstance(row_vals, list):
                        for ci, val in enumerate(row_vals):
                            cell_types[(sheet, min_row + ri, min_col + ci)] = _classify_value(val)

    return cell_types


def _is_alignment_default_for_range(
    op: Operation,
    alignment: str,
    cell_content_types: Dict,
) -> bool:
    """Check if a horizontal alignment is the default for all cells in the range.

    - "left" is default for text and empty cells (no numbers in range)
    - "right" is default for number cells (all cells must be numbers)
    """
    from openpyxl.utils import range_boundaries

    try:
        min_col, min_row, max_col, max_row = range_boundaries(op.cell_range.range)
    except Exception:
        return False

    sheet = op.cell_range.sheet

    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            cell_type = cell_content_types.get((sheet, r, c), "unknown")
            if alignment == "left":
                # Left is default for text/empty/unknown — NOT for numbers
                if cell_type == "number":
                    return False
            elif alignment == "right":
                # Right is default for numbers ONLY
                if cell_type != "number":
                    return False

    return True


class DefaultValueFilter(BaseTransformer):
    """
    Filters out operations that set values to Excel defaults.

    This transformer removes unnecessary operations by detecting:
    1. Known Excel default values (Calibri, General, etc.)
    2. Frequency-based defaults (if 95%+ use the same value, likely default)

    Config:
        enabled: bool - Whether the filter is active
        use_known_defaults: bool - Filter known Excel defaults
        use_frequency_detection: bool - Use frequency-based default detection
        frequency_threshold: float - Threshold for frequency-based detection (default 0.95)
        filter_alignments: bool - Filter alignment operations
        filter_number_formats: bool - Filter number format operations
        filter_fonts: bool - Filter font property operations
        filter_fills: bool - Filter fill color operations
    """

    DEFAULT_CONFIG = {
        "enabled": True,
        "use_known_defaults": True,
        "use_frequency_detection": True,
        "frequency_threshold": 0.95,
        "filter_alignments": True,
        "filter_number_formats": True,
        "filter_fonts": True,
        "filter_fills": True,
    }

    def transform(self, context: SequencingContext) -> SequencingContext:
        if self.can_skip(context):
            return context

        from next_action_pred_eval.core.operations import (
            SetFontProperty, SetAlignment, SetNumberFormat,
            SetFillColor, SetWrapText, SetTextOrientation
        )

        use_known = self.config.get("use_known_defaults", True)
        use_frequency = self.config.get("use_frequency_detection", True)
        freq_threshold = self.config.get("frequency_threshold", 0.95)

        filter_alignments = self.config.get("filter_alignments", True)
        filter_number_formats = self.config.get("filter_number_formats", True)
        filter_fonts = self.config.get("filter_fonts", True)
        filter_fills = self.config.get("filter_fills", True)

        # Determine which operation types to filter
        filterable_types = set()
        if filter_alignments:
            filterable_types.add(SetAlignment)
        if filter_number_formats:
            filterable_types.add(SetNumberFormat)
        if filter_fonts:
            filterable_types.add(SetFontProperty)
        if filter_fills:
            filterable_types.update({SetFillColor, SetWrapText, SetTextOrientation})

        # Separate filterable and non-filterable operations
        filterable_ops = []
        non_filterable_ops = []

        for op in context.operations:
            if type(op) in filterable_types and not op.is_inverse:
                filterable_ops.append(op)
            else:
                non_filterable_ops.append(op)

        if not filterable_ops:
            return context

        # Build cell content types for content-aware alignment filtering
        cell_content_types = {}
        if filter_alignments:
            cell_content_types = _build_cell_content_types(context.operations)

        # Frequency-based default detection
        # Group by operation type (and property for SetFontProperty/SetAlignment)
        frequency_defaults: Dict[str, Set[Any]] = {}

        # Helper to get property name from operation
        def get_prop_name(op):
            """Get property/alignment_type from operation."""
            if hasattr(op, 'property'):
                return op.property
            if hasattr(op, 'alignment_type'):
                return op.alignment_type
            return None

        if use_frequency:
            # Count values by group — restricted to font_size and font_name only.
            # Other properties (bold, italic, fill, etc.) must not be frequency-filtered
            # because parsers often only emit non-default values, making the intentional
            # value appear as the "most frequent" and get incorrectly removed.
            FREQ_ALLOWED_GROUPS = {"SetFontProperty:size", "SetFontProperty:name"}

            value_counts: Dict[str, Counter] = defaultdict(Counter)

            for op in filterable_ops:
                op_type = type(op).__name__
                prop = get_prop_name(op)
                group_key = f"{op_type}:{prop}" if prop else op_type

                if group_key not in FREQ_ALLOWED_GROUPS:
                    continue

                value_key = get_value_key(op)
                if value_key:
                    value_counts[group_key][value_key] += 1

            # Identify frequency-based defaults
            for group_key, counts in value_counts.items():
                total = sum(counts.values())
                if total >= 3:  # Need at least 3 operations to detect frequency
                    most_common_value, most_common_count = counts.most_common(1)[0]
                    if most_common_count / total >= freq_threshold:
                        if group_key not in frequency_defaults:
                            frequency_defaults[group_key] = set()
                        frequency_defaults[group_key].add(most_common_value)

        # Filter operations
        filtered_ops = []
        ops_removed = 0

        for op in filterable_ops:
            op_type = type(op).__name__
            prop = get_prop_name(op)
            value = op.value

            should_remove = False

            # Check known defaults
            if use_known and is_known_default(op_type, prop, value):
                should_remove = True

            # Content-aware alignment: Left is default for text, Right for numbers
            if not should_remove and op_type == "SetAlignment" and prop == "horizontal":
                value_lower = str(value).lower() if value else ""
                if value_lower == "left":
                    should_remove = _is_alignment_default_for_range(op, "left", cell_content_types)
                elif value_lower == "right":
                    should_remove = _is_alignment_default_for_range(op, "right", cell_content_types)

            # Check frequency-based defaults
            if use_frequency and not should_remove:
                group_key = f"{op_type}:{prop}" if prop else op_type
                value_key = get_value_key(op)

                if group_key in frequency_defaults:
                    if value_key in frequency_defaults[group_key]:
                        should_remove = True

            if should_remove:
                ops_removed += 1
            else:
                filtered_ops.append(op)

        # Combine results
        result_ops = non_filterable_ops + filtered_ops

        self.log(
            context,
            f"Default filter: removed {ops_removed} default value operations"
        )

        return context.copy_with_operations(result_ops)
