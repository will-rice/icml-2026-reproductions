"""
Input Normalizer Module for Sequence Refinement

This module provides compression and decompression utilities for large INPUT operations
to reduce token usage when sending operations to LLMs for refinement.

Compression Strategy:
---------------------
For large INPUT operations (those exceeding certain thresholds), we compress them by:

1. **Range-based compression**: For large 2D arrays, we show only corner samples:
   - Top-left cell value
   - Top-right cell value
   - Bottom-left cell value
   - Bottom-right cell value
   - Along with metadata about the full dimensions

2. **String truncation**: For very long string values, truncate to first/last N chars

3. **Placeholder format**: Compressed INPUTs use a special format that:
   - Is clearly marked as compressed (with <<COMPRESSED>> marker)
   - Contains enough info for the LLM to understand the operation's purpose
   - Can be losslessly decompressed using the final state

Decompression:
--------------
When decompressing, we use the final workbook state to reconstruct the full values:
1. Parse the compressed format to extract the range
2. Look up actual cell values from the final state
3. Reconstruct the full INPUT operation

The key insight is that the LLM only needs to understand WHAT the operation does
(fill a range with data) and its SCOPE (which cells), not necessarily every single value.
For validation, we reconstruct from the final state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from openpyxl.utils import range_boundaries, get_column_letter

from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.utils.cell_utils import get_cell_address


# Configuration thresholds for compression
LARGE_RANGE_CELL_THRESHOLD = 20  # Compress if range has more than this many cells
LARGE_STRING_CHAR_THRESHOLD = 100  # Truncate strings longer than this
STRING_PREVIEW_LENGTH = 30  # Show first/last N chars when truncating
COMPRESSED_MARKER = "<<COMPRESSED>>"
COMPRESSED_INPUT_PATTERN = re.compile(
    r'^INPUT\s*\|\s*([^|]+)\s*\|\s*<<COMPRESSED>>\s*(.*)$'
)


@dataclass
class CompressionStats:
    """Statistics about compression operations performed."""
    total_operations: int = 0
    compressed_operations: int = 0
    original_char_count: int = 0
    compressed_char_count: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_char_count == 0:
            return 1.0
        return self.compressed_char_count / self.original_char_count

    @property
    def savings_percent(self) -> float:
        return (1 - self.compression_ratio) * 100


def _get_range_dimensions(range_str: str) -> Tuple[int, int, int, int, int, int]:
    """
    Parse a range string and return (min_col, min_row, max_col, max_row, num_rows, num_cols).
    """
    # Handle sheet prefix
    if "!" in range_str:
        _, range_part = range_str.split("!", 1)
    else:
        range_part = range_str

    min_col, min_row, max_col, max_row = range_boundaries(range_part)
    num_rows = max_row - min_row + 1
    num_cols = max_col - min_col + 1
    return min_col, min_row, max_col, max_row, num_rows, num_cols


def _truncate_string(s: str, max_len: int = LARGE_STRING_CHAR_THRESHOLD,
                     preview_len: int = STRING_PREVIEW_LENGTH) -> str:
    """Truncate a long string, showing first and last preview_len characters."""
    if len(s) <= max_len:
        return s
    return f"{s[:preview_len]}...({len(s)} chars)...{s[-preview_len:]}"


def _format_value_preview(value: Any, max_str_len: int = LARGE_STRING_CHAR_THRESHOLD) -> str:
    """Format a value for preview, truncating long strings."""
    if value is None:
        return "null"
    if isinstance(value, str):
        truncated = _truncate_string(value, max_str_len)
        return json.dumps(truncated)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # For floats, limit precision in display
        if isinstance(value, float):
            if value == int(value):
                return str(int(value))
            return f"{value:.6g}"
        return str(value)
    return json.dumps(value)


def _get_corner_values(data_2d: List[List[Any]]) -> Dict[str, Any]:
    """
    Extract corner values from a 2D array.
    Returns dict with top_left, top_right, bottom_left, bottom_right values.
    """
    if not data_2d or not data_2d[0]:
        return {}

    num_rows = len(data_2d)
    num_cols = len(data_2d[0])

    corners = {
        "top_left": data_2d[0][0],
        "rows": num_rows,
        "cols": num_cols,
    }

    if num_cols > 1:
        corners["top_right"] = data_2d[0][-1]
    if num_rows > 1:
        corners["bottom_left"] = data_2d[-1][0]
    if num_rows > 1 and num_cols > 1:
        corners["bottom_right"] = data_2d[-1][-1]

    return corners


def _count_non_empty_cells(data_2d: List[List[Any]]) -> int:
    """Count non-empty cells in a 2D array."""
    count = 0
    for row in data_2d:
        for cell in row:
            if cell is not None and cell != "":
                count += 1
    return count


def _build_compressed_summary(range_str: str, data_2d: List[List[Any]],
                               cell_range: CellRange) -> str:
    """
    Build a compressed summary string for a large INPUT operation.

    Format: <<COMPRESSED>> {rows}x{cols} [TL:"val", TR:"val", BL:"val", BR:"val"] ({non_empty}/{total} filled)
    """
    corners = _get_corner_values(data_2d)
    num_rows = corners.get("rows", 0)
    num_cols = corners.get("cols", 0)
    total_cells = num_rows * num_cols
    non_empty = _count_non_empty_cells(data_2d)

    # Build corner preview strings
    corner_parts = []
    if "top_left" in corners:
        corner_parts.append(f"TL:{_format_value_preview(corners['top_left'], 50)}")
    if "top_right" in corners:
        corner_parts.append(f"TR:{_format_value_preview(corners['top_right'], 50)}")
    if "bottom_left" in corners:
        corner_parts.append(f"BL:{_format_value_preview(corners['bottom_left'], 50)}")
    if "bottom_right" in corners:
        corner_parts.append(f"BR:{_format_value_preview(corners['bottom_right'], 50)}")

    corners_str = ", ".join(corner_parts) if corner_parts else ""

    return f"{COMPRESSED_MARKER} {num_rows}x{num_cols} [{corners_str}] ({non_empty}/{total_cells} filled)"


def compress_input_operation(symbolic_op: str,
                              cell_threshold: int = LARGE_RANGE_CELL_THRESHOLD,
                              string_threshold: int = LARGE_STRING_CHAR_THRESHOLD) -> Tuple[str, bool]:
    """
    Compress an INPUT operation if it exceeds the size thresholds.

    Args:
        symbolic_op: The symbolic operation string (e.g., "INPUT | Sheet1!A1:Z100 | [[...]]")
        cell_threshold: Compress if the range has more cells than this
        string_threshold: Truncate individual strings longer than this

    Returns:
        Tuple of (compressed_or_original_op, was_compressed)
    """
    # Only process INPUT operations
    if not symbolic_op.strip().startswith("INPUT"):
        return symbolic_op, False

    parts = symbolic_op.split(" | ", 2)
    if len(parts) < 3:
        return symbolic_op, False

    op_type, range_str, value_str = parts[0].strip(), parts[1].strip(), parts[2].strip()

    # Skip clear (inverse) operations
    if value_str == "clear":
        return symbolic_op, False

    # Parse the range to get dimensions
    try:
        cell_range = CellRange.from_string(range_str)
        _, _, _, _, num_rows, num_cols = _get_range_dimensions(range_str)
        total_cells = num_rows * num_cols
    except Exception:
        return symbolic_op, False

    # Check if compression is needed
    if total_cells <= cell_threshold:
        # Small range - no compression needed, but might truncate long strings
        try:
            value = json.loads(value_str)
            if isinstance(value, str) and len(value) > string_threshold:
                truncated = _truncate_string(value, string_threshold)
                return f"{op_type} | {range_str} | {json.dumps(truncated)}", True
        except json.JSONDecodeError:
            pass
        return symbolic_op, False

    # Large range - apply compression
    try:
        value = json.loads(value_str)
    except json.JSONDecodeError:
        # If we can't parse it, just return original
        return symbolic_op, False

    # Handle 2D arrays
    if isinstance(value, list) and value and isinstance(value[0], list):
        compressed_summary = _build_compressed_summary(range_str, value, cell_range)
        return f"INPUT | {range_str} | {compressed_summary}", True

    # Handle 1D arrays (converted to 2D internally)
    if isinstance(value, list):
        if num_rows == 1:
            data_2d = [value]
        else:
            data_2d = [[v] for v in value]
        compressed_summary = _build_compressed_summary(range_str, data_2d, cell_range)
        return f"INPUT | {range_str} | {compressed_summary}", True

    # Single value filling a large range
    compressed_summary = f"{COMPRESSED_MARKER} {num_rows}x{num_cols} [fill: {_format_value_preview(value)}] ({total_cells} cells)"
    return f"INPUT | {range_str} | {compressed_summary}", True


def compress_operations(operations: List[str],
                        cell_threshold: int = LARGE_RANGE_CELL_THRESHOLD,
                        string_threshold: int = LARGE_STRING_CHAR_THRESHOLD) -> Tuple[List[str], CompressionStats]:
    """
    Compress a list of symbolic operations, targeting large INPUT operations.

    Args:
        operations: List of symbolic operation strings
        cell_threshold: Compress INPUTs with ranges larger than this many cells
        string_threshold: Truncate strings longer than this

    Returns:
        Tuple of (compressed_operations, compression_stats)
    """
    stats = CompressionStats()
    compressed = []

    for op in operations:
        stats.total_operations += 1
        stats.original_char_count += len(op)

        compressed_op, was_compressed = compress_input_operation(
            op, cell_threshold, string_threshold
        )

        if was_compressed:
            stats.compressed_operations += 1

        stats.compressed_char_count += len(compressed_op)
        compressed.append(compressed_op)

    return compressed, stats


def is_compressed_input(symbolic_op: str) -> bool:
    """Check if a symbolic operation is a compressed INPUT."""
    return COMPRESSED_MARKER in symbolic_op and symbolic_op.strip().startswith("INPUT")


def decompress_input_operation(symbolic_op: str,
                                final_state: Dict[str, Any],
                                sheet_name: str) -> str:
    """
    Decompress a compressed INPUT operation using the final state.

    Args:
        symbolic_op: The compressed symbolic operation
        final_state: The final workbook state containing actual cell values
        sheet_name: The sheet name to look up values from

    Returns:
        The decompressed operation with full values from final state
    """
    if not is_compressed_input(symbolic_op):
        return symbolic_op

    # Parse the compressed format
    match = COMPRESSED_INPUT_PATTERN.match(symbolic_op.strip())
    if not match:
        return symbolic_op

    range_str = match.group(1).strip()

    # Parse the range
    try:
        cell_range = CellRange.from_string(range_str)
        min_col, min_row, max_col, max_row, num_rows, num_cols = _get_range_dimensions(range_str)
    except Exception:
        return symbolic_op

    # Look up values from final state
    sheet_state = final_state.get("worksheets", {}).get(sheet_name, {})
    cells = sheet_state.get("cells", {})

    # Build the 2D array from final state
    data_2d = []
    for row_idx in range(min_row, max_row + 1):
        row_data = []
        for col_idx in range(min_col, max_col + 1):
            cell_addr = get_cell_address(row_idx, col_idx)
            cell_data = cells.get(cell_addr, {})
            # Get value from cell - could be 'value' or 'formula'
            cell_value = cell_data.get("value")
            if cell_value is None:
                cell_value = cell_data.get("formula")
            row_data.append(cell_value)
        data_2d.append(row_data)

    # Reconstruct the full INPUT operation
    # Handle edge case of single row/column
    if num_rows == 1 and num_cols == 1:
        value = data_2d[0][0]
    elif num_rows == 1:
        value = data_2d[0]
    elif num_cols == 1:
        value = [row[0] for row in data_2d]
    else:
        value = data_2d

    return f"INPUT | {range_str} | {json.dumps(value)}"


def decompress_operations(operations: List[str],
                           final_state: Dict[str, Any],
                           sheet_name: str) -> List[str]:
    """
    Decompress all compressed INPUT operations in a list.

    Args:
        operations: List of symbolic operations (some may be compressed)
        final_state: The final workbook state for looking up values
        sheet_name: The sheet name to use

    Returns:
        List of decompressed operations
    """
    return [
        decompress_input_operation(op, final_state, sheet_name)
        for op in operations
    ]


def get_compression_docs() -> str:
    """
    Get documentation about the compressed INPUT format for prompts.
    """
    return """
### Compressed INPUT Format

Large INPUT operations (those filling many cells at once) are shown in a compressed format to save space:

```
INPUT | Sheet1!A5:Z100 | <<COMPRESSED>> 96x26 [TL:"First", TR:"Last", BL:"...", BR:"End"] (1200/2496 filled)
```

This means:
- The range A5:Z100 contains a 96-row by 26-column data block
- TL/TR/BL/BR show corner values (top-left, top-right, bottom-left, bottom-right)
- "1200/2496 filled" indicates 1200 non-empty cells out of 2496 total

**When you output operations**, you can use either:
1. The compressed format (just copy the line as-is or adjust the range)
2. The full expanded format with actual values

The system will automatically reconstruct full values from the workbook state during validation.

**Key points:**
- You can reorder, split, or merge compressed INPUTs by range
- If splitting a compressed INPUT, specify the sub-ranges (values will be auto-filled)
- The compressed format is mainly for readability - the actual data is preserved
"""


# Convenience function for integration
def normalize_for_prompt(operations: List[str],
                         cell_threshold: int = LARGE_RANGE_CELL_THRESHOLD) -> Tuple[List[str], CompressionStats]:
    """
    Normalize operations for sending to an LLM prompt.
    Compresses large INPUT operations to reduce token usage.

    Args:
        operations: List of symbolic operation strings
        cell_threshold: Threshold for compression (default 20 cells)

    Returns:
        Tuple of (normalized_operations, stats)
    """
    return compress_operations(operations, cell_threshold=cell_threshold)


def denormalize_from_response(operations: List[str],
                               final_state: Dict[str, Any],
                               sheet_name: str) -> List[str]:
    """
    Denormalize operations received from an LLM response.
    Decompresses any compressed INPUT operations using final state values.

    Args:
        operations: List of symbolic operations from LLM response
        final_state: The final workbook state
        sheet_name: The sheet name

    Returns:
        List of fully expanded operations
    """
    return decompress_operations(operations, final_state, sheet_name)
