"""
Symbolic DSL parser/serializer.

Converts between Operation objects and their symbolic string representations.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.operations import OPERATION_MAP
from next_action_pred_eval.utils.cell_utils import parse_cell, get_cell_address


def operations_to_symbolic(operations: List[Operation]) -> List[str]:
    """
    Convert operations to symbolic representation.

    Args:
        operations: List of Operation objects

    Returns:
        List of symbolic strings
    """
    return [op.to_symbolic() for op in operations]


def symbolic_to_operations(symbolic_list: List[str]) -> List[Operation]:
    """
    Convert symbolic representation back to operations.

    Args:
        symbolic_list: List of symbolic strings

    Returns:
        List of Operation objects
    """
    operations = []
    for symbolic in [sym.strip() for sym in symbolic_list if sym.strip() and not sym.strip().startswith('#')]:
        op_type = symbolic.split(" | ")[0]

        op_class = OPERATION_MAP.get(op_type)
        if op_class:
            try:
                operations.append(op_class.from_symbolic(symbolic))
            except Exception as e:
                logger.warning(f"Skipping malformed operation '{symbolic}': {e}")
        else:
            logger.warning(f"Unknown operation type in symbolic: {symbolic}")
    return operations


@dataclass
class ParseResult:
    """Result of detailed symbolic-to-operations parsing."""

    valid_operations: List[Operation]
    valid_symbolic: List[str]
    failed_entries: List[str] = field(default_factory=list)
    failed_reasons: List[str] = field(default_factory=list)


def symbolic_to_operations_detailed(symbolic_list: List[str]) -> ParseResult:
    """
    Convert symbolic representation to operations with detailed failure tracking.

    Unlike :func:`symbolic_to_operations`, this function returns information
    about which entries succeeded and which failed, allowing callers to keep
    ``predicted_symbolic`` in sync with ``predicted_operations``.

    Args:
        symbolic_list: List of symbolic strings.

    Returns:
        ParseResult with valid operations, their corresponding symbolic
        strings, and lists of failed entries with reasons.
    """
    valid_operations: List[Operation] = []
    valid_symbolic: List[str] = []
    failed_entries: List[str] = []
    failed_reasons: List[str] = []

    for symbolic in [
        sym.strip()
        for sym in symbolic_list
        if sym.strip() and not sym.strip().startswith("#")
    ]:
        op_type = symbolic.split(" | ")[0]
        op_class = OPERATION_MAP.get(op_type)

        if op_class:
            try:
                op = op_class.from_symbolic(symbolic)
                # Validate the cell range is resolvable (catches e.g. "B4:E4:C4")
                op.cell_range.get_coordinates()
                valid_operations.append(op)
                valid_symbolic.append(symbolic)
            except Exception as e:
                logger.warning("Skipping malformed operation '%s': %s", symbolic, e)
                failed_entries.append(symbolic)
                failed_reasons.append(str(e))
        else:
            logger.warning("Unknown operation type in symbolic: %s", symbolic)
            failed_entries.append(symbolic)
            failed_reasons.append(f"Unknown operation type: {op_type}")

    return ParseResult(
        valid_operations=valid_operations,
        valid_symbolic=valid_symbolic,
        failed_entries=failed_entries,
        failed_reasons=failed_reasons,
    )


def compress_symbolic(
    symbolic: List[str],
    remove_sheet_name: bool = False,
    compress_inputs: bool = False,
    remove_inputs: bool = False,
    compress_args: bool = False,
    remove_args: bool = False,
    max_len_args: int = 30
) -> List[str]:
    """
    Convert operations to compressed symbolic representation.

    Args:
        symbolic: List of symbolic operations to compress.
        remove_sheet_name: If True, remove sheet names from cell references.
        compress_inputs: If True, compress input values by merging consecutive cells.
        remove_inputs: If True, remove VALUE and FORMULA operations entirely.
        compress_args: If True, truncate argument values to max_len_args.
        remove_args: If True, remove argument values entirely.
        max_len_args: Maximum length of argument values when compressing.

    Returns:
        Compressed symbolic representation of the operations.
    """
    if compress_inputs:
        symbolic = compress_symbolic_inputs(symbolic)
    if remove_sheet_name:
        # Operation types whose value field is also a cell reference
        _REF_VALUE_OPS = frozenset({"AUTOFILL", "PASTE_FROM"})
        result = []
        for op in symbolic:
            parts = op.split(" | ")
            parts[1] = parts[1].split("!")[-1]
            if len(parts) >= 3 and parts[0] in _REF_VALUE_OPS:
                parts[2] = parts[2].split("!")[-1]
            result.append(" | ".join(parts))
        symbolic = result
    if remove_inputs:
        symbolic = [s for s in symbolic if not s.startswith(("VALUE", "FORMULA"))]
    if remove_args:
        symbolic = [" | ".join(s.split(" | ")[:2]) for s in symbolic]
    elif compress_args:
        symbolic = [
            " | ".join(parts[:2] + [(parts[2][:max_len_args] + "...") if len(parts[2]) > max_len_args else parts[2]])
            for parts in (op.split(" | ") for op in symbolic)
        ]
    return symbolic


def compress_symbolic_inputs(symbolic_ops: List[str]) -> List[str]:
    """
    Compress symbolic input operations by merging consecutive similar inputs.

    Args:
        symbolic_ops: List of symbolic operation strings

    Returns:
        Compressed list where adjacent VALUE/FORMULA operations on contiguous
        cells are merged into ranges.
    """
    compressed: List[str] = []
    prev_op_str: Optional[str] = None
    prev_type = None
    prev_sheet = None
    prev_cell_range = None

    for op_str in symbolic_ops:
        parts = op_str.split(" | ")
        op_type = parts[0]
        cell_info = parts[1]

        if op_type in ["VALUE", "FORMULA"]:
            if "!" in cell_info:
                sheet, cell_range = cell_info.rsplit("!", 1)
            else:
                sheet, cell_range = "", cell_info

            if prev_op_str and prev_type == op_type and prev_sheet == sheet:
                if ":" not in cell_range:
                    curr_cell = parse_cell(cell_range)
                    merged = False

                    if ":" not in prev_cell_range:
                        prev_cell = parse_cell(prev_cell_range)
                        start, end = prev_cell_range, cell_range

                        if prev_cell[0] == curr_cell[0] and prev_cell[1] + 1 == curr_cell[1]:
                            new_range = f"{start}:{cell_range}"
                            merged = True
                        elif prev_cell[1] == curr_cell[1] and prev_cell[0] + 1 == curr_cell[0]:
                            new_range = f"{start}:{cell_range}"
                            merged = True
                    else:
                        start, end = prev_cell_range.split(":")
                        start_cell = parse_cell(start)
                        end_cell = parse_cell(end)

                        if (curr_cell[0] == start_cell[0] == end_cell[0] and
                            (curr_cell[1] == end_cell[1] + 1 or curr_cell[1] == start_cell[1] - 1)):
                            if curr_cell[1] > end_cell[1]:
                                new_range = f"{start}:{cell_range}"
                            else:
                                new_range = f"{cell_range}:{end}"
                            merged = True
                        elif (curr_cell[1] == start_cell[1] == end_cell[1] and
                              (curr_cell[0] == end_cell[0] + 1 or curr_cell[0] == start_cell[0] - 1)):
                            if curr_cell[0] > end_cell[0]:
                                new_range = f"{start}:{cell_range}"
                            else:
                                new_range = f"{cell_range}:{end}"
                            merged = True

                    if merged:
                        prev_parts = prev_op_str.split(" | ")
                        prev_parts[1] = f"{sheet}!{new_range}" if sheet else new_range
                        prev_parts[2:] = ["MERGED_CONTENT"]
                        prev_op_str = " | ".join(prev_parts)
                        prev_cell_range = new_range
                        continue

            if prev_op_str:
                compressed.append(prev_op_str)

            prev_op_str = op_str
            prev_type = op_type
            prev_sheet = sheet
            prev_cell_range = cell_range
        else:
            if prev_op_str:
                compressed.append(prev_op_str)
                prev_op_str = None
            compressed.append(op_str)

    if prev_op_str:
        compressed.append(prev_op_str)

    return compressed


def uncompress_symbolic_inputs(symbolic_ops: List[str]) -> List[str]:
    """
    Uncompress symbolic input operations by splitting ranges back into individual cells.

    Args:
        symbolic_ops: List of symbolic operation strings (possibly with ranges)

    Returns:
        Expanded list where VALUE/FORMULA ranges are split into individual cell operations.
    """
    uncompressed: List[str] = []

    for op_str in symbolic_ops:
        parts = op_str.split(" | ")
        op_type = parts[0]
        cell_info = parts[1]

        if op_type in ["VALUE", "FORMULA"]:
            if "!" in cell_info:
                sheet, cell_range = cell_info.rsplit("!", 1)
            else:
                sheet, cell_range = "", cell_info

            if ":" in cell_range:
                start, end = cell_range.split(":")
                start_cell = parse_cell(start)
                end_cell = parse_cell(end)

                for r in range(start_cell[0], end_cell[0] + 1):
                    for c in range(start_cell[1], end_cell[1] + 1):
                        cell_addr = get_cell_address(r, c)
                        if sheet:
                            full_cell = f"{sheet}!{cell_addr}"
                        else:
                            full_cell = cell_addr
                        new_parts = parts.copy()
                        new_parts[1] = full_cell
                        uncompressed.append(" | ".join(new_parts))
            else:
                uncompressed.append(op_str)
        else:
            uncompressed.append(op_str)

    return uncompressed


def parse_symbolic(symbolic_str: str) -> dict:
    """
    Parse a single symbolic string into its components.

    Args:
        symbolic_str: A symbolic operation string like "VALUE | Sheet1!A1 | 123"

    Returns:
        Dictionary with keys: 'operation', 'cell_range', 'value', 'parts'
    """
    parts = symbolic_str.split(" | ")
    result = {
        'operation': parts[0] if len(parts) > 0 else None,
        'cell_range': parts[1] if len(parts) > 1 else None,
        'value': parts[2] if len(parts) > 2 else None,
        'parts': parts
    }

    # Parse cell_range into sheet and range
    if result['cell_range'] and '!' in result['cell_range']:
        result['sheet'], result['range'] = result['cell_range'].rsplit('!', 1)
    elif result['cell_range']:
        result['sheet'] = None
        result['range'] = result['cell_range']

    return result


def format_symbolic(
    operation: str,
    cell_range: str,
    value: Optional[str] = None,
    sheet: Optional[str] = None
) -> str:
    """
    Format components into a symbolic string.

    Args:
        operation: Operation name (e.g., "VALUE", "FORMULA")
        cell_range: Cell range (e.g., "A1" or "A1:B2")
        value: Optional value for the operation
        sheet: Optional sheet name

    Returns:
        Formatted symbolic string
    """
    if sheet:
        full_range = f"{sheet}!{cell_range}"
    else:
        full_range = cell_range

    if value is not None:
        return f"{operation} | {full_range} | {value}"
    else:
        return f"{operation} | {full_range}"
