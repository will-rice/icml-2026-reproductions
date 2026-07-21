from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from openpyxl.utils.cell import coordinate_from_string, column_index_from_string

from next_action_pred_eval.core.symbolic import symbolic_to_operations
from next_action_pred_eval.core.state import StateBuilder


def _cell_within_bounds(address: str, max_dimension: Optional[int]) -> bool:
    if max_dimension is None:
        return True
    coord = coordinate_from_string(address)
    col = column_index_from_string(coord[0])
    row = coord[1]
    return col <= max_dimension and row <= max_dimension


def _merge_within_bounds(merge: Dict[str, int], max_dimension: Optional[int]) -> bool:
    if max_dimension is None:
        return True
    bounds = [merge.get("start_row", 0), merge.get("start_col", 0), merge.get("end_row", 0), merge.get("end_col", 0)]
    return max(bounds) <= max_dimension


def filter_state(
    state: Dict[str, Any],
    sheet_name: str,
    max_dimension: Optional[int],
    include_formatting: bool = True,
) -> Dict[str, Any]:
    filtered: Dict[str, Any] = {"worksheets": {}}
    sheet = state.get("worksheets", {}).get(sheet_name)
    if not sheet:
        return filtered

    cells = {}
    for addr, payload in sheet.get("cells", {}).items():
        if _cell_within_bounds(addr, max_dimension):
            cell_payload = deepcopy(payload)
            if not include_formatting and "Format" in cell_payload:
                cell_payload.pop("Format", None)
            cells[addr] = cell_payload

    merged_cells = [mc for mc in sheet.get("worksheetProperties", {}).get("merged_cells", []) if _merge_within_bounds(mc, max_dimension)]

    filtered["worksheets"][sheet_name] = {
        "cells": cells,
        "worksheetProperties": {"merged_cells": merged_cells},
    }
    return filtered


def load_target_state(
    workbook_path: str,
    sheet_name: str,
    max_dimension: Optional[int],
    include_formatting: bool = True,
) -> Dict[str, Any]:
    """Load target state from a workbook using StateBuilder.from_workbook."""
    builder = StateBuilder.from_workbook(workbook_path)
    state = builder.get_state()
    return filter_state(state, sheet_name, max_dimension, include_formatting=include_formatting)


def build_state_from_operations(
    symbolic_ops: List[str],
    sheet_name: str,
    max_dimension: Optional[int],
    include_formatting: bool = True,
) -> Dict[str, Any]:
    """
    Build target state by applying symbolic operations to an empty state.

    This provides an alternative to loading state from an Excel workbook,
    allowing the target state to be derived purely from the operation sequence.

    Args:
        symbolic_ops: List of symbolic operation strings (e.g., "INPUT | Sheet1!A1 | value")
        sheet_name: Name of the sheet to filter to
        max_dimension: Maximum row/column dimension to include (None for no limit)
        include_formatting: Whether to include formatting properties in the state

    Returns:
        Filtered state dictionary built from applying the operations
    """
    op_objects = symbolic_to_operations(symbolic_ops)
    builder = StateBuilder()
    state = builder.apply_operations(op_objects)
    return filter_state(state, sheet_name, max_dimension, include_formatting=include_formatting)
