"""
Shared state manipulation helpers for operation implementations.
"""

from typing import Any, Dict, List

from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.utils.cell_utils import expand_range, get_cell_address


def _ensure_sheet(state: Dict[str, Any], sheet_name: str) -> Dict[str, Any]:
    """Ensure worksheet exists in state and return reference."""
    if "worksheets" not in state:
        state["worksheets"] = {}
    if sheet_name not in state["worksheets"]:
        state["worksheets"][sheet_name] = {
            "cells": {},
            "worksheetProperties": {"merged_cells": []}
        }
    return state["worksheets"][sheet_name]


def _ensure_cell(state: Dict[str, Any], sheet_name: str, cell_addr: str) -> Dict[str, Any]:
    """Ensure cell exists in state and return reference."""
    sheet = _ensure_sheet(state, sheet_name)
    if cell_addr not in sheet["cells"]:
        sheet["cells"][cell_addr] = {}
    return sheet["cells"][cell_addr]


def _ensure_format(cell: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure Format section exists in cell and return reference."""
    if "Format" not in cell:
        cell["Format"] = {}
    return cell["Format"]


def _get_cells_in_range(cell_range: CellRange) -> List[str]:
    """Get all cell addresses in a range."""
    cells = []
    for row, col in expand_range(cell_range.range):
        cells.append(get_cell_address(row, col))
    return cells
