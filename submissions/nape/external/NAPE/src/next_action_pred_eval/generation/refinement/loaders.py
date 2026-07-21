from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from openpyxl.utils import range_boundaries

from next_action_pred_eval.core.cell_range import CellRange


def _normalize_range(range_token: str, default_sheet: str) -> str:
    return range_token if "!" in range_token else f"{default_sheet}!{range_token}"


def _range_within_bounds(range_token: str, max_dimension: Optional[int]) -> bool:
    if max_dimension is None:
        return True
    _, range_part = range_token.split("!", 1)
    min_col, min_row, max_col, max_row = range_boundaries(range_part)
    return max(min_col, max_col) <= max_dimension and max(min_row, max_row) <= max_dimension


def load_symbolic_operations(
    step_file: Path,
    sheet_name: str,
    max_dimension: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Load symbolic operations, filtering to the requested sheet and bounds."""
    operations: List[str] = []
    skipped: List[str] = []

    for raw_line in step_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if " | " not in line:
            continue

        parts = line.split(" | ")
        if len(parts) < 2:
            skipped.append(f"Invalid format: {line}")
            continue

        op_range = _normalize_range(parts[1], sheet_name)
        try:
            cell_range = CellRange.from_string(op_range)
        except Exception:
            skipped.append(f"Unparseable range: {line}")
            continue

        if cell_range.sheet != sheet_name:
            continue
        if not _range_within_bounds(str(cell_range), max_dimension):
            continue

        parts[1] = str(cell_range)
        operations.append(" | ".join(parts))

    return operations, skipped


def filter_operations_by_scope(
    symbolic_ops: List[str],
    sheet_name: str,
    max_dimension: Optional[int],
) -> Tuple[List[str], List[str]]:
    """Ensure symbolic operations stay within the configured sheet and bounds."""
    in_scope: List[str] = []
    rejected: List[str] = []

    for op in symbolic_ops:
        try:
            parts = op.split(" | ")
            if len(parts) < 2:
                rejected.append(op)
                continue
            op_range = _normalize_range(parts[1], sheet_name)
            cell_range = CellRange.from_string(op_range)
            if cell_range.sheet != sheet_name or not _range_within_bounds(str(cell_range), max_dimension):
                rejected.append(op)
                continue
            parts[1] = str(cell_range)
            in_scope.append(" | ".join(parts))
        except Exception:
            rejected.append(op)
    return in_scope, rejected
