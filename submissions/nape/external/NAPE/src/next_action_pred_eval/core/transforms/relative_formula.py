"""Transform that converts cell references within formulas to relative notation.

Standard:   INPUT | E2 | =SUM(A2:C2) - D2
Encoded:    INPUT | E2 | =SUM(R[0]C[-4]:R[0]C[-2]) - R[0]C[-1]

Regular cell refs become ``R[row_delta]C[col_delta]`` relative to the
operation's own cell.  Non-formula values pass through unchanged.

Formulas inside INPUT list values (e.g. ``[["=A1*2", "=B1*2"]]``) are
also relativized per-cell.

Cross-sheet references (``Sheet2!A1``) have the sheet prefix preserved
but the cell ref is still relativized.
"""

import json
import re
from typing import Any, Dict, List, Tuple

from .base import SymbolicTransform

_CELL_RE = re.compile(r"([A-Za-z]+)(\d+)$")
_DELTA_RANGE_RE = re.compile(r"\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)")

# Match cell refs in formulas.  Allows optional sheet prefix.
# Excludes: followed by ( (function call) or preceded by letters (function name).
_FORMULA_REF_RE = re.compile(
    r"(?<![A-Za-z\d_])"
    r"((?:'[^']*'!|[A-Za-z_]\w*!)?)"
    r"\$?([A-Za-z]{1,3})"
    r"\$?(\d{1,7})"
    r"(?!\w|\()",
)

# Match our relative notation: R[int]C[int]
_RELATIVE_REF_RE = re.compile(
    r"((?:'[^']*'!|[A-Za-z_]\w*!)?)"
    r"R\[(-?\d+)\]C\[(-?\d+)\]"
)


def _col_to_num(col_str: str) -> int:
    result = 0
    for ch in col_str.upper():
        result = result * 26 + (ord(ch) - ord("A") + 1)
    return result


def _num_to_col(n: int) -> str:
    result: list = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result.append(chr(rem + ord("A")))
    return "".join(reversed(result)) if result else "A"


def _parse_cell(cell_str: str) -> Tuple[int, int]:
    m = _CELL_RE.match(cell_str.strip())
    if not m:
        return (1, 1)
    return (int(m.group(2)), _col_to_num(m.group(1)))


def _parse_range(range_str: str) -> Tuple[int, int, int, int]:
    """Parse range to (start_row, start_col, height, width)."""
    parts = range_str.split(":")
    sr, sc = _parse_cell(parts[0])
    if len(parts) > 1:
        er, ec = _parse_cell(parts[1])
    else:
        er, ec = sr, sc
    return (sr, sc, er - sr + 1, ec - sc + 1)


class RelativeFormulaTransform(SymbolicTransform):
    """Converts cell references within formulas to R[dr]C[dc] notation.

    Processes values that start with ``=`` (formulas) and formula strings
    embedded in INPUT list values (``[["=A1*2", ...]]``).

    This transform should be applied BEFORE ``RelativeRangeTransform``
    because it needs absolute cell addresses to compute relative formula
    refs. The canonical ordering is enforced by ``build_transforms()``.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._row: int = 1
        self._col: int = 1
        self._height: int = 1
        self._width: int = 1

    def encode_one(self, symbolic_str: str) -> str:
        parts = symbolic_str.split(" | ")
        op_type = parts[0].strip() if parts else ""
        cell_info = parts[1].strip() if len(parts) > 1 else "A1"
        value = " | ".join(parts[2:]) if len(parts) > 2 else ""

        # Extract position from range
        if "!" in cell_info:
            _, range_str = cell_info.rsplit("!", 1)
        else:
            range_str = cell_info

        # If range is already in delta form, keep saved position
        if _DELTA_RANGE_RE.match(range_str):
            cell_row, cell_col = self._row, self._col
            h, w = self._height, self._width
        else:
            cell_row, cell_col, h, w = _parse_range(range_str)
            self._row, self._col = cell_row, cell_col
            self._height, self._width = h, w

        if value.startswith("="):
            value = self._relativize_formula(value, cell_row, cell_col)
        elif (value.startswith("[[") or value.startswith("[")) and '"=' in value:
            value = self._relativize_list_formulas(
                value, cell_row, cell_col,
            )

        return " | ".join([op_type, cell_info, value])

    def _relativize_formula(
        self, formula: str, cell_row: int, cell_col: int,
    ) -> str:
        def replace_ref(m: re.Match) -> str:
            sheet_prefix = m.group(1)
            col_str = m.group(2).upper()
            row_num = int(m.group(3))
            ref_col = _col_to_num(col_str)
            dr = row_num - cell_row
            dc = ref_col - cell_col
            return f"{sheet_prefix}R[{dr}]C[{dc}]"

        return _FORMULA_REF_RE.sub(replace_ref, formula)

    def _relativize_list_formulas(
        self, list_str: str, start_row: int, start_col: int,
    ) -> str:
        try:
            data = json.loads(list_str)
        except (json.JSONDecodeError, ValueError):
            return list_str
        if not isinstance(data, list):
            return list_str

        changed = False
        for r_idx, row in enumerate(data):
            if not isinstance(row, list):
                continue
            for c_idx, cell in enumerate(row):
                if isinstance(cell, str) and cell.startswith("="):
                    cell_row = start_row + r_idx
                    cell_col = start_col + c_idx
                    data[r_idx][c_idx] = self._relativize_formula(
                        cell, cell_row, cell_col,
                    )
                    changed = True

        return json.dumps(data, ensure_ascii=False) if changed else list_str

    def decode_predictions(self, predictions: List[str]) -> List[str]:
        row, col = self._row, self._col
        height, width = self._height, self._width
        results: list = []

        for pred in predictions:
            parts = pred.split(" | ")
            op_type = parts[0].strip()
            cell_info = parts[1].strip() if len(parts) > 1 else "A1"
            value = " | ".join(parts[2:]) if len(parts) > 2 else ""

            # Extract position from range
            if "!" in cell_info:
                _, range_str = cell_info.rsplit("!", 1)
            else:
                range_str = cell_info

            dm = _DELTA_RANGE_RE.match(range_str)
            if dm:
                dr, dc, dh, dw = (int(x) for x in dm.groups())
                cell_row = max(1, row + dr)
                cell_col = max(1, col + dc)
                h = max(1, height + dh)
                w = max(1, width + dw)
            else:
                cell_row, cell_col, h, w = _parse_range(range_str)

            if value.startswith("=") and _RELATIVE_REF_RE.search(value):
                value = self._absolutize_formula(value, cell_row, cell_col)
            elif (
                (value.startswith("[[") or value.startswith("["))
                and "R[" in value
            ):
                value = self._absolutize_list_formulas(
                    value, cell_row, cell_col,
                )

            row, col = cell_row, cell_col
            height, width = h, w
            results.append(" | ".join([op_type, cell_info, value]))

        return results

    def _absolutize_formula(
        self, formula: str, cell_row: int, cell_col: int,
    ) -> str:
        def replace_ref(m: re.Match) -> str:
            sheet_prefix = m.group(1)
            dr, dc = int(m.group(2)), int(m.group(3))
            abs_row = max(1, cell_row + dr)
            abs_col = max(1, cell_col + dc)
            return f"{sheet_prefix}{_num_to_col(abs_col)}{abs_row}"

        try:
            return _RELATIVE_REF_RE.sub(replace_ref, formula)
        except Exception:
            return formula

    def _absolutize_list_formulas(
        self, list_str: str, start_row: int, start_col: int,
    ) -> str:
        try:
            data = json.loads(list_str)
        except (json.JSONDecodeError, ValueError):
            return list_str
        if not isinstance(data, list):
            return list_str

        changed = False
        for r_idx, row in enumerate(data):
            if not isinstance(row, list):
                continue
            for c_idx, cell in enumerate(row):
                if isinstance(cell, str) and "R[" in cell and "C[" in cell:
                    cell_row = start_row + r_idx
                    cell_col = start_col + c_idx
                    data[r_idx][c_idx] = self._absolutize_formula(
                        cell, cell_row, cell_col,
                    )
                    changed = True

        return json.dumps(data, ensure_ascii=False) if changed else list_str

    def get_config(self) -> Dict[str, Any]:
        return {"type": "relative_formula"}
