"""Transform that converts cell ranges to relative (delta) representation.

Standard:   FILL_COLOR | Sheet1!B1 | #FFFF00
Encoded:    FILL_COLOR | (0,1,0,0) | #FFFF00

Deltas are (row_delta, col_delta, height_delta, width_delta) relative
to the previous operation's range.
"""

import re
from typing import Any, Dict, List, Tuple

from .base import SymbolicTransform

_CELL_RE = re.compile(r"([A-Za-z]+)(\d+)$")
_DELTA_RE = re.compile(r"\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)")


def _parse_cell(cell_str: str) -> Tuple[int, int]:
    """Parse cell address to (row, col), both 1-indexed."""
    m = _CELL_RE.match(cell_str.strip())
    if not m:
        return (1, 1)
    col_s, row_s = m.groups()
    col = 0
    for ch in col_s.upper():
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return (int(row_s), col)


def _col_letter(col: int) -> str:
    """Convert 1-indexed column number to letter(s)."""
    result: list = []
    while col > 0:
        col, rem = divmod(col - 1, 26)
        result.append(chr(rem + ord("A")))
    return "".join(reversed(result)) if result else "A"


def _cell_addr(row: int, col: int) -> str:
    return f"{_col_letter(col)}{row}"


def _parse_range(range_str: str) -> Tuple[int, int, int, int]:
    """Parse range to (start_row, start_col, height, width). All 1-indexed."""
    parts = range_str.split(":")
    sr, sc = _parse_cell(parts[0])
    if len(parts) > 1:
        er, ec = _parse_cell(parts[1])
    else:
        er, ec = sr, sc
    return (sr, sc, er - sr + 1, ec - sc + 1)


def _make_range(row: int, col: int, height: int, width: int) -> str:
    """Build a range string from row, col, height, width."""
    row, col = max(1, row), max(1, col)
    height, width = max(1, height), max(1, width)
    start = _cell_addr(row, col)
    if height == 1 and width == 1:
        return start
    return f"{start}:{_cell_addr(row + height - 1, col + width - 1)}"


class RelativeRangeTransform(SymbolicTransform):
    """Converts cell ranges to relative delta representation.

    Each operation's range is encoded as ``(dr, dc, dh, dw)`` where:
    - ``dr``: row delta from previous operation's start row
    - ``dc``: column delta from previous operation's start column
    - ``dh``: height delta from previous operation's height
    - ``dw``: width delta from previous operation's width

    Sheet names are stored in transform state but stripped from output.
    They are restored during decoding.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._row: int = 1
        self._col: int = 1
        self._height: int = 1
        self._width: int = 1
        self._sheet: str = ""

    def encode_one(self, symbolic_str: str) -> str:
        parts = symbolic_str.split(" | ")
        op_type = parts[0].strip() if parts else ""
        cell_info = parts[1].strip() if len(parts) > 1 else "A1"
        rest = parts[2:] if len(parts) > 2 else []

        if "!" in cell_info:
            sheet, range_str = cell_info.rsplit("!", 1)
        else:
            sheet, range_str = self._sheet, cell_info

        sr, sc, h, w = _parse_range(range_str)

        dr = sr - self._row
        dc = sc - self._col
        dh = h - self._height
        dw = w - self._width

        self._row, self._col = sr, sc
        self._height, self._width = h, w
        if sheet:
            self._sheet = sheet

        range_token = f"({dr},{dc},{dh},{dw})"
        return " | ".join([op_type, range_token] + rest)

    def decode_predictions(self, predictions: List[str]) -> List[str]:
        row, col = self._row, self._col
        height, width = self._height, self._width
        sheet = self._sheet

        results: list = []
        for pred in predictions:
            parts = pred.split(" | ")
            op_type = parts[0].strip()
            range_token = parts[1].strip() if len(parts) > 1 else "(0,0,0,0)"
            rest = parts[2:] if len(parts) > 2 else []

            # Handle optional sheet prefix (e.g. "Sheet1!(dr,dc,dh,dw)")
            pred_sheet = ""
            if "!" in range_token:
                pred_sheet, range_token = range_token.rsplit("!", 1)

            m = _DELTA_RE.match(range_token)
            if not m:
                results.append(pred)
                continue

            dr, dc, dh, dw = (int(x) for x in m.groups())

            row = max(1, row + dr)
            col = max(1, col + dc)
            height = max(1, height + dh)
            width = max(1, width + dw)

            range_str = _make_range(row, col, height, width)
            resolved_sheet = pred_sheet or sheet
            if resolved_sheet:
                range_str = f"{resolved_sheet}!{range_str}"

            results.append(" | ".join([op_type, range_str] + rest))

        return results

    def get_config(self) -> Dict[str, Any]:
        return {"type": "relative_range"}
