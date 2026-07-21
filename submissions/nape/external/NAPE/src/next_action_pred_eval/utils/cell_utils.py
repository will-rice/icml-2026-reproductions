"""
Cell utilities - Helper functions for cell address and range manipulation.
"""

import re
from typing import List, Tuple

from openpyxl.utils import get_column_letter, column_index_from_string
from openpyxl.utils import range_boundaries


def parse_cell(cell_str: str) -> Tuple[int, int]:
    """
    Parse a cell address into (row, col) tuple.

    Args:
        cell_str: Cell address like "A1", "BC42"

    Returns:
        Tuple of (row, col) - both 1-indexed

    Examples:
        >>> parse_cell("A1")
        (1, 1)
        >>> parse_cell("B3")
        (3, 2)
    """
    m = re.match(r"([A-Za-z]+)(\d+)$", cell_str)
    if not m:
        raise ValueError(f"Invalid cell address: {cell_str}")
    col_s, row_s = m.groups()
    return int(row_s), column_index_from_string(col_s.upper())


def get_cell_address(row: int, col: int) -> str:
    """
    Convert row/col to Excel address.

    Args:
        row: Row number (1-indexed)
        col: Column number (1-indexed)

    Returns:
        Cell address string like "A1"

    Examples:
        >>> get_cell_address(1, 1)
        'A1'
        >>> get_cell_address(3, 2)
        'B3'
    """
    return f"{get_column_letter(col)}{row}"


def expand_range(range_str: str) -> List[Tuple[int, int]]:
    """
    Expand a range string into a list of (row, col) tuples.

    Args:
        range_str: Range string like "A1" or "A1:C3"

    Returns:
        List of (row, col) tuples for all cells in the range

    Examples:
        >>> expand_range("A1")
        [(1, 1)]
        >>> expand_range("A1:B2")
        [(1, 1), (1, 2), (2, 1), (2, 2)]
    """
    if ":" not in range_str:
        return [parse_cell(range_str)]

    start, end = range_str.split(":")
    sr, sc = parse_cell(start)
    er, ec = parse_cell(end)

    cells: List[Tuple[int, int]] = []
    for r in range(sr, er + 1):
        for c in range(sc, ec + 1):
            cells.append((r, c))
    return cells


def get_range_string(start_row: int, start_col: int, end_row: int, end_col: int) -> str:
    """
    Convert row/col coordinates to Excel range string.

    Args:
        start_row: Starting row (1-indexed)
        start_col: Starting column (1-indexed)
        end_row: Ending row (1-indexed)
        end_col: Ending column (1-indexed)

    Returns:
        Range string like "A1" or "A1:C3"
    """
    start_addr = get_cell_address(start_row, start_col)
    if start_row == end_row and start_col == end_col:
        return start_addr
    end_addr = get_cell_address(end_row, end_col)
    return f"{start_addr}:{end_addr}"


def cells_to_range(cells: List[Tuple[int, int]]) -> str:
    """
    Convert a list of cell coordinates to a range string.

    Args:
        cells: List of (row, col) tuples

    Returns:
        Range string covering all cells
    """
    if not cells:
        return ""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r1, r2 = min(rows), max(rows)
    c1, c2 = min(cols), max(cols)
    a = get_cell_address(r1, c1)
    b = get_cell_address(r2, c2)
    return a if a == b else f"{a}:{b}"


def ranges_intersect(range1: str, range2: str) -> bool:
    """
    Check if two Excel ranges intersect.

    Args:
        range1: Excel range string (e.g., "A1", "A1:B3")
        range2: Excel range string (e.g., "A1", "A1:B3")

    Returns:
        True if the ranges intersect, False otherwise
    """
    min_col1, min_row1, max_col1, max_row1 = range_boundaries(range1)
    min_col2, min_row2, max_col2, max_row2 = range_boundaries(range2)

    row_overlap = (min_row1 <= max_row2 and min_row2 <= max_row1)
    col_overlap = (min_col1 <= max_col2 and min_col2 <= max_col1)

    return row_overlap and col_overlap
