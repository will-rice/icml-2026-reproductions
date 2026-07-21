"""
CellRange model - Represents a cell or range of cells in a spreadsheet.
"""

from typing import Tuple
from pydantic import BaseModel, ConfigDict, model_validator
from openpyxl.utils import range_boundaries


class CellRange(BaseModel):
    """
    Represents a cell or range of cells in a spreadsheet.

    Attributes:
        sheet: The name of the worksheet
        range: The cell range (e.g., "A1", "A1:B2")

    Examples:
        >>> CellRange(sheet="Sheet1", range="A1")
        >>> CellRange.from_string("Sheet1!A1:B2")
    """

    sheet: str
    range: str

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _validate_range(self) -> "CellRange":
        """Eagerly validate that ``range`` is a valid Excel cell/range."""
        min_col, min_row, max_col, max_row = range_boundaries(self.range)
        if any(v is None for v in (min_col, min_row, max_col, max_row)):
            raise ValueError(
                f"Column-only or row-only ranges are not supported: {self.range}"
            )
        return self

    def __str__(self) -> str:
        return f"{self.sheet}!{self.range}"

    def __repr__(self) -> str:
        return f"CellRange(sheet='{self.sheet}', range='{self.range}')"

    @classmethod
    def from_string(cls, range_str: str) -> "CellRange":
        """
        Create a CellRange from a string like "Sheet1!A1:B2" or just "A1".

        Args:
            range_str: String in format "Sheet!Range" or just "Range"

        Returns:
            CellRange instance

        Examples:
            >>> CellRange.from_string("Sheet1!A1:B2")
            CellRange(sheet='Sheet1', range='A1:B2')
            >>> CellRange.from_string("A1")
            CellRange(sheet='Sheet1', range='A1')
        """
        if '!' in range_str:
            sheet, range_part = range_str.split('!', 1)
        else:
            sheet = "Sheet1"
            range_part = range_str
        return cls(sheet=sheet, range=range_part)

    def get_dimensions(self) -> Tuple[int, int]:
        """
        Get the dimensions (rows, cols) of the range.

        Returns:
            Tuple of (num_rows, num_cols)

        Examples:
            >>> CellRange(sheet="Sheet1", range="A1:C3").get_dimensions()
            (3, 3)
        """
        min_col, min_row, max_col, max_row = range_boundaries(self.range)
        rows = max_row - min_row + 1
        cols = max_col - min_col + 1
        return (rows, cols)

    def get_coordinates(self) -> Tuple[int, int, int, int]:
        """
        Get the coordinates (start_row, start_col, end_row, end_col) of the range.

        Returns:
            Tuple of (min_row, min_col, max_row, max_col) - 1-indexed

        Examples:
            >>> CellRange(sheet="Sheet1", range="B2:D4").get_coordinates()
            (2, 2, 4, 4)
        """
        min_col, min_row, max_col, max_row = range_boundaries(self.range)
        return (min_row, min_col, max_row, max_col)

    def is_subset(self, other: "CellRange") -> bool:
        """
        Check if this range is a subset of another range.

        Args:
            other: Another CellRange to compare against

        Returns:
            True if this range is completely contained within the other range
        """
        if not isinstance(other, CellRange):
            return False
        if self.sheet != other.sheet:
            return False

        s1, s2, s3, s4 = self.get_coordinates()
        e1, e2, e3, e4 = other.get_coordinates()
        return (s1 >= e1 and s2 >= e2 and s3 <= e3 and s4 <= e4)

    def is_single_cell(self) -> bool:
        """
        Check if this range represents a single cell.

        Returns:
            True if the range is a single cell
        """
        rows, cols = self.get_dimensions()
        return rows == 1 and cols == 1
