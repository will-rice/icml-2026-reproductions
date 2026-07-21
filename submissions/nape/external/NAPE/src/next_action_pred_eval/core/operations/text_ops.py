"""
Text operations - SetWrapText, SetTextOrientation operations.
"""

from typing import Any, Dict, List

from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.defaults import EXCEL_DEFAULTS
from next_action_pred_eval.core.operations._helpers import (
    _ensure_cell,
    _ensure_format,
    _get_cells_in_range,
)


# ============= Operation Classes =============

class SetTextOrientation(Operation):
    """Set text orientation/rotation."""

    def to_symbolic(self) -> str:
        return f"TEXT_ORIENTATION | {self.cell_range} | {self.value}"

    def to_officejs(self, sheet_var: str = "sheet") -> str:
        return f'{sheet_var}.getRange("{self.cell_range.range}").format.textOrientation = {self.value};'

    def to_openpyxl(self, sheet_var: str = "ws") -> str:
        return f'{sheet_var}["{self.cell_range.range}"].set_alignment(text_rotation={self.value})'

    def to_xlwings(self, sheet_var: str = "sheet") -> str:
        return f'{sheet_var}["{self.cell_range.range}"].api.Orientation = {self.value}'

    @classmethod
    def from_symbolic(cls, symbolic: str) -> 'SetTextOrientation':
        parts = symbolic.split(' | ', 2)
        cell_range = CellRange.from_string(parts[1])

        return cls(cell_range=cell_range, value=int(parts[2]), is_inverse=False)

    def apply_to_state(self, state: Dict[str, Any]) -> None:
        """Apply SetTextOrientation to state."""
        for cell_addr in _get_cells_in_range(self.cell_range):
            cell = _ensure_cell(state, self.cell_range.sheet, cell_addr)
            fmt = _ensure_format(cell)

            if self.value == EXCEL_DEFAULTS["text_orientation"]:
                fmt.pop("textOrientation", None)
            else:
                fmt["textOrientation"] = self.value

    def get_inverse(self) -> 'Operation':
        """Return operation to reset text orientation to horizontal."""
        return SetTextOrientation(cell_range=self.cell_range, value=EXCEL_DEFAULTS["text_orientation"], is_inverse=True)

    @property
    def modifies_format(self) -> bool:
        return True


class SetWrapText(Operation):
    """Set text wrapping for cells."""

    def to_symbolic(self) -> str:
        return f"WRAP_TEXT | {self.cell_range} | {str(self.value).lower()}"

    def to_officejs(self, sheet_var: str = "sheet") -> str:
        wrap_value = "true" if self.value else "false"
        return f'{sheet_var}.getRange("{self.cell_range.range}").format.wrapText = {wrap_value};'

    def to_openpyxl(self, sheet_var: str = "ws") -> str:
        return f'{sheet_var}["{self.cell_range.range}"].set_alignment(wrap_text={self.value})'

    def to_xlwings(self, sheet_var: str = "sheet") -> str:
        value = 'True' if self.value else 'False'
        return f'{sheet_var}["{self.cell_range.range}"].wrap_text = {value}'

    @classmethod
    def from_symbolic(cls, symbolic: str) -> 'SetWrapText':
        parts = symbolic.split(' | ', 2)
        cell_range = CellRange.from_string(parts[1])

        value = parts[2].lower() == 'true'
        return cls(cell_range=cell_range, value=value, is_inverse=False)

    def apply_to_state(self, state: Dict[str, Any]) -> None:
        """Apply SetWrapText to state."""
        for cell_addr in _get_cells_in_range(self.cell_range):
            cell = _ensure_cell(state, self.cell_range.sheet, cell_addr)
            fmt = _ensure_format(cell)

            if self.value == EXCEL_DEFAULTS["wrap_text"]:
                fmt.pop("wrapText", None)
            else:
                fmt["wrapText"] = self.value

    def get_inverse(self) -> 'Operation':
        """Return operation to reset wrap text to default (no wrapping)."""
        return SetWrapText(cell_range=self.cell_range, value=EXCEL_DEFAULTS["wrap_text"], is_inverse=True)
