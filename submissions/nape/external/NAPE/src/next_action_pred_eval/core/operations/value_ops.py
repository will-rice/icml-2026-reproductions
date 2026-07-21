"""
Value operations - SetValue, SetFormula, SetInput operations.
"""

from datetime import datetime, time
from typing import Any, Dict, List, Optional

import json
from pydantic import model_validator

from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.utils.cell_utils import expand_range, get_cell_address
from next_action_pred_eval.core.operations._helpers import (
    _ensure_cell,
    _get_cells_in_range,
)


def get_cells_in_range(cell_range: CellRange) -> List[str]:
    """
    Get all cell addresses in a range.

    Public interface for the _get_cells_in_range helper function.

    Args:
        cell_range: CellRange object

    Returns:
        List of cell addresses (e.g., ["A1", "A2", "B1", "B2"])
    """
    return _get_cells_in_range(cell_range)


def _process_value_for_openpyxl(val: Any) -> str:
    """Process individual values for datetime/time handling in OpenPyXL output."""
    try:
        datetime.fromisoformat(str(val))
        return f'datetime.fromisoformat("{val}")'
    except (ValueError, TypeError):
        pass
    try:
        time.fromisoformat(str(val))
        return f'time.fromisoformat("{val}")'
    except (ValueError, TypeError):
        pass
    return json.dumps(val) if isinstance(val, str) else str(val)


# ============= Operation Classes =============

class SetValue(Operation):
    """Set cell value operation."""

    datatype: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def process_datetime_values(cls, data: Any) -> Any:
        """Process datetime/time values before validation."""
        if isinstance(data, dict):
            value = data.get('value')
            datatype = data.get('datatype')

            if value is not None:
                try:
                    if datatype == "datetime":
                        data['value'] = f'{datetime.fromisoformat(value)}'
                    elif datatype == "time":
                        data['value'] = f'{time.fromisoformat(value)}'
                except (ValueError, TypeError):
                    pass
        return data

    def to_symbolic(self) -> str:
        if self.value is None:
            return f"VALUE | {self.cell_range} | clear"
        value_str = json.dumps(self.value)
        return f"VALUE | {self.cell_range} | {value_str}"

    def to_officejs(self, sheet_var: str = "sheet") -> str:
        if self.value is None:
            return f'{sheet_var}.getRange("{self.cell_range.range}").clear(Excel.ClearApplyTo.contents);'
        elif isinstance(self.value, str):
            value_str = json.dumps(self.value)
        else:
            value_str = str(self.value)
        return f'{sheet_var}.getRange("{self.cell_range.range}").values = [[{value_str}]];'

    def to_openpyxl(self, sheet_var: str = "ws") -> str:
        value_str = _process_value_for_openpyxl(self.value)
        return f'{sheet_var}["{self.cell_range.range}"].value = {value_str}'

    def to_xlwings(self, sheet_var: str = "sheet") -> str:
        if self.value is None:
            return f'{sheet_var}["{self.cell_range.range}"].value = None'
        elif isinstance(self.value, str):
            value_str = json.dumps(self.value)
        else:
            value_str = str(self.value)
        return f'{sheet_var}["{self.cell_range.range}"].value = {value_str}'

    @classmethod
    def from_symbolic(cls, symbolic: str) -> 'SetValue':
        parts = symbolic.split(' | ', 2)
        cell_range = CellRange.from_string(parts[1])

        if len(parts) > 2 and parts[2] == 'clear':
            return cls(cell_range=cell_range, value=None, is_inverse=True)

        try:
            value = json.loads(parts[2])
        except (json.JSONDecodeError, ValueError):
            value = parts[2]
        return cls(cell_range=cell_range, value=value, is_inverse=False)

    def apply_to_state(self, state: Dict[str, Any]) -> None:
        """Apply SetValue to state."""
        cell = _ensure_cell(state, self.cell_range.sheet, self.cell_range.range)

        if self.value is None:
            cell.pop("value", None)
            cell.pop("datatype", None)
        else:
            cell["value"] = self.value
            if self.datatype:
                cell["datatype"] = self.datatype

    def get_inverse(self) -> 'Operation':
        """Return operation to clear this cell value."""
        return SetValue(cell_range=self.cell_range, value=None, is_inverse=True)


class SetFormula(Operation):
    """Set cell formula operation."""

    def to_symbolic(self) -> str:
        if self.value is None:
            return f"FORMULA | {self.cell_range} | clear"
        return f"FORMULA | {self.cell_range} | {self.value}"

    def to_officejs(self, sheet_var: str = "sheet") -> str:
        if self.value is None:
            return f'{sheet_var}.getRange("{self.cell_range.range}").clear(Excel.ClearApplyTo.contents);'
        formula_json = json.dumps([[self.value]])
        return f'{sheet_var}.getRange("{self.cell_range.range}").formulas = {formula_json};'

    def to_openpyxl(self, sheet_var: str = "ws") -> str:
        if self.value is None:
            return f'{sheet_var}["{self.cell_range.range}"].value = None'
        formula_str = json.dumps(self.value)
        return f'{sheet_var}["{self.cell_range.range}"].value = {formula_str}'

    def to_xlwings(self, sheet_var: str = "sheet") -> str:
        if self.value is None:
            return f'{sheet_var}["{self.cell_range.range}"].value = None'
        formula_str = json.dumps(self.value)
        return f'{sheet_var}["{self.cell_range.range}"].formula = {formula_str}'

    @classmethod
    def from_symbolic(cls, symbolic: str) -> 'SetFormula':
        parts = symbolic.split(' | ', 2)
        cell_range = CellRange.from_string(parts[1])

        if len(parts) > 2 and parts[2] == 'clear':
            return cls(cell_range=cell_range, value=None, is_inverse=True)

        value = None if parts[2] == "null" else parts[2]

        # LLMs sometimes JSON-encode formula values (e.g. "=SUM(A1:B1)")
        # Strip the encoding to get the raw formula string.
        if value and value.startswith('"') and value.endswith('"'):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, str):
                    value = parsed
            except (json.JSONDecodeError, ValueError):
                pass

        return cls(cell_range=cell_range, value=value, is_inverse=False)

    def apply_to_state(self, state: Dict[str, Any]) -> None:
        """Apply SetFormula to state."""
        cell = _ensure_cell(state, self.cell_range.sheet, self.cell_range.range)

        if self.value is None:
            cell.pop("formula", None)
        else:
            cell["formula"] = self.value

    def get_inverse(self) -> 'Operation':
        """Return operation to clear this cell formula."""
        return SetValue(cell_range=self.cell_range, value=None, is_inverse=True)


class SetInput(Operation):
    """A generic input operation (value or formula) - supports 2D arrays for ranges."""

    @staticmethod
    def _process_datetime_entry(value: Any) -> Any:
        """Recursively convert ISO datetime/time strings within nested arrays."""
        if isinstance(value, list):
            return [SetInput._process_datetime_entry(item) for item in value]
        if isinstance(value, str):
            try:
                return f"{datetime.fromisoformat(value)}"
            except (ValueError, TypeError):
                pass
            try:
                return f"{time.fromisoformat(value)}"
            except (ValueError, TypeError):
                pass
        return value

    @model_validator(mode='before')
    @classmethod
    def process_datetime_values(cls, data: Any) -> Any:
        """Normalize datetime/time strings before Pydantic validation."""
        if isinstance(data, dict) and data.get('value') is not None:
            data['value'] = cls._process_datetime_entry(data['value'])
        return data

    @model_validator(mode='before')
    @classmethod
    def validate_and_adjust_value(cls, data: Any) -> Any:
        """Validate and adjust value to match range dimensions."""
        if isinstance(data, dict):
            cell_range = data.get('cell_range')
            value = data.get('value')
            is_inverse = data.get('is_inverse', False)

            # For inverse operations, keep value=None as-is
            if is_inverse and value is None:
                return data

            if cell_range and value is not None:
                target_rows, target_cols = cell_range.get_dimensions()

                # Single value: expand to fill range or keep as-is for single cell
                if not isinstance(value, list):
                    data['value'] = value if (target_rows == 1 and target_cols == 1) else \
                        [[value] * target_cols for _ in range(target_rows)]
                    return data

                # 2D array: validate dimensions
                if all(isinstance(row, list) for row in value):
                    actual_rows = len(value)
                    actual_cols = len(value[0]) if value else 0
                    if not all(len(row) == actual_cols for row in value):
                        raise ValueError("All rows in 2D array must have the same length")
                    if (actual_rows, actual_cols) != (target_rows, target_cols):
                        raise ValueError(
                            f"Array dimensions {actual_rows}x{actual_cols} don't match "
                            f"range dimensions {target_rows}x{target_cols}"
                        )
                    return data

                # 1D array: convert to row or column if dimensions match
                if len(value) == target_cols and target_rows == 1:
                    data['value'] = [value]
                    return data
                if len(value) == target_rows and target_cols == 1:
                    data['value'] = [[item] for item in value]
                    return data

                raise ValueError(
                    f"1D array length {len(value)} doesn't match "
                    f"range dimensions {target_rows}x{target_cols}"
                )
        return data

    def to_symbolic(self) -> str:
        if self.value is None:
            return f"INPUT | {self.cell_range} | clear"
        value_str = json.dumps(self.value)
        return f"INPUT | {self.cell_range} | {value_str}"

    def to_officejs(self, sheet_var: str = "sheet") -> str:
        if self.value == [] or self.value is None:
            return f'{sheet_var}.getRange("{self.cell_range.range}").clear(Excel.ClearApplyTo.contents);'

        if isinstance(self.value, list) and all(isinstance(row, list) for row in self.value):
            values_json = json.dumps(self.value)
        elif isinstance(self.value, list):
            values_json = json.dumps([self.value])
        else:
            values_json = json.dumps([[self.value]])
        return f'{sheet_var}.getRange("{self.cell_range.range}").values = {values_json};'

    def to_openpyxl(self, sheet_var: str = "ws") -> str:
        if self.value == [] or self.value is None:
            return f'{sheet_var}["{self.cell_range.range}"].value = None'

        if isinstance(self.value, list) and all(isinstance(row, list) for row in self.value):
            processed = [[_process_value_for_openpyxl(cell) for cell in row] for row in self.value]
            values_str = str(processed).replace("'", "")
            return f'{sheet_var}["{self.cell_range.range}"].value = {values_str}'
        elif isinstance(self.value, list):
            processed = [_process_value_for_openpyxl(cell) for cell in self.value]
            values_str = str(processed).replace("'", "")
            return f'{sheet_var}["{self.cell_range.range}"].value = {values_str}'
        else:
            value_str = _process_value_for_openpyxl(self.value)
            return f'{sheet_var}["{self.cell_range.range}"].value = {value_str}'

    def to_xlwings(self, sheet_var: str = "sheet") -> str:
        if self.value == [] or self.value is None:
            return f'{sheet_var}["{self.cell_range.range}"].value = None'

        if isinstance(self.value, list):
            values_str = json.dumps(self.value)
            return f'{sheet_var}["{self.cell_range.range}"].value = {values_str}'
        else:
            if isinstance(self.value, str):
                value_str = json.dumps(self.value)
            else:
                value_str = str(self.value)
            return f'{sheet_var}["{self.cell_range.range}"].value = {value_str}'

    @classmethod
    def from_symbolic(cls, symbolic: str) -> 'SetInput':
        parts = symbolic.split(' | ', 2)
        cell_range = CellRange.from_string(parts[1])

        if len(parts) > 2 and parts[2] == 'clear':
            return cls(cell_range=cell_range, value=None, is_inverse=True)

        try:
            value = json.loads(parts[2])
        except json.JSONDecodeError:
            value = parts[2]
        return cls(cell_range=cell_range, value=value, is_inverse=False)

    def apply_to_state(self, state: Dict[str, Any]) -> None:
        """Apply SetInput to state."""
        start_row, start_col, end_row, end_col = self.cell_range.get_coordinates()
        value = self.value

        # If value is None or empty array, delete values
        if value is None or value == []:
            for cell_addr in _get_cells_in_range(self.cell_range):
                cell = _ensure_cell(state, self.cell_range.sheet, cell_addr)
                cell.pop("value", None)
            return

        # Handle 2D array
        if isinstance(value, list) and all(isinstance(row, list) for row in value):
            for row_offset, row_data in enumerate(value):
                for col_offset, cell_value in enumerate(row_data):
                    row = start_row + row_offset
                    col = start_col + col_offset
                    if row <= end_row and col <= end_col:
                        cell_addr = get_cell_address(row, col)
                        cell = _ensure_cell(state, self.cell_range.sheet, cell_addr)
                        if cell_value is None:
                            cell.pop("value", None)
                        else:
                            cell["value"] = cell_value

        # Handle 1D array
        elif isinstance(value, list):
            if end_row == start_row:  # Row vector
                for col_offset, cell_value in enumerate(value):
                    col = start_col + col_offset
                    if col <= end_col:
                        cell_addr = get_cell_address(start_row, col)
                        cell = _ensure_cell(state, self.cell_range.sheet, cell_addr)
                        if cell_value is None:
                            cell.pop("value", None)
                        else:
                            cell["value"] = cell_value
            else:  # Column vector
                for row_offset, cell_value in enumerate(value):
                    row = start_row + row_offset
                    if row <= end_row:
                        cell_addr = get_cell_address(row, start_col)
                        cell = _ensure_cell(state, self.cell_range.sheet, cell_addr)
                        if cell_value is None:
                            cell.pop("value", None)
                        else:
                            cell["value"] = cell_value

        # Handle single value
        else:
            for cell_addr in _get_cells_in_range(self.cell_range):
                cell = _ensure_cell(state, self.cell_range.sheet, cell_addr)
                cell["value"] = value

    def get_inverse(self) -> 'Operation':
        """Return operation to clear this range's values."""
        return SetInput(cell_range=self.cell_range, value=None, is_inverse=True)

    def breakdown(self) -> List['SetInput']:
        """Break down operation into individual cell SetInput operations."""
        operations = []
        start_row, start_col, end_row, end_col = self.cell_range.get_coordinates()
        value = self.value

        # Handle 2D array
        if isinstance(value, list) and all(isinstance(row, list) for row in value):
            for row_offset, row_data in enumerate(value):
                for col_offset, cell_value in enumerate(row_data):
                    row = start_row + row_offset
                    col = start_col + col_offset
                    if row <= end_row and col <= end_col:
                        cell_addr = get_cell_address(row, col)
                        cell_range = CellRange(sheet=self.cell_range.sheet, range=cell_addr)
                        operations.append(SetInput(cell_range=cell_range, value=cell_value))

        # Handle 1D array
        elif isinstance(value, list):
            if end_row == start_row:  # Row vector
                for col_offset, cell_value in enumerate(value):
                    col = start_col + col_offset
                    if col <= end_col:
                        cell_addr = get_cell_address(start_row, col)
                        cell_range = CellRange(sheet=self.cell_range.sheet, range=cell_addr)
                        operations.append(SetInput(cell_range=cell_range, value=cell_value))
            else:  # Column vector
                for row_offset, cell_value in enumerate(value):
                    row = start_row + row_offset
                    if row <= end_row:
                        cell_addr = get_cell_address(row, start_col)
                        cell_range = CellRange(sheet=self.cell_range.sheet, range=cell_addr)
                        operations.append(SetInput(cell_range=cell_range, value=cell_value))

        # Handle single value
        else:
            for cell_addr in _get_cells_in_range(self.cell_range):
                cell_range = CellRange(sheet=self.cell_range.sheet, range=cell_addr)
                operations.append(SetInput(cell_range=cell_range, value=value))

        return operations
