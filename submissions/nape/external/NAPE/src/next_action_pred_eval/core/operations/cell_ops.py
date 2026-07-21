"""
Cell operations - MergeCells, SetNumberFormat operations.
"""

import copy
import json
from typing import Any, Dict, List

from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.defaults import EXCEL_DEFAULTS
from next_action_pred_eval.utils.cell_utils import expand_range, get_cell_address
from next_action_pred_eval.core.operations._helpers import (
    _ensure_sheet,
    _ensure_cell,
    _get_cells_in_range,
)


# ============= Operation Classes =============

class SetNumberFormat(Operation):
    """Set number format operation."""

    def to_symbolic(self) -> str:
        return f"NUMBER_FORMAT | {self.cell_range} | {self.value}"

    def to_officejs(self, sheet_var: str = "sheet") -> str:
        format_json = json.dumps([[self.value]])
        return f'{sheet_var}.getRange("{self.cell_range.range}").numberFormat = {format_json};'

    def to_openpyxl(self, sheet_var: str = "ws") -> str:
        format_str = json.dumps(self.value)
        return f'{sheet_var}["{self.cell_range.range}"].number_format = {format_str}'

    def to_xlwings(self, sheet_var: str = "sheet") -> str:
        format_str = json.dumps(self.value)
        return f'{sheet_var}["{self.cell_range.range}"].number_format = {format_str}'

    @classmethod
    def from_symbolic(cls, symbolic: str) -> 'SetNumberFormat':
        parts = symbolic.split(' | ', 2)
        cell_range = CellRange.from_string(parts[1])

        return cls(cell_range=cell_range, value=parts[2], is_inverse=False)

    def apply_to_state(self, state: Dict[str, Any]) -> None:
        """Apply SetNumberFormat to state."""
        for cell_addr in _get_cells_in_range(self.cell_range):
            cell = _ensure_cell(state, self.cell_range.sheet, cell_addr)

            if self.value == EXCEL_DEFAULTS["number_format"]:
                cell.pop("number_format", None)
            else:
                cell["number_format"] = self.value

    def get_inverse(self) -> 'Operation':
        """Return operation to reset number format to General."""
        return SetNumberFormat(cell_range=self.cell_range, value=EXCEL_DEFAULTS["number_format"], is_inverse=True)


class MergeCells(Operation):
    """Merge or unmerge cells operation (value=True to merge, False to unmerge)."""

    def to_symbolic(self) -> str:
        return f"MERGE | {self.cell_range} | {str(self.value).lower()}"

    def to_officejs(self, sheet_var: str = "sheet") -> str:
        if self.value:
            return f'{sheet_var}.getRange("{self.cell_range.range}").merge(false);'
        else:
            return f'{sheet_var}.getRange("{self.cell_range.range}").unmerge();'

    def to_openpyxl(self, sheet_var: str = "ws") -> str:
        if self.value:
            return f'{sheet_var}.merge_cells("{self.cell_range.range}")'
        else:
            return f'{sheet_var}.unmerge_cells("{self.cell_range.range}")'

    def to_xlwings(self, sheet_var: str = "sheet") -> str:
        if self.value:
            return f'{sheet_var}.range("{self.cell_range.range}").merge()'
        else:
            return f'{sheet_var}.range("{self.cell_range.range}").unmerge()'

    @classmethod
    def from_symbolic(cls, symbolic: str) -> 'MergeCells':
        parts = symbolic.split(' | ', 2)
        cell_range = CellRange.from_string(parts[1])

        if len(parts) <= 2:
            # No explicit value — infer from the operation type keyword
            merge = (parts[0] == 'MERGE')
            return cls(cell_range=cell_range, value=merge, is_inverse=False)

        merge = parts[2].lower() == 'true'
        return cls(cell_range=cell_range, value=merge, is_inverse=False)

    def apply_to_state(self, state: Dict[str, Any]) -> None:
        """Apply MergeCells to state - matches Excel's actual merge/unmerge behavior.

        MERGE behavior:
        1. Outer borders: Kept if COMPLETE across edge, cleared if partial. Anchor gets all.
        2. Interior borders on anchor row/col: MOVED to outer edge + anchor keeps copy.
        3. Interior borders elsewhere: CLEARED.
        4. Fill/font/alignment: Only anchor's format kept.

        UNMERGE behavior:
        1. Fill/font/alignment: Propagated from anchor to ALL cells.
        2. Borders: NOT propagated - each cell keeps its own borders from merged state.
        """
        sheet = _ensure_sheet(state, self.cell_range.sheet)
        start_row, start_col, end_row, end_col = self.cell_range.get_coordinates()
        anchor_addr = get_cell_address(start_row, start_col)
        merge_info = {"start_row": start_row, "start_col": start_col, "end_row": end_row, "end_col": end_col}

        if self.value:
            # === MERGE OPERATION ===
            if merge_info in sheet["worksheetProperties"]["merged_cells"]:
                return

            if self.is_inverse:
                sheet["worksheetProperties"]["merged_cells"].append(merge_info)
                return

            sheet_cells = sheet.get("cells", {})

            def get_border(r, c, side):
                return sheet_cells.get(get_cell_address(r, c), {}).get("Format", {}).get("borders", {}).get(side)

            def borders_equal(b1, b2):
                if b1 is None or b2 is None:
                    return b1 is None and b2 is None
                return (b1.get("lineStyle") == b2.get("lineStyle") and
                        b1.get("color") == b2.get("color"))

            left_borders = [get_border(r, start_col, "left") for r in range(start_row, end_row + 1)]
            edge_borders = {
                "left": left_borders[0] if left_borders[0] and all(borders_equal(left_borders[0], b) for b in left_borders) else None,
            }

            right_borders = [get_border(r, end_col, "right") for r in range(start_row, end_row + 1)]
            edge_borders["right"] = right_borders[0] if right_borders[0] and all(borders_equal(right_borders[0], b) for b in right_borders) else None

            top_borders = [get_border(start_row, c, "top") for c in range(start_col, end_col + 1)]
            edge_borders["top"] = top_borders[0] if top_borders[0] and all(borders_equal(top_borders[0], b) for b in top_borders) else None

            bottom_borders = [get_border(end_row, c, "bottom") for c in range(start_col, end_col + 1)]
            edge_borders["bottom"] = bottom_borders[0] if bottom_borders[0] and all(borders_equal(bottom_borders[0], b) for b in bottom_borders) else None

            has_partial_right = any(right_borders) and not edge_borders["right"]
            has_partial_bottom = any(bottom_borders) and not edge_borders["bottom"]

            if edge_borders["right"] is None and end_col > start_col:
                anchor_right = get_border(start_row, start_col, "right")
                if anchor_right and not has_partial_right:
                    edge_borders["right"] = copy.deepcopy(anchor_right)

            if edge_borders["bottom"] is None and end_row > start_row:
                anchor_bottom = get_border(start_row, start_col, "bottom")
                if anchor_bottom and not has_partial_bottom:
                    edge_borders["bottom"] = copy.deepcopy(anchor_bottom)

            anchor_borders = {}
            if edge_borders.get("left"):
                anchor_borders["left"] = copy.deepcopy(edge_borders["left"])
            if edge_borders.get("top"):
                anchor_borders["top"] = copy.deepcopy(edge_borders["top"])
            if start_col == end_col and edge_borders.get("right"):
                anchor_borders["right"] = copy.deepcopy(edge_borders["right"])
            if start_row == end_row and edge_borders.get("bottom"):
                anchor_borders["bottom"] = copy.deepcopy(edge_borders["bottom"])

            for row, col in expand_range(self.cell_range.range):
                cell_addr = get_cell_address(row, col)
                cell = sheet_cells.get(cell_addr)
                is_anchor = (cell_addr == anchor_addr)

                if is_anchor:
                    if cell is None:
                        cell = sheet_cells[cell_addr] = {}
                    if anchor_borders:
                        cell.setdefault("Format", {})["borders"] = anchor_borders
                    elif "Format" in cell:
                        cell["Format"].pop("borders", None)
                        if not cell["Format"]:
                            del cell["Format"]
                else:
                    new_borders = {}
                    if col == start_col and edge_borders["left"]:
                        new_borders["left"] = copy.deepcopy(edge_borders["left"])
                    if col == end_col and edge_borders["right"]:
                        new_borders["right"] = copy.deepcopy(edge_borders["right"])
                    if row == start_row and edge_borders["top"]:
                        new_borders["top"] = copy.deepcopy(edge_borders["top"])
                    if row == end_row and edge_borders["bottom"]:
                        new_borders["bottom"] = copy.deepcopy(edge_borders["bottom"])

                    if cell:
                        cell.clear()
                    if new_borders:
                        sheet_cells[cell_addr] = {"Format": {"borders": new_borders}}
                    elif cell_addr in sheet_cells:
                        del sheet_cells[cell_addr]

            sheet["worksheetProperties"]["merged_cells"].append(merge_info)
        else:
            # === UNMERGE OPERATION ===
            if self.is_inverse:
                if merge_info in sheet["worksheetProperties"]["merged_cells"]:
                    sheet["worksheetProperties"]["merged_cells"].remove(merge_info)
                return

            sheet_cells = sheet["cells"]
            anchor_cell = _ensure_cell(state, self.cell_range.sheet, anchor_addr)
            anchor_format = anchor_cell.get("Format", {})

            anchor_format_no_borders = {k: copy.deepcopy(v) for k, v in anchor_format.items() if k != "borders"}

            anchor_borders = anchor_format.get("borders", {})
            if anchor_borders:
                fixed_anchor_borders = {}
                if "left" in anchor_borders:
                    fixed_anchor_borders["left"] = anchor_borders["left"]
                if "top" in anchor_borders:
                    fixed_anchor_borders["top"] = anchor_borders["top"]
                if start_col == end_col and "right" in anchor_borders:
                    fixed_anchor_borders["right"] = anchor_borders["right"]
                if start_row == end_row and "bottom" in anchor_borders:
                    fixed_anchor_borders["bottom"] = anchor_borders["bottom"]

                if fixed_anchor_borders:
                    anchor_cell["Format"]["borders"] = fixed_anchor_borders
                else:
                    anchor_cell["Format"].pop("borders", None)
                    if not anchor_cell["Format"]:
                        del anchor_cell["Format"]

            for cell_addr in _get_cells_in_range(self.cell_range):
                if cell_addr == anchor_addr:
                    continue

                cell = sheet_cells.get(cell_addr)

                if cell is None:
                    if not anchor_format_no_borders:
                        continue
                    cell = sheet_cells[cell_addr] = {}
                else:
                    for key in list(cell.keys()):
                        if key != "Format":
                            cell.pop(key, None)

                existing_borders = cell.get("Format", {}).get("borders")

                if anchor_format_no_borders:
                    cell["Format"] = copy.deepcopy(anchor_format_no_borders)
                    if existing_borders:
                        cell["Format"]["borders"] = existing_borders
                else:
                    if existing_borders:
                        cell["Format"] = {"borders": existing_borders}
                    else:
                        cell.pop("Format", None)
                        if not cell:
                            sheet_cells.pop(cell_addr, None)

            if merge_info in sheet["worksheetProperties"]["merged_cells"]:
                sheet["worksheetProperties"]["merged_cells"].remove(merge_info)

    def get_inverse(self) -> 'Operation':
        """Return operation to toggle merge/unmerge."""
        return MergeCells(cell_range=self.cell_range, value=not self.value, is_inverse=True)
