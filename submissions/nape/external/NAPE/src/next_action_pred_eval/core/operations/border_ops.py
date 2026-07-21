"""
Border operations - SetBorder operation.
"""

import json
from typing import Any, Dict, List

from pydantic import model_validator

from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.defaults import EXCEL_DEFAULTS
from next_action_pred_eval.utils.cell_utils import expand_range, get_cell_address
from next_action_pred_eval.core.operations._helpers import (
    _ensure_cell,
    _ensure_format,
)


# ============= Operation Class =============

class SetBorder(Operation):
    """Set cell border."""
    side: str  # left, right, top, bottom, outside, all, inside_horizontal, inside_vertical

    @model_validator(mode='before')
    @classmethod
    def construct_border_value(cls, data: Any) -> Any:
        """Construct the value dict from border properties."""
        if isinstance(data, dict) and 'weight' in data:
            weight = data.pop('weight', None)
            lineStyle = data.pop('lineStyle', None)
            color = data.pop('color', None)
            data['value'] = {"weight": weight, "style": lineStyle, "color": color}
        return data

    def to_symbolic(self) -> str:
        if self.value is None or self.value.get('weight') is None:
            return f"BORDER_{self.side.upper()} | {self.cell_range} | clear"
        style_part = f", {self.value['style']}" if self.value.get('style') else ""
        color_part = f", {self.value['color']}" if self.value.get('color') else ""
        return f"BORDER_{self.side.upper()} | {self.cell_range} | {self.value['weight']}{style_part}{color_part}"

    def to_officejs(self, sheet_var: str = "sheet") -> str:
        return self._generate_border_officejs(sheet_var, self.cell_range.range, self.side, self.value)

    def _generate_border_officejs(self, sheet_var: str, range_str: str, side: str, border_props: dict) -> str:
        """Generate Office.js code for border operations with support for outside/all/inside.*"""
        def apply_to(edge_name: str) -> list:
            lines = []
            if border_props.get('weight'):
                lines.append(f'{sheet_var}.getRange("{range_str}").format.borders.getItem("{edge_name}").weight = "{border_props["weight"]}";')
            if border_props.get('style'):
                lines.append(f'{sheet_var}.getRange("{range_str}").format.borders.getItem("{edge_name}").style = "{border_props["style"]}";')
            if border_props.get('color'):
                lines.append(f'{sheet_var}.getRange("{range_str}").format.borders.getItem("{edge_name}").color = "{border_props["color"]}";')
            return lines

        side_map = {
            'left': 'EdgeLeft',
            'right': 'EdgeRight',
            'top': 'EdgeTop',
            'bottom': 'EdgeBottom',
            'inside_horizontal': 'InsideHorizontal',
            'inside_vertical': 'InsideVertical',
        }

        if side == 'outside':
            return f'setOutsideBorders({sheet_var}.getRange("{range_str}"), "{border_props["weight"]}", "{border_props.get("style", "")}", {json.dumps(border_props.get("color"))});'
        elif side == 'all':
            return f'setAllBorders({sheet_var}.getRange("{range_str}"), "{border_props["weight"]}", "{border_props.get("style", "")}", {json.dumps(border_props.get("color"))});'
        else:
            edge = side_map.get(side, side)
            code_lines = apply_to(edge)
            return "\n    ".join(code_lines)

    def to_openpyxl(self, sheet_var: str = "ws") -> str:
        style = SetBorder._map_border_to_openpyxl(self.value.get('weight'), self.value.get('style'))
        color = self.value.get('color', '000000').lstrip('#') if self.value.get('color') else '000000'

        if self.side in ['left', 'right', 'top', 'bottom']:
            return f'{sheet_var}["{self.cell_range.range}"].border = Border({self.side}=Side(style="{style}", color="{color}"))'
        elif self.side == 'outside':
            return f'{sheet_var}["{self.cell_range.range}"].border = Border(left=Side(style="{style}", color="{color}"), right=Side(style="{style}", color="{color}"), top=Side(style="{style}", color="{color}"), bottom=Side(style="{style}", color="{color}"))'
        elif self.side == 'all':
            return f'{sheet_var}["{self.cell_range.range}"].border = Border(left=Side(style="{style}", color="{color}"), right=Side(style="{style}", color="{color}"), top=Side(style="{style}", color="{color}"), bottom=Side(style="{style}", color="{color}"))'
        elif self.side == 'inside_horizontal':
            return f'# inside_horizontal borders require range iteration'
        elif self.side == 'inside_vertical':
            return f'# inside_vertical borders require range iteration'

        return f'# Unsupported border side: {self.side}'

    def to_xlwings(self, sheet_var: str = "sheet") -> str:
        """Generate xlwings code for setting borders."""
        style_map = {
            'Continuous': 'LineStyle.xlContinuous',
            'Dash': 'LineStyle.xlDash',
            'DashDot': 'LineStyle.xlDashDot',
            'DashDotDot': 'LineStyle.xlDashDotDot',
            'Dot': 'LineStyle.xlDot',
            'Double': 'LineStyle.xlDouble',
            'SlantDashDot': 'LineStyle.xlSlantDashDot'
        }

        weight_map = {
            'Hairline': 1,
            'Thin': 2,
            'Medium': -4138,
            'Thick': 4
        }

        side_map = {
            'left': 'BordersIndex.xlEdgeLeft',
            'right': 'BordersIndex.xlEdgeRight',
            'top': 'BordersIndex.xlEdgeTop',
            'bottom': 'BordersIndex.xlEdgeBottom',
        }

        style_val = style_map.get(self.value.get('style', 'Continuous'), 'LineStyle.xlContinuous')
        weight_val = weight_map.get(self.value.get('weight', 'Thin'), 2)
        color_val = f'"{self.value.get("color", "")}"' if self.value.get('color') else 'None'

        if self.side in side_map:
            side_const = side_map[self.side]
            return f'set_border_side({sheet_var}["{self.cell_range.range}"], {side_const}, {style_val}, {weight_val}, {color_val})'
        elif self.side == 'outside':
            return f'set_border_outside({sheet_var}["{self.cell_range.range}"], {style_val}, {weight_val}, {color_val})'
        elif self.side == 'all':
            return f'set_border_all({sheet_var}["{self.cell_range.range}"], {style_val}, {weight_val}, {color_val})'
        elif self.side == 'inside_horizontal':
            return f'set_border_side({sheet_var}["{self.cell_range.range}"], BordersIndex.xlInsideHorizontal, {style_val}, {weight_val}, {color_val})'
        elif self.side == 'inside_vertical':
            return f'set_border_side({sheet_var}["{self.cell_range.range}"], BordersIndex.xlInsideVertical, {style_val}, {weight_val}, {color_val})'
        else:
            return f'# Unsupported border side: {self.side}'

    @staticmethod
    def _map_border_to_openpyxl(weight: str, lineStyle: str) -> str:
        """Map canonical border style names to openpyxl styles."""
        style_map = {
            ("Hairline", "Continuous"): "hair",
            ("Thin", "Continuous"): "thin",
            ("Medium", "Continuous"): "medium",
            ("Thick", "Continuous"): "thick",
            ("Thin", "Dot"): "dotted",
            ("Thin", "Dash"): "dashed",
            ("Thin", "DashDot"): "dashdot",
            ("Thin", "DashDotDot"): "dashdotdot",
            ("Thin", "Double"): "double",
            ("Medium", "Dash"): "mediumdashed",
            ("Medium", "DashDot"): "mediumdashdot",
            ("Medium", "DashDotDot"): "mediumdashdotdot",
            ("Thin", "SlantDashDot"): "slantdashdot",
        }
        return style_map.get((weight, lineStyle), "thin")

    @classmethod
    def from_symbolic(cls, symbolic: str) -> 'SetBorder':
        parts = symbolic.split(' | ', 2)
        side = parts[0].replace('BORDER_', '').lower()
        cell_range = CellRange.from_string(parts[1])

        if len(parts) > 2 and parts[2].strip() in ('clear', 'None'):
            return cls(
                cell_range=cell_range,
                side=side,
                weight=EXCEL_DEFAULTS["border_weight"],
                lineStyle=EXCEL_DEFAULTS["border_style"],
                color=EXCEL_DEFAULTS["border_color"],
                value=None,
                is_inverse=True
            )

        border_parts = parts[2].split(', ')
        weight = border_parts[0] if len(border_parts) > 0 else 'Thin'
        line_style = border_parts[1] if len(border_parts) > 1 else 'Continuous'
        color = border_parts[2] if len(border_parts) > 2 else None
        return cls(cell_range=cell_range, side=side, weight=weight, lineStyle=line_style, color=color, value=None, is_inverse=False)

    def apply_to_state(self, state: Dict[str, Any]) -> None:
        """Apply SetBorder to state."""
        style_map = {
            "Continuous": "thin", "Dash": "dashed", "DashDot": "dashdot",
            "DashDotDot": "dashdotdot", "Dot": "dotted", "Double": "double",
            "SlantDashDot": "slantdashdot",
        }
        lineStyle = style_map.get(self.value.get('style', 'Continuous'), "thin")
        color = self.value.get('color', '#000000')

        is_clearing = (
            self.value.get('style', 'Continuous') == EXCEL_DEFAULTS["border_style"] and
            color == EXCEL_DEFAULTS["border_color"]
        )

        start_row, start_col, end_row, end_col = self.cell_range.get_coordinates()

        for row, col in expand_range(self.cell_range.range):
            cell_addr = get_cell_address(row, col)
            cell = _ensure_cell(state, self.cell_range.sheet, cell_addr)
            fmt = _ensure_format(cell)
            if "borders" not in fmt:
                fmt["borders"] = {}

            side = self.side.lower()
            if side == "all":
                for s in ["left", "right", "top", "bottom"]:
                    if is_clearing:
                        fmt["borders"].pop(s, None)
                    else:
                        fmt["borders"][s] = {"lineStyle": lineStyle, "color": color}
            elif side == "outside":
                if col == start_col:
                    if is_clearing:
                        fmt["borders"].pop("left", None)
                    else:
                        fmt["borders"]["left"] = {"lineStyle": lineStyle, "color": color}
                if col == end_col:
                    if is_clearing:
                        fmt["borders"].pop("right", None)
                    else:
                        fmt["borders"]["right"] = {"lineStyle": lineStyle, "color": color}
                if row == start_row:
                    if is_clearing:
                        fmt["borders"].pop("top", None)
                    else:
                        fmt["borders"]["top"] = {"lineStyle": lineStyle, "color": color}
                if row == end_row:
                    if is_clearing:
                        fmt["borders"].pop("bottom", None)
                    else:
                        fmt["borders"]["bottom"] = {"lineStyle": lineStyle, "color": color}
            elif side == "left":
                if col == start_col:
                    if is_clearing:
                        fmt["borders"].pop("left", None)
                    else:
                        fmt["borders"]["left"] = {"lineStyle": lineStyle, "color": color}
            elif side == "right":
                if col == end_col:
                    if is_clearing:
                        fmt["borders"].pop("right", None)
                    else:
                        fmt["borders"]["right"] = {"lineStyle": lineStyle, "color": color}
            elif side == "top":
                if row == start_row:
                    if is_clearing:
                        fmt["borders"].pop("top", None)
                    else:
                        fmt["borders"]["top"] = {"lineStyle": lineStyle, "color": color}
            elif side == "bottom":
                if row == end_row:
                    if is_clearing:
                        fmt["borders"].pop("bottom", None)
                    else:
                        fmt["borders"]["bottom"] = {"lineStyle": lineStyle, "color": color}
            elif side == "inside_horizontal":
                # Inside horizontal borders: bottom border on every row except the last
                if row != end_row:
                    if is_clearing:
                        fmt["borders"].pop("bottom", None)
                    else:
                        fmt["borders"]["bottom"] = {"lineStyle": lineStyle, "color": color}
            elif side == "inside_vertical":
                # Inside vertical borders: right border on every column except the last
                if col != end_col:
                    if is_clearing:
                        fmt["borders"].pop("right", None)
                    else:
                        fmt["borders"]["right"] = {"lineStyle": lineStyle, "color": color}
            else:
                # Handles diagonal_down, diagonal_up, and any other sides
                if is_clearing:
                    fmt["borders"].pop(side, None)
                else:
                    fmt["borders"][side] = {"lineStyle": lineStyle, "color": color}

    def get_inverse(self) -> 'Operation':
        """Return operation to remove/clear border."""
        return SetBorder(
            cell_range=self.cell_range,
            side=self.side,
            weight=EXCEL_DEFAULTS["border_weight"],
            lineStyle=EXCEL_DEFAULTS["border_style"],
            color=EXCEL_DEFAULTS["border_color"],
            value=None,
            is_inverse=True
        )

    @property
    def modifies_format(self) -> bool:
        return True
