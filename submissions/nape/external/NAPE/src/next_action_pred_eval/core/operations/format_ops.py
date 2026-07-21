"""
Format operations - SetFillColor, SetFontProperty, SetAlignment operations.
"""

import json
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

class SetFillColor(Operation):
    """Set cell fill color (use None to clear fill)."""

    def to_symbolic(self) -> str:
        value_str = "clear" if self.value is None else self.value
        return f"FILL_COLOR | {self.cell_range} | {value_str}"

    def to_officejs(self, sheet_var: str = "sheet") -> str:
        if self.value is None:
            return f'{sheet_var}.getRange("{self.cell_range.range}").format.fill.clear();'
        return f'{sheet_var}.getRange("{self.cell_range.range}").format.fill.color = "{self.value}";'

    def to_openpyxl(self, sheet_var: str = "ws") -> str:
        if self.value is None:
            return f'{sheet_var}["{self.cell_range.range}"].fill = PatternFill(fill_type=None)'
        color = self.value.lstrip('#')
        return f'{sheet_var}["{self.cell_range.range}"].fill = PatternFill(start_color="{color}", end_color="{color}", fill_type="solid")'

    def to_xlwings(self, sheet_var: str = "sheet") -> str:
        if self.value is None:
            return f'{sheet_var}["{self.cell_range.range}"].color = None'
        return f'{sheet_var}["{self.cell_range.range}"].color = "{self.value}"'

    @classmethod
    def from_symbolic(cls, symbolic: str) -> 'SetFillColor':
        parts = symbolic.split(' | ', 2)
        cell_range = CellRange.from_string(parts[1])

        raw = parts[2]
        if raw == "clear":
            value = None
        elif raw.startswith('#'):
            # Normalize hex color to uppercase for consistent state comparison
            value = raw.upper()
        else:
            value = raw
        return cls(cell_range=cell_range, value=value, is_inverse=False)

    def apply_to_state(self, state: Dict[str, Any]) -> None:
        """Apply SetFillColor to state."""
        for cell_addr in _get_cells_in_range(self.cell_range):
            cell = _ensure_cell(state, self.cell_range.sheet, cell_addr)
            fmt = _ensure_format(cell)
            if self.value is None:
                fmt.pop("fill", None)
            else:
                if "fill" not in fmt:
                    fmt["fill"] = {}
                fmt["fill"]["patternType"] = "solid"
                fmt["fill"]["fgColor"] = self.value

    def get_inverse(self) -> 'Operation':
        """Return operation to clear fill color."""
        return SetFillColor(cell_range=self.cell_range, value=None, is_inverse=True)

    @property
    def modifies_format(self) -> bool:
        return True


class SetFontProperty(Operation):
    """Set font properties (bold, italic, size, color, name, etc.)."""
    property: str

    def to_symbolic(self) -> str:
        val = str(self.value).lower() if isinstance(self.value, bool) else self.value
        # Emit integer string for whole-number font sizes (TS emits "14" not "14.0")
        if self.property.lower() == 'size' and isinstance(val, float) and val.is_integer():
            val = int(val)
        return f"FONT_{self.property.upper()} | {self.cell_range} | {val}"

    def to_officejs(self, sheet_var: str = "sheet") -> str:
        prop_map = {
            'bold': 'bold',
            'italic': 'italic',
            'size': 'size',
            'color': 'color',
            'underline': 'underline',
            'name': 'name'
        }

        prop = prop_map.get(self.property.lower(), self.property.lower())

        if self.property.lower() in ['bold', 'italic']:
            value = 'true' if self.value else 'false'
        elif self.property.lower() == 'color':
            value = f'"{self.value}"'
        elif self.property.lower() == 'name':
            value = f'"{self.value}"'
        elif self.property.lower() == 'underline':
            if isinstance(self.value, bool):
                value = '"Single"' if self.value else '"None"'
            else:
                value = f'"{self.value}"'
        else:
            value = self.value

        return f'{sheet_var}.getRange("{self.cell_range.range}").format.font.{prop} = {value};'

    def to_openpyxl(self, sheet_var: str = "ws") -> str:
        prop = self.property.lower()

        if prop == 'bold':
            return f'{sheet_var}["{self.cell_range.range}"].font = Font(bold={self.value})'
        elif prop == 'italic':
            return f'{sheet_var}["{self.cell_range.range}"].font = Font(italic={self.value})'
        elif prop == 'size':
            return f'{sheet_var}["{self.cell_range.range}"].font = Font(size={self.value})'
        elif prop == 'color':
            color = self.value.lstrip("#")
            return f'{sheet_var}["{self.cell_range.range}"].font = Font(color="{color}")'
        elif prop == 'name':
            return f'{sheet_var}["{self.cell_range.range}"].font = Font(name="{self.value}")'
        elif prop == 'underline':
            underline_val = '"single"' if self.value else '"none"'
            return f'{sheet_var}["{self.cell_range.range}"].font = Font(underline={underline_val})'
        return f'# Unsupported font property: {self.property}'

    def to_xlwings(self, sheet_var: str = "sheet") -> str:
        prop = self.property.lower()

        if prop in ['bold', 'italic']:
            value = 'True' if self.value else 'False'
            return f'{sheet_var}["{self.cell_range.range}"].font.{prop} = {value}'
        elif prop == 'size':
            return f'{sheet_var}["{self.cell_range.range}"].font.size = {self.value}'
        elif prop == 'color':
            if isinstance(self.value, str) and self.value.startswith('#'):
                return f'{sheet_var}["{self.cell_range.range}"].font.color = "{self.value}"'
            else:
                return f'{sheet_var}["{self.cell_range.range}"].font.color = "{self.value}"'
        elif prop == 'name':
            return f'{sheet_var}["{self.cell_range.range}"].font.name = "{self.value}"'
        elif prop == 'underline':
            if self.value in [True, 'single', 'Single']:
                underline_const = 'UnderlineStyle.xlUnderlineStyleSingle'
            elif self.value == 'double':
                underline_const = 'UnderlineStyle.xlUnderlineStyleDouble'
            elif self.value == 'singleAccounting':
                underline_const = 'UnderlineStyle.xlUnderlineStyleSingleAccounting'
            elif self.value == 'doubleAccounting':
                underline_const = 'UnderlineStyle.xlUnderlineStyleDoubleAccounting'
            else:
                underline_const = 'UnderlineStyle.xlUnderlineStyleNone'
            return f'{sheet_var}["{self.cell_range.range}"].api.Font.Underline = {underline_const}'
        return f'# Unsupported font property: {self.property}'

    @classmethod
    def from_symbolic(cls, symbolic: str) -> 'SetFontProperty':
        parts = symbolic.split(' | ', 2)
        property_name = parts[0].replace('FONT_', '').lower()
        cell_range = CellRange.from_string(parts[1])

        if property_name in ['bold', 'italic']:
            value = parts[2].lower() == 'true'
        elif property_name == 'size':
            if parts[2] == 'None' or parts[2] is None:
                value = EXCEL_DEFAULTS.get('font_size', 11.0)
            else:
                value = float(parts[2])
        elif property_name == 'underline':
            # Normalize to lowercase (TS/Office.js emits PascalCase e.g. "None", "Single")
            value = parts[2].lower()
        elif property_name == 'color':
            # Normalize hex color to uppercase for consistent state comparison
            value = parts[2].upper() if parts[2].startswith('#') else parts[2]
        else:
            value = parts[2]

        return cls(cell_range=cell_range, property=property_name, value=value, is_inverse=False)

    def apply_to_state(self, state: Dict[str, Any]) -> None:
        """Apply SetFontProperty to state."""
        for cell_addr in _get_cells_in_range(self.cell_range):
            cell = _ensure_cell(state, self.cell_range.sheet, cell_addr)
            fmt = _ensure_format(cell)
            if "font" not in fmt:
                fmt["font"] = {}

            default_key = f"font_{self.property}"
            if default_key in EXCEL_DEFAULTS and self.value == EXCEL_DEFAULTS[default_key]:
                fmt["font"].pop(self.property, None)
            else:
                fmt["font"][self.property] = self.value

    def get_inverse(self) -> 'Operation':
        """Return operation to reset font property to default."""
        default_map = {
            'bold': EXCEL_DEFAULTS["font_bold"],
            'italic': EXCEL_DEFAULTS["font_italic"],
            'size': EXCEL_DEFAULTS["font_size"],
            'color': EXCEL_DEFAULTS["font_color"],
            'underline': EXCEL_DEFAULTS["font_underline"],
            'name': EXCEL_DEFAULTS["font_name"],
        }
        default_value = default_map.get(self.property.lower(), None)
        return SetFontProperty(cell_range=self.cell_range, property=self.property, value=default_value, is_inverse=True)

    @property
    def modifies_format(self) -> bool:
        return True


class SetAlignment(Operation):
    """Set cell alignment (horizontal or vertical)."""
    alignment_type: str  # 'horizontal' or 'vertical'

    def to_symbolic(self) -> str:
        return f"ALIGN_{self.alignment_type.upper()} | {self.cell_range} | {self.value}"

    def to_officejs(self, sheet_var: str = "sheet") -> str:
        value_capitalized = self.value.capitalize()
        return f'{sheet_var}.getRange("{self.cell_range.range}").format.{self.alignment_type}Alignment = "{value_capitalized}";'

    def to_openpyxl(self, sheet_var: str = "ws") -> str:
        value_lower = self.value.lower() if isinstance(self.value, str) else self.value
        if self.alignment_type == 'horizontal':
            return f'{sheet_var}["{self.cell_range.range}"].alignment = Alignment(horizontal="{value_lower}")'
        elif self.alignment_type == 'vertical':
            return f'{sheet_var}["{self.cell_range.range}"].alignment = Alignment(vertical="{value_lower}")'
        return f'# Unsupported alignment type: {self.alignment_type}'

    def to_xlwings(self, sheet_var: str = "sheet") -> str:
        if self.alignment_type == 'horizontal':
            align_map = {
                'left': 'HAlign.xlHAlignLeft',
                'center': 'HAlign.xlHAlignCenter',
                'right': 'HAlign.xlHAlignRight',
                'justify': 'HAlign.xlHAlignJustify',
                'general': 'HAlign.xlHAlignGeneral',
                'centeracrossselection': 'HAlign.xlHAlignCenterAcrossSelection',
                'distributed': 'HAlign.xlHAlignDistributed',
                'fill': 'HAlign.xlHAlignFill'
            }
            const_value = align_map.get(self.value.lower(), f'HAlign.xlHAlign{self.value}')
            return f'{sheet_var}["{self.cell_range.range}"].api.HorizontalAlignment = {const_value}'
        elif self.alignment_type == 'vertical':
            align_map = {
                'top': 'VAlign.xlVAlignTop',
                'center': 'VAlign.xlVAlignCenter',
                'middle': 'VAlign.xlVAlignCenter',
                'bottom': 'VAlign.xlVAlignBottom',
                'justify': 'VAlign.xlVAlignJustify',
                'distributed': 'VAlign.xlVAlignDistributed'
            }
            const_value = align_map.get(self.value.lower(), f'VAlign.xlVAlign{self.value}')
            return f'{sheet_var}["{self.cell_range.range}"].api.VerticalAlignment = {const_value}'
        return f'# Unsupported alignment type: {self.alignment_type}'

    @classmethod
    def from_symbolic(cls, symbolic: str) -> 'SetAlignment':
        parts = symbolic.split(' | ', 2)
        align_type = parts[0].replace('ALIGN_', '').lower()
        cell_range = CellRange.from_string(parts[1])

        # Normalize to lowercase (TS/Office.js emits PascalCase e.g. "Left", "Center")
        return cls(cell_range=cell_range, alignment_type=align_type, value=parts[2].lower(), is_inverse=False)

    def apply_to_state(self, state: Dict[str, Any]) -> None:
        """Apply SetAlignment to state."""
        for cell_addr in _get_cells_in_range(self.cell_range):
            cell = _ensure_cell(state, self.cell_range.sheet, cell_addr)
            fmt = _ensure_format(cell)

            if self.alignment_type == "horizontal":
                key = "horizontalAlignment"
                default = EXCEL_DEFAULTS["horizontal_alignment"]
            else:
                key = "verticalAlignment"
                default = EXCEL_DEFAULTS["vertical_alignment"]

            if isinstance(self.value, str) and self.value.lower() == default.lower():
                fmt.pop(key, None)
            else:
                fmt[key] = self.value

    def get_inverse(self) -> 'Operation':
        """Return operation to reset alignment to default."""
        if self.alignment_type == "horizontal":
            default_value = EXCEL_DEFAULTS["horizontal_alignment"]
        else:
            default_value = EXCEL_DEFAULTS["vertical_alignment"]
        return SetAlignment(cell_range=self.cell_range, alignment_type=self.alignment_type, value=default_value, is_inverse=True)

    @property
    def modifies_format(self) -> bool:
        return True
