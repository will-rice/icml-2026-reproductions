"""
Excel Parser - Extracts operations from workbook state.

Converts a workbook state dictionary (or .xlsx file via openpyxl) into
a list of Operation objects. This is the reverse of apply_to_state().

Uses openpyxl for .xlsx parsing via ``workbook_to_state()``.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.operations import (
    MergeCells,
    SetAlignment,
    SetBorder,
    SetFillColor,
    SetFontProperty,
    SetFormula,
    SetInput,
    SetNumberFormat,
    SetTextOrientation,
    SetValue,
    SetWrapText,
)
from next_action_pred_eval.core.defaults import EXCEL_DEFAULTS
from next_action_pred_eval.utils.cell_utils import get_cell_address, get_range_string


class ExcelParser:
    """
    Parses workbook state dicts or .xlsx files into Operation lists.

    Usage:
        # From state dict
        parser = ExcelParser()
        ops = parser.parse(state=my_state)

        # From .xlsx file
        ops = parser.parse(filepath="workbook.xlsx")
    """

    def parse(
        self,
        filepath: Optional[Union[str, Path]] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> List[Operation]:
        """
        Parse Excel file or state dict and extract operations.

        Args:
            filepath: Path to .xlsx file (uses openpyxl via workbook_to_state)
            state: Pre-built state dict in standard format

        Returns:
            List of Operation objects
        """
        if state is None and filepath is None:
            raise ValueError("At least one of filepath or state must be provided.")

        if filepath and state is None:
            from next_action_pred_eval.utils.workbook.sheet_to_state import workbook_to_state
            state = workbook_to_state(filepath)

        # Read workbook defaults from state metadata (set by workbook_to_state)
        # to filter inherited formatting that matches the Normal style.
        wb_defaults = state.get("workbook_defaults", {})
        default_font_name = wb_defaults.get("font_name", EXCEL_DEFAULTS["font_name"])
        default_font_size = wb_defaults.get("font_size", EXCEL_DEFAULTS["font_size"])
        default_font_color = wb_defaults.get("font_color", EXCEL_DEFAULTS["font_color"])

        operations = []
        merged_cells: Dict[str, CellRange] = {}

        for sheet_name, sheet_data in state.get("worksheets", {}).items():
            if " | " in sheet_name:
                sheet_name = sheet_name.replace(" | ", "_|_")
            sheet_operations = []

            # Handle merged cells first
            if "worksheetProperties" in sheet_data:
                for merged in sheet_data["worksheetProperties"].get("merged_cells", []):
                    range_str = get_range_string(
                        merged["start_row"],
                        merged["start_col"],
                        merged["end_row"],
                        merged["end_col"],
                    )
                    cell_range = CellRange(sheet=sheet_name, range=range_str)
                    sheet_operations.append(MergeCells(cell_range=cell_range, value=True))

                    first_cell_addr = get_cell_address(merged["start_row"], merged["start_col"])
                    merged_cells[f"{sheet_name}!{first_cell_addr}"] = cell_range

            # Process cells
            for cell_addr, cell_data in sheet_data.get("cells", {}).items():
                cell_range = CellRange(sheet=sheet_name, range=cell_addr)
                cell_key = f"{sheet_name}!{cell_addr}"

                # Value or Formula
                if "formula" in cell_data:
                    sheet_operations.append(SetInput(cell_range=cell_range, value=cell_data["formula"]))
                elif "value" in cell_data and cell_data["value"] is not None:
                    sheet_operations.append(SetInput(cell_range=cell_range, value=cell_data["value"]))

                # Number format — skip if default ("General")
                if "number_format" in cell_data:
                    nf = cell_data["number_format"]
                    if nf and nf != EXCEL_DEFAULTS["number_format"]:
                        sheet_operations.append(
                            SetNumberFormat(cell_range=cell_range, value=nf)
                        )

                # Format properties
                if "Format" in cell_data:
                    format_data = cell_data["Format"]

                    # Font properties
                    if "font" in format_data:
                        font = format_data["font"]
                        if font.get("name"):
                            font_name = font["name"].strip('"').strip("'")
                            # Only skip if it matches the workbook's actual
                            # Normal-style default font (from state metadata).
                            if font_name.lower() != default_font_name.lower():
                                sheet_operations.append(
                                    SetFontProperty(cell_range=cell_range, property="name", value=font_name)
                                )
                        if font.get("bold"):
                            sheet_operations.append(
                                SetFontProperty(cell_range=cell_range, property="bold", value=True)
                            )
                        if font.get("italic"):
                            sheet_operations.append(
                                SetFontProperty(cell_range=cell_range, property="italic", value=True)
                            )
                        if "size" in font and font["size"] != default_font_size:
                            sheet_operations.append(
                                SetFontProperty(cell_range=cell_range, property="size", value=font["size"])
                            )
                        if "color" in font:
                            color = self._extract_color(font["color"])
                            if color and color.upper() != (default_font_color or "").upper():
                                sheet_operations.append(
                                    SetFontProperty(cell_range=cell_range, property="color", value=color)
                                )
                        if font.get("underline") and font["underline"] != EXCEL_DEFAULTS["font_underline"]:
                            sheet_operations.append(
                                SetFontProperty(cell_range=cell_range, property="underline", value=font["underline"])
                            )

                    # Wrap text — skip if default (False)
                    if "wrapText" in format_data and format_data["wrapText"] != EXCEL_DEFAULTS["wrap_text"]:
                        sheet_operations.append(
                            SetWrapText(cell_range=cell_range, value=format_data["wrapText"])
                        )

                    # Fill
                    if "fill" in format_data:
                        fill_color = self._extract_fill_color(format_data["fill"])
                        if fill_color:
                            sheet_operations.append(SetFillColor(cell_range=cell_range, value=fill_color))

                    # Alignment — skip defaults
                    if "horizontalAlignment" in format_data and format_data["horizontalAlignment"] != EXCEL_DEFAULTS["horizontal_alignment"]:
                        sheet_operations.append(
                            SetAlignment(
                                cell_range=cell_range,
                                alignment_type="horizontal",
                                value=format_data["horizontalAlignment"],
                            )
                        )
                    if "verticalAlignment" in format_data and format_data.get("verticalAlignment") != EXCEL_DEFAULTS["vertical_alignment"]:
                        sheet_operations.append(
                            SetAlignment(
                                cell_range=cell_range,
                                alignment_type="vertical",
                                value=format_data["verticalAlignment"],
                            )
                        )

                    # Text orientation — skip if default (0)
                    if "textOrientation" in format_data and format_data["textOrientation"] != EXCEL_DEFAULTS["text_orientation"]:
                        sheet_operations.append(
                            SetTextOrientation(cell_range=cell_range, value=format_data["textOrientation"])
                        )

                    # Borders
                    if "borders" in format_data:
                        for side, border_data in format_data["borders"].items():
                            if isinstance(border_data, dict) and "lineStyle" in border_data:
                                weight, lineStyle = self._map_border_style(border_data["lineStyle"])
                                color = self._extract_color(border_data.get("color"))

                                cell_range_complete = merged_cells.get(cell_key, cell_range)
                                sheet_operations.append(
                                    SetBorder(
                                        cell_range=cell_range_complete,
                                        side=side,
                                        weight=weight,
                                        lineStyle=lineStyle,
                                        color=color,
                                        value=None,
                                    )
                                )
            operations.extend(sheet_operations)

        return operations

    @staticmethod
    def _extract_color(color_data: Any) -> Optional[str]:
        """Extract color from various color formats."""
        if isinstance(color_data, str):
            return color_data
        elif isinstance(color_data, dict):
            if "rgb" in color_data:
                return color_data["rgb"]
            elif "theme" in color_data:
                return None
        return None

    @staticmethod
    def _extract_fill_color(fill_data: Dict) -> Optional[str]:
        """Extract fill color from fill data."""
        if not fill_data.get("patternType", None):
            return None
        if "fgColor" in fill_data:
            return ExcelParser._extract_color(fill_data["fgColor"])
        elif "bgColor" in fill_data:
            return ExcelParser._extract_color(fill_data["bgColor"])
        return None

    @staticmethod
    def _map_border_style(style: str) -> Tuple[str, str]:
        """Map border styles to (weight, lineStyle) tuples."""
        style_map = {
            "hair": ("Hairline", "Continuous"),
            "thin": ("Thin", "Continuous"),
            "medium": ("Medium", "Continuous"),
            "thick": ("Thick", "Continuous"),
            "dotted": ("Thin", "Dot"),
            "dashed": ("Thin", "Dash"),
            "dashdot": ("Thin", "DashDot"),
            "dashdotdot": ("Thin", "DashDotDot"),
            "double": ("Thin", "Double"),
            "mediumdashed": ("Medium", "Dash"),
            "mediumdashdot": ("Medium", "DashDot"),
            "mediumdashdotdot": ("Medium", "DashDotDot"),
            "slantdashdot": ("Thin", "SlantDashDot"),
        }
        return style_map.get(style, ("Thin", "Continuous"))
