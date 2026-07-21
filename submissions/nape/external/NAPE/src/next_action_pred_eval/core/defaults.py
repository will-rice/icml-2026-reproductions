"""
Excel default values - Constants for spreadsheet defaults.
"""

# Excel default values based on standard Excel defaults
EXCEL_DEFAULTS = {
    # Font defaults
    "font_name": "Calibri",
    "font_size": 11,
    "font_color": "#000000",  # Black
    "font_bold": False,
    "font_italic": False,
    "font_underline": "none",
    "font_strikethrough": False,

    # Alignment defaults
    "horizontal_alignment": "General",
    "vertical_alignment": "Bottom",
    "text_orientation": 0,
    "wrap_text": False,
    "indent_level": 0,

    # Fill defaults
    "fill_color": None,  # No fill / transparent

    # Border defaults
    "border_style": "None",
    "border_weight": None,
    "border_color": None,

    # Number format default
    "number_format": "General",

    # Other defaults
    "shrink_to_fit": False,
    "reading_order": "Context",
    "protection_locked": True,
    "protection_hidden": False,
}
