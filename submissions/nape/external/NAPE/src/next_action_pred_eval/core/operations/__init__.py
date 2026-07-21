"""
Operations module - All spreadsheet operation implementations.

This module exports:
- All operation classes (SetValue, SetFormula, SetInput, etc.)
- OPERATION_MAP: Mapping of symbolic names to operation classes
- OPERATION_ORDER: Predefined order for operations
- OPERATION_DOCS: Documentation for each operation type

Note: SetValue and SetFormula are rarely seen in trajectory data. The sequencing
pipeline's OperationMerger converts them into SetInput (INPUT) ops. They exist
as intermediate representations used during generation (before merging) and are
consumed by transformers like AutoFillDetector that run before the merger.
"""

from next_action_pred_eval.core.operations.value_ops import (
    SetValue,
    SetFormula,
    SetInput,
    get_cells_in_range,
)
from next_action_pred_eval.core.operations.format_ops import (
    SetFillColor,
    SetFontProperty,
    SetAlignment,
)
from next_action_pred_eval.core.operations.border_ops import SetBorder
from next_action_pred_eval.core.operations.cell_ops import (
    MergeCells,
    SetNumberFormat,
)
from next_action_pred_eval.core.operations.text_ops import (
    SetWrapText,
    SetTextOrientation,
)
from next_action_pred_eval.core.operations.paste_ops import PasteFrom
from next_action_pred_eval.core.operations.autofill_ops import AutoFill


# Mapping of symbolic operation names to classes
OPERATION_MAP = {
    'VALUE': SetValue,
    'FORMULA': SetFormula,
    'INPUT': SetInput,
    'NUMBER_FORMAT': SetNumberFormat,
    'FILL_COLOR': SetFillColor,
    'MERGE': MergeCells,
    'UNMERGE': MergeCells,
    'TEXT_ORIENTATION': SetTextOrientation,
    'WRAP_TEXT': SetWrapText,
    'PASTE_FROM': PasteFrom,
    'AUTOFILL': AutoFill,
}

# Add dynamic mappings for font operations
for prop in ['BOLD', 'ITALIC', 'SIZE', 'COLOR', 'UNDERLINE', 'NAME']:
    OPERATION_MAP[f'FONT_{prop}'] = SetFontProperty

# Add dynamic mappings for alignment operations
for align in ['HORIZONTAL', 'VERTICAL']:
    OPERATION_MAP[f'ALIGN_{align}'] = SetAlignment

# Add dynamic mappings for border operations
for side in ['LEFT', 'RIGHT', 'TOP', 'BOTTOM', 'OUTSIDE', 'ALL', 'INSIDE_HORIZONTAL', 'INSIDE_VERTICAL', 'DIAGONAL_DOWN', 'DIAGONAL_UP']:
    OPERATION_MAP[f'BORDER_{side}'] = SetBorder


# Documentation for each operation type
OPERATION_DOCS = {
    'VALUE': {
        'description': 'Writes a literal value into a single cell. Use for strings, numbers, booleans, or nulls.',
        'usage': 'VALUE | Sheet1!B2 | "Hello world"',
    },
    'FORMULA': {
        'description': 'Assigns an Excel formula to the specified cell. Provide the exact formula text, starting with =.',
        'usage': 'FORMULA | Sheet1!C2 | =SUM(A2:B2)',
    },
    'INPUT': {
        'description': 'Fills a range with a 2D array or single scalar. Automatically expands to match the range shape.',
        'usage': 'INPUT | Sheet1!B4:C5 | [["Topic","Owner"],["Math","Alex"]]',
    },
    'NUMBER_FORMAT': {
        'description': 'Applies a built-in or custom number format string.',
        'usage': 'NUMBER_FORMAT | Sheet1!D2:D10 | "#,##0.00"',
    },
    'FILL_COLOR': {
        'description': 'Sets the background fill color for the range using a hex color (e.g., #FF0000).',
        'usage': 'FILL_COLOR | Sheet1!A1:C1 | #F2F2F2',
    },
    'MERGE': {
        'description': 'Merges the target range into a single cell when value is true; use false to unmerge.',
        'usage': 'MERGE | Sheet1!A1:B1 | true',
    },
    'UNMERGE': {
        'description': 'Explicitly unmerges the provided range (same semantics as MERGE with false).',
        'usage': 'UNMERGE | Sheet1!A1:B1 | false',
    },
    'TEXT_ORIENTATION': {
        'description': 'Rotates text within the range. Provide degrees (-90 to 90).',
        'usage': 'TEXT_ORIENTATION | Sheet1!A2:A10 | 90',
    },
    'WRAP_TEXT': {
        'description': 'Enables or disables text wrapping for the range.',
        'usage': 'WRAP_TEXT | Sheet1!C2:C20 | true',
    },
    'PASTE_FROM': {
        'description': 'Copies a source range into the destination. Paste modes: all, values, formats, formulas.',
        'usage': 'PASTE_FROM | Sheet1!D2:E3 | Sheet1!A2:B3 | values',
    },
    'AUTOFILL': {
        'description': 'Extends patterns from a source range into a larger destination range (drag-fill). Direction is inferred from geometry.',
        'usage': 'AUTOFILL | Sheet1!A1:A10 | Sheet1!A1:A3',
    },
}

# Add documentation for font operations
for prop in ['BOLD', 'ITALIC', 'SIZE', 'COLOR', 'UNDERLINE', 'NAME']:
    if prop in {'BOLD', 'ITALIC'}:
        example_value = 'true'
    elif prop == 'SIZE':
        example_value = '12'
    elif prop == 'COLOR':
        example_value = '#000000'
    elif prop == 'UNDERLINE':
        example_value = 'single'
    else:
        example_value = 'Arial'
    OPERATION_DOCS[f'FONT_{prop}'] = {
        'description': f'Sets the {prop.lower()} font attribute across the range.',
        'usage': f'FONT_{prop} | Sheet1!A1:C3 | {example_value}',
    }

# Add documentation for alignment operations
for align in ['HORIZONTAL', 'VERTICAL']:
    target = 'horizontal' if align == 'HORIZONTAL' else 'vertical'
    OPERATION_DOCS[f'ALIGN_{align}'] = {
        'description': f'Applies {target} alignment (e.g., left, center, right, top, middle).',
        'usage': f'ALIGN_{align} | Sheet1!B2:D2 | center',
    }

# Add documentation for border operations
for side in ['LEFT', 'RIGHT', 'TOP', 'BOTTOM', 'OUTSIDE', 'ALL', 'INSIDE_HORIZONTAL', 'INSIDE_VERTICAL', 'DIAGONAL_DOWN', 'DIAGONAL_UP']:
    readable = side.replace('_', ' ').title()
    OPERATION_DOCS[f'BORDER_{side}'] = {
        'description': f'Sets the {readable} border style and color.',
        'usage': f'BORDER_{side} | Sheet1!A1:C3 | Thin, Continuous, #000000',
    }


# Predefined order for operations
OPERATION_ORDER = [
    'INPUT',
    'PASTE_FROM',
    'VALUE',
    'FORMULA',
    'AUTOFILL',
    'MERGE',
    'NUMBER_FORMAT',
    'FONT_NAME',
    'FONT_SIZE',
    'FONT_BOLD',
    'FONT_ITALIC',
    'FONT_UNDERLINE',
    'FONT_COLOR',
    'ALIGN_HORIZONTAL',
    'ALIGN_VERTICAL',
    'TEXT_ORIENTATION',
    'WRAP_TEXT',
    'FILL_COLOR',
    'BORDER_OUTSIDE',
    'BORDER_ALL',
    'BORDER_LEFT',
    'BORDER_RIGHT',
    'BORDER_TOP',
    'BORDER_BOTTOM',
    'BORDER_INSIDE_HORIZONTAL',
    'BORDER_INSIDE_VERTICAL',
    'BORDER_DIAGONAL_DOWN',
    'BORDER_DIAGONAL_UP',
]

# Map operation classes to their order index
OPERATION_ORDER_DICT = {OPERATION_MAP[op]: i for i, op in enumerate(OPERATION_ORDER)}

# Operations that are primarily visible to users
PRIME_VISIBLE_OPS = (SetValue, SetFormula, SetInput, SetBorder, SetFillColor)


__all__ = [
    # Operation classes
    'SetValue',
    'SetFormula',
    'SetInput',
    'SetFillColor',
    'SetFontProperty',
    'SetAlignment',
    'SetBorder',
    'MergeCells',
    'SetNumberFormat',
    'SetWrapText',
    'SetTextOrientation',
    'PasteFrom',
    'AutoFill',
    # Utility functions
    'get_cells_in_range',
    # Mappings and constants
    'OPERATION_MAP',
    'OPERATION_ORDER',
    'OPERATION_ORDER_DICT',
    'OPERATION_DOCS',
    'PRIME_VISIBLE_OPS',
]
