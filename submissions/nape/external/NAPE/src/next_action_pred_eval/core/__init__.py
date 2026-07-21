"""
Core module - Fundamental primitives for spreadsheet operations.
"""

from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.defaults import EXCEL_DEFAULTS
from next_action_pred_eval.core.state import StateBuilder
from next_action_pred_eval.core.symbolic import (
    operations_to_symbolic,
    symbolic_to_operations,
    compress_symbolic,
    uncompress_symbolic_inputs,
    parse_symbolic,
    format_symbolic,
)
from next_action_pred_eval.core.corrections import CorrectionGenerator, PropertyDifference

# Transforms
from next_action_pred_eval.core.transforms import (
    SymbolicTransform,
    RelativeRangeTransform,
    RelativeFormulaTransform,
    ValueLookupTransform,
    build_transforms,
)

# Import all operations
from next_action_pred_eval.core.operations import (
    SetValue,
    SetFormula,
    SetInput,
    SetFillColor,
    SetFontProperty,
    SetAlignment,
    SetBorder,
    MergeCells,
    SetNumberFormat,
    SetWrapText,
    SetTextOrientation,
    PasteFrom,
    OPERATION_MAP,
    OPERATION_ORDER,
    OPERATION_ORDER_DICT,
    OPERATION_DOCS,
    PRIME_VISIBLE_OPS,
)

__all__ = [
    # Base classes
    "CellRange",
    "Operation",
    "EXCEL_DEFAULTS",
    # State management
    "StateBuilder",
    # Symbolic conversion
    "operations_to_symbolic",
    "symbolic_to_operations",
    "compress_symbolic",
    "uncompress_symbolic_inputs",
    "parse_symbolic",
    "format_symbolic",
    # Corrections
    "CorrectionGenerator",
    "PropertyDifference",
    # Operations
    "SetValue",
    "SetFormula",
    "SetInput",
    "SetFillColor",
    "SetFontProperty",
    "SetAlignment",
    "SetBorder",
    "MergeCells",
    "SetNumberFormat",
    "SetWrapText",
    "SetTextOrientation",
    "PasteFrom",
    # Mappings
    "OPERATION_MAP",
    "OPERATION_ORDER",
    "OPERATION_ORDER_DICT",
    "OPERATION_DOCS",
    "PRIME_VISIBLE_OPS",
    # Transforms
    "SymbolicTransform",
    "RelativeRangeTransform",
    "RelativeFormulaTransform",
    "ValueLookupTransform",
    "build_transforms",
]
