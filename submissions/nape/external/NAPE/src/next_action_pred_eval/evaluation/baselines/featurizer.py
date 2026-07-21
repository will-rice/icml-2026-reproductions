"""
Shared featurizer for unified v2 baseline solvers.

Single parsing + feature-extraction module used by all four baseline solvers
(NGram, OnlineNGram, LSTM, XGBoost) in both absolute and relative modes.

Core responsibilities:
  1. Parse symbolic operation strings into structured ParsedOp objects.
  2. Compute per-operation relative features (deltas, movement class).
  3. Classify operation values into categorical types.
  4. Bucket range features for embedding-based models (LSTM).
  5. Reconstruct symbolic strings from predicted features.
  6. Shift formula cell references when predicting formula operations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════ #
#  Constants                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

OP_TYPES: List[str] = [
    "ALIGN_HORIZONTAL", "ALIGN_VERTICAL",
    "BORDER_ALL", "BORDER_BOTTOM", "BORDER_DIAGONAL_DOWN", "BORDER_DIAGONAL_UP",
    "BORDER_INSIDE_HORIZONTAL", "BORDER_INSIDE_VERTICAL",
    "BORDER_LEFT", "BORDER_OUTSIDE", "BORDER_RIGHT", "BORDER_TOP",
    "FILL_COLOR", "FONT_BOLD", "FONT_COLOR", "FONT_ITALIC",
    "FONT_NAME", "FONT_SIZE", "FONT_UNDERLINE",
    "FORMULA", "INPUT", "MERGE",
    "NUMBER_FORMAT", "PASTE_FROM", "TEXT_ORIENTATION", "UNMERGE",
    "VALUE", "WRAP_TEXT",
]
OP_TYPE_TO_ID: Dict[str, int] = {op: i for i, op in enumerate(OP_TYPES)}
NUM_OP_TYPES: int = len(OP_TYPES)
UNKNOWN_OP_ID: int = NUM_OP_TYPES

VALUE_TYPES: List[str] = [
    "true", "false",
    "number", "string", "formula", "list",
    "color", "border_spec",
    "align_h", "align_v",
    "font_name", "font_size",
    "number_format", "orientation",
    "underline_style",
    "null", "other",
]
VALUE_TYPE_TO_ID: Dict[str, int] = {v: i for i, v in enumerate(VALUE_TYPES)}
NUM_VALUE_TYPES: int = len(VALUE_TYPES)

MOVEMENT_CLASSES: List[str] = [
    "same", "right1", "left1", "down1", "up1",
    "right_n", "left_n", "down_n", "up_n",
    "diagonal", "sheet_change",
]
MOVEMENT_CLASS_TO_ID: Dict[str, int] = {m: i for i, m in enumerate(MOVEMENT_CLASSES)}
NUM_MOVEMENT_CLASSES: int = len(MOVEMENT_CLASSES)

# Bucketing constants
RANGE_BUCKETS: int = 101          # 0..100
ABS_CLIP_LOW: int = 1
ABS_CLIP_HIGH: int = 101          # row/col 1→0, 101+→100
REL_CLIP: int = 50                # delta -50→0, 0→50, +50→100

# Regex patterns
_CELL_RE = re.compile(r"([A-Za-z]+)(\d+)$")
_CELL_REF_IN_FORMULA = re.compile(
    r"(?<![A-Za-z_])(\$?)([A-Z]{1,3})(\$?)(\d{1,7})(?![A-Za-z0-9_(])"
)

# Default values per op_type for value reconstruction
DEFAULT_VALUES: Dict[str, str] = {
    "FONT_BOLD": "True",
    "FONT_ITALIC": "True",
    "FONT_UNDERLINE": "single",
    "FONT_SIZE": "11",
    "FONT_COLOR": "#000000",
    "FONT_NAME": "Calibri",
    "FILL_COLOR": "#FFFFFF",
    "ALIGN_HORIZONTAL": "center",
    "ALIGN_VERTICAL": "middle",
    "WRAP_TEXT": "True",
    "MERGE": "true",
    "UNMERGE": "true",
    "TEXT_ORIENTATION": "0",
    "NUMBER_FORMAT": "#,##0",
    "BORDER_LEFT": "Thin, Continuous, #000000",
    "BORDER_RIGHT": "Thin, Continuous, #000000",
    "BORDER_TOP": "Thin, Continuous, #000000",
    "BORDER_BOTTOM": "Thin, Continuous, #000000",
    "BORDER_OUTSIDE": "Thin, Continuous, #000000",
    "BORDER_ALL": "Thin, Continuous, #000000",
    "BORDER_INSIDE_HORIZONTAL": "Thin, Continuous, #000000",
    "BORDER_INSIDE_VERTICAL": "Thin, Continuous, #000000",
    "BORDER_DIAGONAL_DOWN": "Thin, Continuous, #000000",
    "BORDER_DIAGONAL_UP": "Thin, Continuous, #000000",
    "INPUT": "0",
    "VALUE": "0",
    "FORMULA": "=0",
    "PASTE_FROM": "Sheet1!A1 | values",
}


# ═══════════════════════════════════════════════════════════════════════════ #
#  Data classes                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #


@dataclass(frozen=True)
class ParsedOp:
    """A parsed symbolic operation with extracted features."""

    op_type: str
    op_type_id: int
    sheet: str
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    height: int
    width: int
    raw_value: str
    value_type: str
    value_type_id: int
    symbolic: str


@dataclass(frozen=True)
class FeaturizedOp:
    """A fully featurized operation with absolute + relative features."""

    parsed: ParsedOp
    row_delta: int
    col_delta: int
    height_delta: int
    width_delta: int
    sheet_changed: bool
    movement_class: str
    movement_class_id: int


# ═══════════════════════════════════════════════════════════════════════════ #
#  Low-level parsing helpers                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #


def _parse_cell(cell_str: str) -> Tuple[int, int]:
    """Parse 'A1' → (row=1, col=1).  Returns (1, 1) on failure."""
    m = _CELL_RE.match(cell_str.strip())
    if not m:
        return (1, 1)
    col_str, row_str = m.groups()
    col = 0
    for ch in col_str.upper():
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return int(row_str), col


def _col_to_letter(col: int) -> str:
    """Convert 1-indexed column to letters.  1→A, 27→AA."""
    parts: list = []
    c = col
    while c > 0:
        c, rem = divmod(c - 1, 26)
        parts.append(chr(rem + ord("A")))
    return "".join(reversed(parts)) if parts else "A"


def _make_range_str(sr: int, sc: int, er: int, ec: int) -> str:
    """Build range string from coordinates.  (1,1,1,1)→'A1', (1,1,3,2)→'A1:B3'."""
    start = f"{_col_to_letter(sc)}{sr}"
    if sr == er and sc == ec:
        return start
    return f"{start}:{_col_to_letter(ec)}{er}"


# ═══════════════════════════════════════════════════════════════════════════ #
#  Classification functions                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #


def classify_value(op_type: str, value: str) -> Tuple[str, int]:
    """Classify a value string into one of VALUE_TYPES.

    Returns (value_type_name, value_type_id).
    """
    v = value.strip()

    if not v or v.lower() in ("null", "none", ""):
        return "null", VALUE_TYPE_TO_ID["null"]

    # Op-type-specific classification
    if op_type in ("FONT_BOLD", "FONT_ITALIC", "WRAP_TEXT", "MERGE", "UNMERGE"):
        if v.lower() in ("true", "1"):
            return "true", VALUE_TYPE_TO_ID["true"]
        return "false", VALUE_TYPE_TO_ID["false"]
    if op_type == "ALIGN_HORIZONTAL":
        return "align_h", VALUE_TYPE_TO_ID["align_h"]
    if op_type == "ALIGN_VERTICAL":
        return "align_v", VALUE_TYPE_TO_ID["align_v"]
    if op_type in ("FONT_COLOR", "FILL_COLOR"):
        return "color", VALUE_TYPE_TO_ID["color"]
    if op_type.startswith("BORDER_"):
        return "border_spec", VALUE_TYPE_TO_ID["border_spec"]
    if op_type == "FONT_NAME":
        return "font_name", VALUE_TYPE_TO_ID["font_name"]
    if op_type == "FONT_SIZE":
        return "font_size", VALUE_TYPE_TO_ID["font_size"]
    if op_type == "FONT_UNDERLINE":
        return "underline_style", VALUE_TYPE_TO_ID["underline_style"]
    if op_type == "NUMBER_FORMAT":
        return "number_format", VALUE_TYPE_TO_ID["number_format"]
    if op_type == "TEXT_ORIENTATION":
        return "orientation", VALUE_TYPE_TO_ID["orientation"]
    if op_type in ("PASTE_FROM", "AUTOFILL"):
        return "other", VALUE_TYPE_TO_ID["other"]

    # Content ops — classify by value structure
    if v.startswith("="):
        return "formula", VALUE_TYPE_TO_ID["formula"]
    if v.startswith("["):
        return "list", VALUE_TYPE_TO_ID["list"]
    try:
        float(v.replace(",", ""))
        return "number", VALUE_TYPE_TO_ID["number"]
    except (ValueError, AttributeError):
        pass
    return "string", VALUE_TYPE_TO_ID["string"]


def classify_movement(
    row_delta: int, col_delta: int, sheet_changed: bool
) -> Tuple[str, int]:
    """Classify positional movement into one of MOVEMENT_CLASSES."""
    if sheet_changed:
        return "sheet_change", MOVEMENT_CLASS_TO_ID["sheet_change"]
    if row_delta == 0 and col_delta == 0:
        return "same", MOVEMENT_CLASS_TO_ID["same"]
    if row_delta == 0 and col_delta == 1:
        return "right1", MOVEMENT_CLASS_TO_ID["right1"]
    if row_delta == 0 and col_delta == -1:
        return "left1", MOVEMENT_CLASS_TO_ID["left1"]
    if row_delta == 1 and col_delta == 0:
        return "down1", MOVEMENT_CLASS_TO_ID["down1"]
    if row_delta == -1 and col_delta == 0:
        return "up1", MOVEMENT_CLASS_TO_ID["up1"]
    if row_delta == 0 and col_delta > 1:
        return "right_n", MOVEMENT_CLASS_TO_ID["right_n"]
    if row_delta == 0 and col_delta < -1:
        return "left_n", MOVEMENT_CLASS_TO_ID["left_n"]
    if col_delta == 0 and row_delta > 1:
        return "down_n", MOVEMENT_CLASS_TO_ID["down_n"]
    if col_delta == 0 and row_delta < -1:
        return "up_n", MOVEMENT_CLASS_TO_ID["up_n"]
    return "diagonal", MOVEMENT_CLASS_TO_ID["diagonal"]


# ═══════════════════════════════════════════════════════════════════════════ #
#  Bucketing (for LSTM embedding layers)                                     #
# ═══════════════════════════════════════════════════════════════════════════ #


def bucket_absolute(val: int) -> int:
    """Bucket an absolute row/col/height/width value to [0, 100].

    val 1 → 0, val 101+ → 100.
    """
    return max(0, min(RANGE_BUCKETS - 1, val - 1))


def unbucket_absolute(bucket: int) -> int:
    """Reverse of bucket_absolute.  0 → 1, 100 → 101."""
    return bucket + 1


def bucket_relative(delta: int) -> int:
    """Bucket a delta value to [0, 100].

    delta -50 → 0, delta 0 → 50, delta +50 → 100.
    """
    return max(0, min(RANGE_BUCKETS - 1, delta + REL_CLIP))


def unbucket_relative(bucket: int) -> int:
    """Reverse of bucket_relative.  0 → -50, 50 → 0, 100 → +50."""
    return bucket - REL_CLIP


# ═══════════════════════════════════════════════════════════════════════════ #
#  Symbolic parsing                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #


def parse_symbolic(symbolic: str) -> ParsedOp:
    """Parse a symbolic operation string into a ParsedOp (stateless)."""
    parts = symbolic.split(" | ")
    op_type = parts[0].strip()
    cell_ref = parts[1].strip() if len(parts) > 1 else "Sheet1!A1"
    value = " | ".join(parts[2:]).strip() if len(parts) > 2 else ""

    # Split sheet from range
    if "!" in cell_ref:
        sheet, range_str = cell_ref.rsplit("!", 1)
    else:
        sheet, range_str = "Sheet1", cell_ref

    # Parse range coordinates
    if ":" in range_str:
        start_str, end_str = range_str.split(":", 1)
        sr, sc = _parse_cell(start_str)
        er, ec = _parse_cell(end_str)
    else:
        sr, sc = _parse_cell(range_str)
        er, ec = sr, sc

    height = max(1, er - sr + 1)
    width = max(1, ec - sc + 1)
    op_id = OP_TYPE_TO_ID.get(op_type, UNKNOWN_OP_ID)
    vtype, vtype_id = classify_value(op_type, value)

    return ParsedOp(
        op_type=op_type,
        op_type_id=op_id,
        sheet=sheet,
        start_row=sr,
        start_col=sc,
        end_row=er,
        end_col=ec,
        height=height,
        width=width,
        raw_value=value,
        value_type=vtype,
        value_type_id=vtype_id,
        symbolic=symbolic,
    )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Stateful featurizer                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #


class OperationFeaturizer:
    """Stateful featurizer: parses ops incrementally, computing deltas."""

    def __init__(self) -> None:
        self._prev: Optional[ParsedOp] = None
        self._history: List[FeaturizedOp] = []

    def reset(self) -> None:
        self._prev = None
        self._history = []

    @property
    def history(self) -> List[FeaturizedOp]:
        return self._history

    def featurize_one(self, symbolic: str) -> FeaturizedOp:
        """Parse and featurize one operation (incremental, updates state)."""
        parsed = parse_symbolic(symbolic)

        if self._prev is None:
            rd = cd = hd = wd = 0
            sc = False
        else:
            rd = parsed.start_row - self._prev.start_row
            cd = parsed.start_col - self._prev.start_col
            hd = parsed.height - self._prev.height
            wd = parsed.width - self._prev.width
            sc = parsed.sheet != self._prev.sheet

        mv_cls, mv_id = classify_movement(rd, cd, sc)
        feat = FeaturizedOp(
            parsed=parsed,
            row_delta=rd,
            col_delta=cd,
            height_delta=hd,
            width_delta=wd,
            sheet_changed=sc,
            movement_class=mv_cls,
            movement_class_id=mv_id,
        )
        self._prev = parsed
        self._history.append(feat)
        return feat

    def featurize_sequence(self, ops: List[str]) -> List[FeaturizedOp]:
        """Parse a full sequence (resets state first)."""
        self.reset()
        return [self.featurize_one(op) for op in ops]


def featurize_predicted(
    symbolic: str, prev: Optional[ParsedOp] = None
) -> FeaturizedOp:
    """Featurize a predicted op WITHOUT modifying any featurizer state.

    Used during auto-regressive multi-step prediction to build virtual
    history entries.
    """
    parsed = parse_symbolic(symbolic)
    if prev is None:
        rd = cd = hd = wd = 0
        sc = False
    else:
        rd = parsed.start_row - prev.start_row
        cd = parsed.start_col - prev.start_col
        hd = parsed.height - prev.height
        wd = parsed.width - prev.width
        sc = parsed.sheet != prev.sheet
    mv_cls, mv_id = classify_movement(rd, cd, sc)
    return FeaturizedOp(
        parsed=parsed,
        row_delta=rd,
        col_delta=cd,
        height_delta=hd,
        width_delta=wd,
        sheet_changed=sc,
        movement_class=mv_cls,
        movement_class_id=mv_id,
    )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Formula cell-reference shifting                                           #
# ═══════════════════════════════════════════════════════════════════════════ #


def shift_formula_refs(formula: str, row_delta: int, col_delta: int) -> str:
    """Shift non-absolute A1-style cell references in *formula*.

    ``$A$1`` stays fixed, ``A1`` shifts by (row_delta, col_delta).
    Mixed references (``$A1``, ``A$1``) shift only the non-fixed part.
    """

    def _shift(m: re.Match) -> str:
        col_abs, col_letters, row_abs, row_num = m.groups()
        new_col = col_letters
        new_row = row_num

        if not col_abs:  # relative column
            c = 0
            for ch in col_letters.upper():
                c = c * 26 + (ord(ch) - ord("A") + 1)
            c = max(1, c + col_delta)
            new_col = _col_to_letter(c)

        if not row_abs:  # relative row
            r = max(1, int(row_num) + row_delta)
            new_row = str(r)

        return f"{col_abs}{new_col}{row_abs}{new_row}"

    return _CELL_REF_IN_FORMULA.sub(_shift, formula)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Value reconstruction                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #


def reconstruct_value(
    op_type: str,
    value_type: str,
    history: List[FeaturizedOp],
    target_row: int,
    target_col: int,
) -> str:
    """Reconstruct the actual value for a predicted operation.

    Strategy:
      1. Find the most recent op in *history* with the same op_type.
      2. Reuse its value, shifting formula references if needed.
      3. Fall back to DEFAULT_VALUES for the op_type.
    """
    source = None
    for feat in reversed(history):
        if feat.parsed.op_type == op_type:
            source = feat
            break

    if source is not None:
        val = source.parsed.raw_value
        # Shift cell references inside formulas
        if value_type == "formula" and val.startswith("="):
            dr = target_row - source.parsed.start_row
            dc = target_col - source.parsed.start_col
            if dr != 0 or dc != 0:
                val = shift_formula_refs(val, dr, dc)
        return val

    return DEFAULT_VALUES.get(op_type, "")


# ═══════════════════════════════════════════════════════════════════════════ #
#  Symbolic string construction                                              #
# ═══════════════════════════════════════════════════════════════════════════ #


def build_symbolic(
    op_type: str,
    sheet: str,
    start_row: int,
    start_col: int,
    height: int,
    width: int,
    value: str,
) -> str:
    """Build a symbolic string from predicted features."""
    end_row = start_row + height - 1
    end_col = start_col + width - 1
    range_str = _make_range_str(start_row, start_col, end_row, end_col)
    if value:
        return f"{op_type} | {sheet}!{range_str} | {value}"
    return f"{op_type} | {sheet}!{range_str}"
