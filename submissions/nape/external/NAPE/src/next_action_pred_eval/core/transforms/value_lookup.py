"""Transform that replaces values with occurrence-indexed lookup tokens.

Standard:   FILL_COLOR | B1 | #FFFF00
Encoded:    FILL_COLOR | B1 | fill_color_1

Common defaults (e.g. #000000 for font_color) get the ``_0`` suffix as
trajectory-independent constants.  Content values (INPUT/VALUE/FORMULA) are
classified by type (``inp_string``, ``inp_number``, ``inp_formula``,
``inp_list``).
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from .base import SymbolicTransform

# Default values per op type that always map to <category>_0.
DEFAULT_VALUES: Dict[str, Set[str]] = {
    "FONT_BOLD": {"True", "true"},
    "FONT_ITALIC": {"True", "true"},
    "FONT_COLOR": {"#000000"},
    "FILL_COLOR": {"#FFFFFF", "#ffffff"},
    "FONT_SIZE": {"11", "11.0"},
    "FONT_UNDERLINE": {"Single", "single"},
    "FONT_NAME": {"Calibri"},
    "WRAP_TEXT": {"True", "true"},
    "MERGE": {"true", "True"},
    "UNMERGE": {"true", "True"},
    "ALIGN_HORIZONTAL": {"left"},
    "ALIGN_VERTICAL": {"bottom"},
    "TEXT_ORIENTATION": {"0"},
    "NUMBER_FORMAT": {"General"},
}

for _side in [
    "LEFT", "RIGHT", "TOP", "BOTTOM", "OUTSIDE", "ALL",
    "INSIDE_HORIZONTAL", "INSIDE_VERTICAL", "DIAGONAL_DOWN", "DIAGONAL_UP",
]:
    DEFAULT_VALUES[f"BORDER_{_side}"] = {"Thin, Continuous, #000000"}

CONTENT_OP_TYPES = {"INPUT", "VALUE", "FORMULA"}


def _classify_content_value(value: str) -> str:
    """Classify an INPUT/VALUE/FORMULA value into a category token prefix."""
    v = value.strip()
    if v.startswith("[[") or v.startswith("["):
        return "inp_list"
    if v.startswith("="):
        return "inp_formula"
    if v.startswith('"') and v.endswith('"'):
        return "inp_string"
    try:
        float(v)
        return "inp_number"
    except ValueError:
        return "inp_string"


class ValueLookupTransform(SymbolicTransform):
    """Replaces values with occurrence-indexed tokens.

    Each unique value is assigned a token based on its category and
    occurrence order (e.g. ``fill_color_1`` for the first non-default fill
    color, ``inp_string_3`` for the third unique string input).

    Tokens ending in ``_0`` represent trajectory-independent defaults
    (e.g. ``font_color_0`` always means ``#000000``).
    """

    def __init__(self, defaults: Optional[Dict[str, Set[str]]] = None):
        self._defaults = defaults or DEFAULT_VALUES
        self.reset()

    def reset(self) -> None:
        self._counter: Dict[str, int] = {}
        self._value_to_token: Dict[Tuple[str, str], str] = {}
        self._token_to_value: Dict[str, str] = {}

    def encode_one(self, symbolic_str: str) -> str:
        parts = symbolic_str.split(" | ")
        op_type = parts[0].strip() if parts else ""
        prefix = parts[:2]
        value = parts[2].strip() if len(parts) > 2 else ""
        suffix = parts[3:] if len(parts) > 3 else []

        token = "<empty>" if not value else self._encode_value(op_type, value)
        return " | ".join(prefix + [token] + suffix)

    def _encode_value(self, op_type: str, value: str) -> str:
        if op_type in CONTENT_OP_TYPES:
            category = _classify_content_value(value)
        else:
            category = op_type.lower()

        defaults = self._defaults.get(op_type, set())
        if value in defaults:
            return f"{category}_0"

        key = (category, value)
        if key in self._value_to_token:
            return self._value_to_token[key]

        idx = self._counter.get(category, 0) + 1
        self._counter[category] = idx
        token = f"{category}_{idx}"
        self._value_to_token[key] = token
        self._token_to_value[token] = value
        return token

    def decode_predictions(self, predictions: List[str]) -> List[str]:
        results: list = []
        for pred in predictions:
            parts = pred.split(" | ")
            op_type = parts[0].strip() if parts else ""
            prefix = parts[:2]
            value_token = parts[2].strip() if len(parts) > 2 else ""
            suffix = parts[3:] if len(parts) > 3 else []

            value = self._resolve_token(op_type, value_token)
            results.append(" | ".join(prefix + [value] + suffix))
        return results

    def _resolve_token(self, op_type: str, token: str) -> str:
        if token == "<empty>":
            return ""
        if token in self._token_to_value:
            return self._token_to_value[token]
        if token.endswith("_0"):
            defaults = self._defaults.get(op_type, set())
            if defaults:
                return next(iter(defaults))
            return "True" if op_type not in CONTENT_OP_TYPES else '""'
        return token

    def get_value_map(self) -> Dict[str, str]:
        """Return current token -> value mapping (for inspection/debugging)."""
        result = dict(self._token_to_value)
        for op_type, defaults in self._defaults.items():
            if op_type in CONTENT_OP_TYPES:
                continue
            category = op_type.lower()
            for val in defaults:
                result[f"{category}_0"] = val
                break
        return result

    def get_config(self) -> Dict[str, Any]:
        return {"type": "value_lookup"}
