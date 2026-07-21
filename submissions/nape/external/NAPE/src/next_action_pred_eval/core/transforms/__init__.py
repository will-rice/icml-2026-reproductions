"""Composable symbolic transforms for DSL operations.

Transforms convert between the standard symbolic DSL and alternative
representations (relative ranges, value tokens, relative formula refs).
They are composable: multiple transforms can be chained in a pipeline,
applied in a canonical order for encoding and the reverse for decoding.

Usage::

    from next_action_pred_eval.core.transforms import build_transforms

    transforms = build_transforms([
        {"type": "relative_formula"},
        {"type": "relative_range"},
        {"type": "value_lookup"},
    ])

    # Encode a full trajectory
    for t in transforms:
        encoded_ops = t.encode_sequence(ops)
        ops = encoded_ops

    # Or incrementally via TransformedSolver (typical evaluation usage)
"""

from typing import Any, Dict, List

from .base import SymbolicTransform
from .relative_formula import RelativeFormulaTransform
from .relative_range import RelativeRangeTransform
from .value_lookup import ValueLookupTransform

# Canonical encoding order: formula first (needs absolute ranges),
# then range (makes ranges relative), then value (maps values to tokens).
# Decoding applies in reverse order.
TRANSFORM_ENCODE_ORDER = ["relative_formula", "relative_range", "value_lookup"]

_TRANSFORM_REGISTRY: Dict[str, type] = {
    "relative_range": RelativeRangeTransform,
    "relative_formula": RelativeFormulaTransform,
    "value_lookup": ValueLookupTransform,
}


def build_transforms(
    configs: List[Dict[str, Any]],
) -> List[SymbolicTransform]:
    """Build an ordered list of transforms from config dicts.

    Transforms are sorted into the canonical encoding order regardless
    of the order specified in the config.

    Args:
        configs: List of dicts, each with at least a ``"type"`` key.
            Additional keys are passed as constructor kwargs.

    Returns:
        List of SymbolicTransform instances in correct encoding order.

    Raises:
        ValueError: If an unknown transform type is specified.
    """
    if not configs:
        return []

    transforms: Dict[str, SymbolicTransform] = {}
    for cfg in configs:
        ttype = cfg["type"]
        if ttype not in _TRANSFORM_REGISTRY:
            raise ValueError(
                f"Unknown transform type: {ttype!r}. "
                f"Available: {list(_TRANSFORM_REGISTRY)}"
            )
        kwargs = {k: v for k, v in cfg.items() if k != "type"}
        transforms[ttype] = _TRANSFORM_REGISTRY[ttype](**kwargs)

    # Sort into canonical encoding order
    ordered: list = []
    for ttype in TRANSFORM_ENCODE_ORDER:
        if ttype in transforms:
            ordered.append(transforms[ttype])

    return ordered


__all__ = [
    "SymbolicTransform",
    "RelativeRangeTransform",
    "RelativeFormulaTransform",
    "ValueLookupTransform",
    "build_transforms",
    "TRANSFORM_ENCODE_ORDER",
]
