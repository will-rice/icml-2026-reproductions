"""Abstract base class for composable symbolic operation transforms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class SymbolicTransform(ABC):
    """Stateful transform between standard DSL and a transformed representation.

    Each transform maintains internal state that tracks the trajectory as
    operations are encoded. Encoding is incremental — each ``encode_one()``
    call processes one operation and updates state.

    Decoding predictions uses the current encoder state but does NOT modify it,
    since predictions are tentative and may not be accepted.

    Transforms are composable: multiple transforms can be chained in a pipeline.
    Encoding applies transforms in order; decoding applies them in reverse.
    """

    @abstractmethod
    def encode_one(self, symbolic_str: str) -> str:
        """Encode a single symbolic operation, updating internal state.

        Args:
            symbolic_str: Standard DSL string,
                e.g. ``"FILL_COLOR | Sheet1!B1 | #FFFF00"``.

        Returns:
            Transformed string.
        """

    @abstractmethod
    def decode_predictions(self, predictions: List[str]) -> List[str]:
        """Decode a batch of predicted operations back to standard DSL.

        Uses current encoder state to resolve the first prediction.
        Subsequent predictions are resolved relative to each other
        (for multi-op prediction sequences).

        Does NOT modify encoder state.

        Args:
            predictions: List of transformed prediction strings.

        Returns:
            List of standard DSL strings. Invalid predictions may be
            returned as-is or filtered.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new trajectory."""

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for reproducibility."""
        return {"type": self.__class__.__name__}

    def encode_sequence(self, symbolic_ops: List[str]) -> List[str]:
        """Convenience: encode a full sequence from scratch.

        Resets state, then encodes each op.
        """
        self.reset()
        return [self.encode_one(op) for op in symbolic_ops]
