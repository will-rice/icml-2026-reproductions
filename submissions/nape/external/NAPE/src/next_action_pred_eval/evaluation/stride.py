"""
Stride Configuration Module
Defines prediction frequency configuration for the evaluation framework.

"Stride" refers to how often predictions are made during evaluation:
- Every step: Predict after each user action
- Fixed interval: Predict every N steps
- Adaptive: Use heuristics to decide when to predict
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Union

from next_action_pred_eval.core.operation import Operation


class StrideMode(Enum):
    """Defines when predictions should be made during evaluation."""

    EVERY_STEP = "every_step"
    """Predict at every step in the action stream."""

    FIXED_INTERVAL = "fixed_interval"
    """Predict at fixed intervals (e.g., every 3 steps)."""

    ADAPTIVE = "adaptive"
    """Use a strategy function to decide when to predict."""


@dataclass
class StrideConfig:
    """
    Configuration for prediction frequency during evaluation.

    Examples::

        # Predict at every step
        config = StrideConfig(mode=StrideMode.EVERY_STEP)

        # Predict every 5 steps
        config = StrideConfig(
            mode=StrideMode.FIXED_INTERVAL,
            interval=5,
        )

        # Adaptive prediction based on operation patterns
        config = StrideConfig(
            mode=StrideMode.ADAPTIVE,
            strategy=my_stride_strategy,
        )
    """

    mode: StrideMode = StrideMode.EVERY_STEP
    """The stride mode to use."""

    interval: int = 1
    """
    For FIXED_INTERVAL mode: predict every N steps.
    For EVERY_STEP mode: this is always 1.
    """

    strategy: Optional[Callable[[int, List[Operation], Dict[str, Any]], bool]] = None
    """
    For ADAPTIVE mode: function that decides whether to predict.

    Signature: strategy(step_index, previous_actions, context) -> bool
    Returns True if prediction should be made at this step.
    """

    min_context_ops: int = 1
    """Minimum number of operations before first prediction."""

    max_predictions_per_run: Optional[int] = None
    """Maximum predictions per evaluation run (None = unlimited)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional configuration metadata."""

    def should_predict(
        self,
        step_index: int,
        previous_actions: List[Operation],
        context: Optional[Dict[str, Any]] = None,
        predictions_made: int = 0,
    ) -> bool:
        """
        Determine if a prediction should be made at this step.

        Args:
            step_index: Current step index (0-based).
            previous_actions: Operations executed so far.
            context: Optional additional context.
            predictions_made: Number of predictions already made this run.

        Returns:
            True if prediction should be made, False otherwise.
        """
        # Check minimum context requirement
        if len(previous_actions) < self.min_context_ops:
            return False

        # Check maximum predictions limit
        if (
            self.max_predictions_per_run is not None
            and predictions_made >= self.max_predictions_per_run
        ):
            return False

        # Apply mode-specific logic
        if self.mode == StrideMode.EVERY_STEP:
            return True

        elif self.mode == StrideMode.FIXED_INTERVAL:
            # Predict at fixed intervals (step 0, interval, 2*interval, ...)
            return step_index % self.interval == 0

        elif self.mode == StrideMode.ADAPTIVE:
            if self.strategy is None:
                raise ValueError(
                    "ADAPTIVE mode requires a strategy function. "
                    "Set stride_config.strategy to a callable."
                )
            return self.strategy(step_index, previous_actions, context or {})

        return False

    def __repr__(self) -> str:
        if self.mode == StrideMode.EVERY_STEP:
            return "StrideConfig(mode=EVERY_STEP)"
        elif self.mode == StrideMode.FIXED_INTERVAL:
            return f"StrideConfig(mode=FIXED_INTERVAL, interval={self.interval})"
        else:
            strategy_name = (
                self.strategy.__name__ if self.strategy else "None"
            )
            return f"StrideConfig(mode=ADAPTIVE, strategy={strategy_name})"


# ===== Predefined Stride Strategies =====


def stride_after_format_change(
    step_index: int,
    previous_actions: List[Operation],
    context: Dict[str, Any],
) -> bool:
    """
    Predict after formatting operations (fill color, font, border, etc.).

    This strategy assumes formatting often indicates the start of a
    new section or pattern that could be predicted.
    """
    if not previous_actions:
        return False

    last_op = previous_actions[-1]
    format_op_names = {
        "SetFillColor",
        "SetFontProperty",
        "SetBorder",
        "SetAlignment",
        "SetNumberFormat",
        "MergeCells",
    }
    return type(last_op).__name__ in format_op_names


def stride_after_data_entry(
    step_index: int,
    previous_actions: List[Operation],
    context: Dict[str, Any],
) -> bool:
    """
    Predict after data entry operations (values, formulas).

    This strategy assumes data entry patterns are often predictable.
    """
    if not previous_actions:
        return False

    last_op = previous_actions[-1]
    data_op_names = {"SetValue", "SetFormula", "SetInput", "AutoFill"}
    return type(last_op).__name__ in data_op_names


def stride_after_n_same_type(
    n: int = 3,
) -> Callable[[int, List[Operation], Dict[str, Any]], bool]:
    """
    Create a strategy that predicts after N consecutive operations of the same type.

    Args:
        n: Number of consecutive same-type operations to trigger prediction.

    Returns:
        Strategy function.
    """

    def strategy(
        step_index: int,
        previous_actions: List[Operation],
        context: Dict[str, Any],
    ) -> bool:
        if len(previous_actions) < n:
            return False

        # Check if last N operations are the same type
        recent = previous_actions[-n:]
        first_type = type(recent[0]).__name__
        return all(type(op).__name__ == first_type for op in recent)

    return strategy


def stride_every_n_steps(n: int = 5) -> Callable[[int, List[Operation], Dict[str, Any]], bool]:
    """
    Create a strategy that predicts every N steps.

    This is a convenience function that creates an ADAPTIVE strategy
    equivalent to FIXED_INTERVAL mode.

    Args:
        n: Interval between predictions.

    Returns:
        Strategy function.
    """

    def strategy(
        step_index: int,
        previous_actions: List[Operation],
        context: Dict[str, Any],
    ) -> bool:
        return step_index > 0 and step_index % n == 0

    return strategy


# ===== Convenience Factory Functions =====


def every_step() -> StrideConfig:
    """Create a config that predicts at every step."""
    return StrideConfig(mode=StrideMode.EVERY_STEP)


def every_n_steps(n: int) -> StrideConfig:
    """Create a config that predicts every N steps."""
    return StrideConfig(mode=StrideMode.FIXED_INTERVAL, interval=n)


def adaptive(
    strategy: Callable[[int, List[Operation], Dict[str, Any]], bool],
) -> StrideConfig:
    """Create a config with an adaptive prediction strategy."""
    return StrideConfig(mode=StrideMode.ADAPTIVE, strategy=strategy)


__all__ = [
    # Core classes
    "StrideMode",
    "StrideConfig",
    # Predefined strategies
    "stride_after_format_change",
    "stride_after_data_entry",
    "stride_after_n_same_type",
    "stride_every_n_steps",
    # Factory functions
    "every_step",
    "every_n_steps",
    "adaptive",
]
