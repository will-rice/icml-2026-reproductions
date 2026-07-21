"""
Solver Interface Module
Defines the abstract base class for prediction solvers.

Users implement this interface to create their own prediction systems
that can be evaluated using the framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel

from next_action_pred_eval.core.operation import Operation


class DecodingConfig(BaseModel, frozen=True):
    """Configuration for multi-step prediction decoding.

    Controls how solvers generate and truncate prediction sequences.
    """

    max_predictions: int = 5
    """Hard ceiling on the number of operations to predict per call."""

    stop_on_type_change: bool = True
    """Stop when the predicted operation type differs from the first prediction.
    Set to False for solvers that can reliably predict across type boundaries
    (e.g., OnlineNGram suffix matching)."""


@dataclass
class PredictionResult:
    """Result from a prediction call."""

    predicted_operations: List[Operation]
    """Predicted operations as Operation objects."""

    predicted_symbolic: List[str]
    """Predicted operations in symbolic form."""

    input_tokens: int = 0
    """Input/prompt tokens consumed."""

    output_tokens: int = 0
    """Output/completion tokens generated."""

    total_tokens: int = 0
    """Total tokens consumed (input + output)."""

    generation_time: float = 0.0
    """Time taken for generation in seconds."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata from the prediction."""

    @property
    def tokens_used(self) -> int:
        """Backward compatibility property."""
        return self.total_tokens


class ISolver(ABC):
    """
    Abstract base class for prediction solvers.

    Users implement this interface to create their own prediction systems.
    The solver is responsible for predicting the next operations given
    the current context (previous actions and optional workbook state).

    Example implementation::

        class MySolver(ISolver):
            def __init__(self, model_name: str = "gpt-4"):
                self.model_name = model_name

            def predict(
                self,
                previous_actions: List[Union[Operation, str]],
                workbook_state: Optional[Dict[str, Any]] = None,
                context: Optional[Dict[str, Any]] = None,
            ) -> PredictionResult:
                # Your prediction logic here
                predicted_ops = self._call_model(previous_actions)
                return PredictionResult(
                    predicted_operations=predicted_ops,
                    predicted_symbolic=operations_to_symbolic(predicted_ops),
                )
    """

    @abstractmethod
    def predict(
        self,
        previous_actions: List[Union[Operation, str]],
        workbook_state: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        """
        Predict next operations given the current context.

        Args:
            previous_actions: Previous operations (as Operation objects or
                            symbolic strings) that have been executed.
            workbook_state: Optional current workbook state dictionary.
                          Can be used for state-aware predictions.
            context: Optional additional context (e.g., region metadata,
                    user preferences, etc.).

        Returns:
            PredictionResult containing the predicted operations and metadata.
        """
        pass

    def reset(self) -> None:
        """
        Reset any internal state for a new evaluation run.

        Override this method if your solver maintains state between
        predictions (e.g., caches, conversation history).
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Return the solver's configuration for logging/reproducibility.

        Override this method to include solver-specific configuration.

        Returns:
            Dictionary containing solver configuration.
        """
        return {"solver_class": self.__class__.__name__}


class ConstantSolver(ISolver):
    """
    A simple solver that always returns an empty prediction.

    Useful as a baseline for evaluation.
    """

    def predict(
        self,
        previous_actions: List[Union[Operation, str]],
        workbook_state: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        """Return empty prediction."""
        return PredictionResult(
            predicted_operations=[],
            predicted_symbolic=[],
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            generation_time=0.0,
            metadata={"solver": "constant", "action": "empty"},
        )

    def get_config(self) -> Dict[str, Any]:
        return {"solver_class": "ConstantSolver", "prediction": "empty"}


__all__ = [
    "ISolver",
    "PredictionResult",
    "ConstantSolver",
]
