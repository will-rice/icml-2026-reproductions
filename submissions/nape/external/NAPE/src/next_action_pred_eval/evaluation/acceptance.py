"""
Acceptance Heuristic Module
Defines criteria for accepting/rejecting predictions in document-level evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any

from next_action_pred_eval.evaluation.metrics import EvaluationMetrics


@dataclass
class ConstraintCheck:
    """Result of checking a single constraint."""

    metric: str
    """Name of the metric being checked."""

    value: Any
    """Actual value of the metric."""

    minimum: Optional[float]
    """Minimum threshold (None if no minimum)."""

    maximum: Optional[float]
    """Maximum threshold (None if no maximum)."""

    passed: bool
    """Whether the constraint was satisfied."""

    def __repr__(self) -> str:
        bounds = []
        if self.minimum is not None:
            bounds.append(f">= {self.minimum}")
        if self.maximum is not None:
            bounds.append(f"<= {self.maximum}")
        bounds_str = " AND ".join(bounds) if bounds else "no bounds"
        status = "✓" if self.passed else "✗"
        return f"{status} {self.metric}={self.value} ({bounds_str})"


@dataclass
class AcceptanceHeuristic:
    """
    Defines criteria for accepting/rejecting a prediction.

    Uses a flexible constraint system where any metric from EvaluationMetrics
    can be constrained with min/max values.

    Example::

        # Require at least 90% precision and at least 1 action saved
        heuristic = AcceptanceHeuristic(
            name="ideal_user",
            description="Ideal user acceptance criteria",
            constraints={
                "final_state_precision": (0.9, None),  # min 90%, no max
                "final_state_ops_saved": (1, None),    # at least 1 saved
            }
        )

        # Check if prediction should be accepted
        if heuristic.evaluate(evaluation_metrics):
            # Accept prediction
            pass
    """

    name: str
    """Unique identifier for the heuristic."""

    description: str
    """Human-readable description."""

    weight: float = 1.0
    """Weight for weighted aggregation (optional)."""

    constraints: Dict[str, Tuple[Optional[float], Optional[float]]] = field(
        default_factory=dict
    )
    """
    Constraints on metrics.
    Format: {metric_name: (min_value, max_value)}
    Use None for no constraint on that side.
    """

    def evaluate(self, metrics: EvaluationMetrics) -> bool:
        """
        Check if prediction meets all constraints.

        Args:
            metrics: EvaluationMetrics from prediction evaluation.

        Returns:
            True if all constraints are satisfied, False otherwise.

        Raises:
            ValueError: If constraint references unknown metric.
        """
        accepted, _ = self.evaluate_with_details(metrics)
        return accepted

    def evaluate_with_details(
        self, metrics: EvaluationMetrics
    ) -> Tuple[bool, List[ConstraintCheck]]:
        """
        Evaluate constraints and return decision with detailed results.

        Args:
            metrics: EvaluationMetrics from prediction evaluation.

        Returns:
            Tuple of (accepted: bool, constraint_checks: List[ConstraintCheck]).
        """
        if not self.constraints:
            return True, []

        results: List[ConstraintCheck] = []
        accepted = True

        for metric_name, (min_val, max_val) in self.constraints.items():
            if not hasattr(metrics, metric_name):
                raise ValueError(
                    f"Unknown metric '{metric_name}' in heuristic '{self.name}'. "
                    f"Check EvaluationMetrics for available metrics."
                )

            value = getattr(metrics, metric_name)
            passed_min = True if min_val is None else value >= min_val
            passed_max = True if max_val is None else value <= max_val
            passed = passed_min and passed_max

            results.append(
                ConstraintCheck(
                    metric=metric_name,
                    value=value,
                    minimum=min_val,
                    maximum=max_val,
                    passed=passed,
                )
            )

            if not passed:
                accepted = False

        return accepted, results

    def __repr__(self) -> str:
        constraint_strs = []
        for metric, (min_val, max_val) in self.constraints.items():
            if min_val is not None and max_val is not None:
                constraint_strs.append(f"{min_val} <= {metric} <= {max_val}")
            elif min_val is not None:
                constraint_strs.append(f"{metric} >= {min_val}")
            elif max_val is not None:
                constraint_strs.append(f"{metric} <= {max_val}")

        constraint_str = " AND ".join(constraint_strs) if constraint_strs else "no constraints"
        return f"AcceptanceHeuristic({self.name}, {constraint_str})"


# ===== Predefined Heuristics =====

HEURISTIC_PERFECT = AcceptanceHeuristic(
    name="perfect",
    description="Perfect prediction: 100% state precision and recall",
    weight=1.0,
    constraints={
        "final_state_precision": (1.0, None),
        "final_state_recall": (1.0, None),
    },
)

HEURISTIC_PRECISION_100 = AcceptanceHeuristic(
    name="precision_100",
    description="100% precision (no wrong predictions, missing is OK)",
    weight=1.0,
    constraints={
        "final_state_precision": (1.0, None),
    },
)

HEURISTIC_PRECISION_90 = AcceptanceHeuristic(
    name="precision_90",
    description="At least 90% precision",
    weight=1.0,
    constraints={
        "final_state_precision": (0.90, None),
    },
)

HEURISTIC_PRECISION_60 = AcceptanceHeuristic(
    name="precision_60",
    description="At least 60% precision",
    weight=1.0,
    constraints={
        "final_state_precision": (0.60, None),
    },
)

HEURISTIC_IDEAL_USER = AcceptanceHeuristic(
    name="ideal_user",
    description="Ideal user: operations saved > 0 and at least 90% precision",
    weight=1.0,
    constraints={
        "final_state_precision": (0.90, None),
        "final_state_ops_saved": (1, None),
    },
)

HEURISTIC_IDEAL_USER_STRICT = AcceptanceHeuristic(
    name="ideal_user_strict",
    description="Strict ideal user: operations saved >= 2 and 100% precision",
    weight=1.0,
    constraints={
        "final_state_precision": (1.0, None),
        "final_state_ops_saved": (2, None),
    },
)

HEURISTIC_IDEAL_USER_HARD_2 = AcceptanceHeuristic(
    name="ideal_user_hard_2",
    description="Ideal user: operations saved >= 2 and at least 100% precision",
    weight=1.0,
    constraints={
        "final_state_precision": (1.0, None),
        "final_state_ops_saved": (2, None),
    },
)

HEURISTIC_STEPS_SAVED = AcceptanceHeuristic(
    name="steps_saved",
    description="Operations saved > 0 (any precision)",
    weight=1.0,
    constraints={
        "final_state_ops_saved": (1, None),
    },
)

HEURISTIC_GREEDY = AcceptanceHeuristic(
    name="greedy",
    description="Operations saved >= 0 (accept unless prediction makes things worse)",
    weight=1.0,
    constraints={
        "final_state_ops_saved": (0, None),
    },
)

HEURISTIC_SSAV2 = AcceptanceHeuristic(
    name="ssav2",
    description="Operations saved >= 2 (any precision)",
    weight=1.0,
    constraints={
        "final_state_ops_saved": (2, None),
    },
)

HEURISTIC_F1_80 = AcceptanceHeuristic(
    name="f1_80",
    description="F1 score >= 0.80 (balanced precision and recall)",
    weight=1.0,
    constraints={
        "final_state_f1_score": (0.80, None),
    },
)

HEURISTIC_ACCEPT_ALL = AcceptanceHeuristic(
    name="accept_all",
    description="Accept all predictions (baseline for analysis)",
    weight=1.0,
    constraints={},
)

HEURISTIC_REJECT_ALL = AcceptanceHeuristic(
    name="reject_all",
    description="Reject all predictions (baseline for analysis)",
    weight=1.0,
    constraints={
        "final_state_precision": (2.0, None),  # Impossible condition
    },
)

# Heuristic collections
COMMON_HEURISTICS = [
    HEURISTIC_PERFECT,
    HEURISTIC_PRECISION_100,
    HEURISTIC_PRECISION_90,
    HEURISTIC_PRECISION_60,
    HEURISTIC_IDEAL_USER,
    HEURISTIC_IDEAL_USER_STRICT,
    HEURISTIC_IDEAL_USER_HARD_2,
    HEURISTIC_STEPS_SAVED,
    HEURISTIC_GREEDY,
    HEURISTIC_SSAV2,
    HEURISTIC_F1_80,
]

BASELINE_HEURISTICS = [
    HEURISTIC_ACCEPT_ALL,
    HEURISTIC_REJECT_ALL,
]

ALL_PREDEFINED_HEURISTICS = COMMON_HEURISTICS + BASELINE_HEURISTICS


def get_heuristic_by_name(name: str) -> Optional[AcceptanceHeuristic]:
    """
    Get a predefined heuristic by name.

    Args:
        name: Heuristic name (e.g., "ideal_user", "precision_90").

    Returns:
        AcceptanceHeuristic if found, None otherwise.
    """
    for h in ALL_PREDEFINED_HEURISTICS:
        if h.name == name:
            return h
    return None


def create_custom_heuristic(
    name: str,
    constraints: Dict[str, Tuple[Optional[float], Optional[float]]],
    description: Optional[str] = None,
    weight: float = 1.0,
) -> AcceptanceHeuristic:
    """
    Create a custom acceptance heuristic.

    Args:
        name: Unique identifier.
        constraints: Dict mapping metric names to (min, max) bounds.
        description: Optional human-readable description.
        weight: Weight for aggregation.

    Returns:
        AcceptanceHeuristic instance.

    Example::

        # Require high precision and at least 5 ops saved
        heuristic = create_custom_heuristic(
            name="my_heuristic",
            constraints={
                "final_state_precision": (0.95, None),
                "final_state_ops_saved": (5, None),
            },
        )
    """
    return AcceptanceHeuristic(
        name=name,
        description=description or f"Custom heuristic: {name}",
        weight=weight,
        constraints=constraints,
    )


__all__ = [
    # Core classes
    "AcceptanceHeuristic",
    "ConstraintCheck",
    # Predefined heuristics
    "HEURISTIC_PERFECT",
    "HEURISTIC_PRECISION_100",
    "HEURISTIC_PRECISION_90",
    "HEURISTIC_PRECISION_60",
    "HEURISTIC_IDEAL_USER",
    "HEURISTIC_IDEAL_USER_STRICT",
    "HEURISTIC_IDEAL_USER_HARD_2",
    "HEURISTIC_STEPS_SAVED",
    "HEURISTIC_F1_80",
    "HEURISTIC_ACCEPT_ALL",
    "HEURISTIC_REJECT_ALL",
    # Collections
    "COMMON_HEURISTICS",
    "BASELINE_HEURISTICS",
    "ALL_PREDEFINED_HEURISTICS",
    # Utility functions
    "get_heuristic_by_name",
    "create_custom_heuristic",
]
