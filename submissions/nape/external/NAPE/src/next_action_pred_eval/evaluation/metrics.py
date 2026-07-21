"""
Evaluation Metrics Module
Defines metrics for evaluating predicted operations against ground truth.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class EvaluationMetrics:
    """
    Metrics for evaluating predicted operations.

    This dataclass contains all metrics computed during evaluation,
    including operation-level and state-level comparisons.

    The primary metrics for paper evaluation are the final_state_* metrics,
    which compare the predicted state against the target final state.
    """

    # Basic counts
    total_ground_truth: int
    """Total operations in ground truth segment."""

    total_predicted: int
    """Total operations predicted."""

    # Operation-level matching (secondary metrics)
    exact_matches: int
    """Operations that match exactly (type, range, and value)."""

    correct_op_wrong_range: int
    """Correct operation type but wrong cell range."""

    wrong_op: int
    """Completely wrong operation type."""

    attribute_mismatch_matches: int
    """Correct operation type and range but wrong attribute value."""

    # Lookahead metrics (eventual matching)
    lookahead_window: Optional[int]
    """Lookahead window size: None=full sequence, N=next N ops, 0=disabled."""

    lookahead_matches: int
    """Predictions that exactly match operations in lookahead window."""

    lookahead_flex_matches: int
    """Lookahead matches allowing attribute mismatch."""

    lookahead_accuracy: float
    """lookahead_matches / total_predicted."""

    lookahead_recall: float
    """lookahead_matches / total_lookahead_truth."""

    lookahead_flex_accuracy: float
    """lookahead_flex_matches / total_predicted."""

    lookahead_flex_recall: float
    """lookahead_flex_matches / total_lookahead_truth."""

    match_positions: List[int]
    """Position offsets where matches occurred (for analysis)."""

    # ===== PRIMARY METRICS: Final State Comparison =====
    # These compare predicted state vs final workbook state

    final_state_tp: int
    """True positives: Properties correctly predicted."""

    final_state_fp: int
    """False positives: Properties predicted but wrong/missing in target."""

    final_state_fn: int
    """False negatives: Properties in target but missing in prediction."""

    final_state_mm: int
    """Mismatches: properties in both states with different values."""

    final_state_precision: float
    """Precision: TP / (TP + FP)."""

    final_state_recall: float
    """Recall: TP / (TP + FN)."""

    final_state_f1_score: float
    """F1 Score: 2*TP / (2*TP + FP + FN)."""

    final_state_ops_diff: int
    """Number of operations required to go from predicted to final state."""

    final_state_ops_saved: int
    """
    User Actions Saved (UAS): Operations user doesn't need to perform.
    Calculated as: total_final_ops - ops_diff
    """

    # Missing/extra operations
    missing_operations: int
    """Operations in ground truth but not predicted."""

    extra_operations: int
    """Operations predicted but not in ground truth."""

    # Additional details
    details: Dict[str, Any]
    """Detailed breakdown for debugging."""

    # Token usage (for LLM-based predictions)
    input_tokens: int = 0
    """Tokens in prompt/input."""

    output_tokens: int = 0
    """Tokens in completion/output."""

    total_tokens: int = 0
    """Total tokens: input_tokens + output_tokens."""

    # Undo-aware savings (for online evaluation)
    undo_gain_from_final_state: int = 0
    """Ops saved by replacing region with final-state prediction."""

    undo_gain_from_dedup: int = 0
    """Operations saved from deduplicating future steps."""

    undo_inverse_ops: int = 0
    """Operations added to undo over-predictions."""

    undo_net_ops_saved: int = 0
    """Net operations saved after accounting for undo cost."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Return a human-readable summary of key metrics."""
        return (
            f"Precision: {self.final_state_precision:.2%} | "
            f"Recall: {self.final_state_recall:.2%} | "
            f"F1: {self.final_state_f1_score:.2%} | "
            f"UAS: {self.final_state_ops_saved}"
        )


@dataclass
class DocumentMetrics:
    """
    Aggregated metrics across multiple predictions for a single document.

    Used in document-level evaluation where multiple predictions are made
    and potentially accepted/rejected based on heuristics.
    """

    heuristic_name: str
    """Name of the acceptance heuristic used."""

    heuristic_weight: float
    """Weight of the heuristic."""

    total_predictions: int
    """Total predictions made."""

    accepted_predictions: int
    """Predictions accepted by the heuristic."""

    rejected_predictions: int
    """Predictions rejected by the heuristic."""

    skipped_predictions: int
    """Predictions skipped (e.g., empty)."""

    acceptance_rate: float
    """accepted_predictions / total_predictions."""

    # Operation counts
    total_operations_in_accepted: int
    """Total operations in accepted predictions."""

    total_operations_in_rejected: int
    """Total operations in rejected predictions."""

    total_correct_operations_in_accepted: int
    """Correct operations in accepted predictions."""

    correctness_rate_in_accepted: float
    """Correctness rate for accepted predictions."""

    # State-level aggregates
    total_tp: int
    accepted_tp: int
    total_fp: int
    accepted_fp: int
    total_fn: int
    accepted_fn: int
    total_mm: int
    accepted_mm: int

    # UAS metrics
    total_final_state_ops_saved: int
    accepted_final_state_ops_saved: int

    # Token usage
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    accepted_input_tokens: int
    accepted_output_tokens: int
    accepted_total_tokens: int
    rejected_input_tokens: int
    rejected_output_tokens: int
    rejected_total_tokens: int

    # Per-region results
    region_results: List[Dict[str, Any]] = field(default_factory=list)

    def precision(self) -> float:
        """Overall precision across all predictions."""
        total = self.total_tp + self.total_fp + self.total_mm
        return self.total_tp / total if total > 0 else 0.0

    def recall(self) -> float:
        """Overall recall across all predictions."""
        total = self.total_tp + self.total_fn + self.total_mm
        return self.total_tp / total if total > 0 else 0.0

    def f1_score(self) -> float:
        """Overall F1 score."""
        denom = 2 * self.total_tp + self.total_fp + self.total_fn + self.total_mm
        return 2 * self.total_tp / denom if denom > 0 else 0.0

    def accepted_precision(self) -> float:
        """Precision for accepted predictions only."""
        total = self.accepted_tp + self.accepted_fp + self.accepted_mm
        return self.accepted_tp / total if total > 0 else 0.0

    def accepted_recall(self) -> float:
        """Recall for accepted predictions only."""
        total = self.accepted_tp + self.accepted_fn + self.accepted_mm
        return self.accepted_tp / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def create_empty_metrics() -> EvaluationMetrics:
    """Create an empty EvaluationMetrics instance with default values."""
    return EvaluationMetrics(
        total_ground_truth=0,
        total_predicted=0,
        exact_matches=0,
        correct_op_wrong_range=0,
        wrong_op=0,
        attribute_mismatch_matches=0,
        lookahead_window=None,
        lookahead_matches=0,
        lookahead_flex_matches=0,
        lookahead_accuracy=0.0,
        lookahead_recall=0.0,
        lookahead_flex_accuracy=0.0,
        lookahead_flex_recall=0.0,
        match_positions=[],
        final_state_tp=0,
        final_state_fp=0,
        final_state_fn=0,
        final_state_mm=0,
        final_state_precision=0.0,
        final_state_recall=0.0,
        final_state_f1_score=0.0,
        final_state_ops_diff=0,
        final_state_ops_saved=0,
        missing_operations=0,
        extra_operations=0,
        details={},
    )


__all__ = [
    "EvaluationMetrics",
    "DocumentMetrics",
    "create_empty_metrics",
]
