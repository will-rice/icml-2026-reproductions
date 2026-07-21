"""
Step Evaluator Module
Evaluates predicted operations against ground truth using state-based comparison.
"""

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.core.state import StateBuilder
from next_action_pred_eval.core.symbolic import (
    operations_to_symbolic,
    symbolic_to_operations,
)
from next_action_pred_eval.core.operations import get_cells_in_range
from next_action_pred_eval.evaluation.metrics import EvaluationMetrics, create_empty_metrics
from next_action_pred_eval.evaluation.state_comparator import (
    StateComparator,
    ComparisonResult,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from evaluating predictions."""

    metrics: EvaluationMetrics
    """Computed evaluation metrics."""

    ground_truth_operations: List[Operation]
    """Ground truth operations for the segment."""

    predicted_operations: List[Operation]
    """Predicted operations."""

    matched_pairs: List[Tuple[Optional[Operation], Optional[Operation], str]]
    """Matched pairs: (gt_op, pred_op, match_type)."""

    ground_truth_workbook: str
    """Path or description of ground truth workbook."""

    predicted_workbook: str
    """Path or description of predicted workbook."""

    execution_metadata: Dict[str, Any]
    """Execution metadata."""

    final_state_comparison: Optional[ComparisonResult] = None
    """State comparison result."""


class StepEvaluator:
    """
    Evaluates predicted operations against ground truth.

    Uses state-based comparison to calculate TP, FP, FN metrics.
    """

    def __init__(self):
        """Initialize StepEvaluator."""
        self._state_comparator = StateComparator(ignore_defaults=True)
        logger.debug("StepEvaluator initialized (state-based mode)")

    def evaluate(
        self,
        ground_truth_operations: List[Union[Operation, str]],
        predicted_operations: List[Union[Operation, str]],
        lookahead_window: Optional[int] = None,
        all_future_operations: Optional[List[Union[Operation, str]]] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        initial_state_cache: Optional[Dict] = None,
        lookahead_state_cache: Optional[Dict] = None,
        skip_ops_diff: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate predicted operations against ground truth.

        Args:
            ground_truth_operations: Ground truth operations for the segment.
            predicted_operations: Predicted operations.
            lookahead_window: Lookahead window size (None=full, 0=disabled).
            all_future_operations: All operations after region start.
            input_tokens: Input tokens used in prediction.
            output_tokens: Output tokens generated.
            total_tokens: Total tokens used.
            initial_state_cache: Pre-computed initial state.
            lookahead_state_cache: Pre-computed lookahead/final state.

        Returns:
            EvaluationResult with metrics and details.
        """
        logger.debug("Starting evaluation")

        # Convert to operations if needed
        gt_ops = self._to_operations(ground_truth_operations)
        pred_ops = self._to_operations(predicted_operations)

        # Build states incrementally
        if initial_state_cache is not None:
            base_state_pred = initial_state_cache.copy()
        else:
            base_state_pred = StateBuilder().state

        # Build predicted state (raise_on_error=False because LLM predictions
        # may contain invalid cell ranges that pass from_symbolic() but fail
        # during apply_to_state(); skipping them keeps evaluation running)
        state_builder_pred = StateBuilder(base_state_pred)
        pred_state = state_builder_pred.apply_operations(pred_ops, raise_on_error=False)

        # Operation-level comparison (secondary metrics)
        matched_pairs, match_stats = self._compare_operations(gt_ops, pred_ops)

        # Prepare lookahead operations
        lookahead_ops = self._prepare_lookahead_ops(
            gt_ops, all_future_operations, lookahead_window
        )

        # Perform lookahead matching
        lookahead_stats = (
            self._perform_lookahead_matching(pred_ops, lookahead_ops, lookahead_window)
            if lookahead_window != 0
            else None
        )

        # Build and compare final states
        final_state_comparison = None
        if lookahead_window != 0 and lookahead_ops:
            logger.debug("Building states for final comparison")

            if lookahead_state_cache is not None:
                lookahead_state = lookahead_state_cache
            else:
                if initial_state_cache is not None:
                    state_builder_lookahead = StateBuilder(initial_state_cache.copy())
                else:
                    state_builder_lookahead = StateBuilder()
                lookahead_state = state_builder_lookahead.apply_operations(lookahead_ops)

            # Restrict comparison to cells touched by operations
            final_cells = self._collect_operation_cells(
                pred_ops
            ) | self._collect_operation_cells(lookahead_ops)
            filtered_pred_state = self._restrict_state_to_cells(pred_state, final_cells)
            filtered_lookahead_state = self._restrict_state_to_cells(
                lookahead_state, final_cells
            )

            final_state_comparison = self._state_comparator.compare(
                filtered_pred_state, filtered_lookahead_state,
                skip_ops_diff=skip_ops_diff,
            )

            # --- Delta-based precision ---
            # Only count properties the prediction actually changed (vs pre-state).
            # Without this, history TPs inflate precision, letting bad predictions
            # slip through acceptance heuristics.
            pred_touched_cells = self._collect_operation_cells(pred_ops)
            if pred_touched_cells:
                pre_state = (
                    initial_state_cache
                    if initial_state_cache is not None
                    else StateBuilder().state
                )
                filtered_pre = self._restrict_state_to_cells(
                    pre_state, pred_touched_cells
                )
                filtered_post_delta = self._restrict_state_to_cells(
                    pred_state, pred_touched_cells
                )

                # Step 1: find properties that changed (post ≠ pre)
                pre_vs_post = self._state_comparator.compare(
                    filtered_post_delta, filtered_pre,
                    skip_ops_diff=True,
                )
                delta_props = {
                    (d.sheet, d.cell, d.property_path)
                    for d in pre_vs_post.differences
                    if d.match_type != "TP"
                }

                # Step 2: reclassify using post-vs-target comparison
                diffs_by_key = {
                    (d.sheet, d.cell, d.property_path): d.match_type
                    for d in final_state_comparison.differences
                }
                delta_tp = 0
                delta_fp = 0
                delta_mm = 0
                for key in delta_props:
                    match_type = diffs_by_key.get(key)
                    if match_type is None:
                        # Both post and target are default — correct match
                        delta_tp += 1
                    elif match_type == "TP":
                        delta_tp += 1
                    elif match_type == "FP":
                        delta_fp += 1
                    elif match_type == "MISMATCH":
                        delta_mm += 1
                    elif match_type == "FN":
                        # Prediction removed something the target needs
                        delta_mm += 1

                # Overwrite precision-relevant counts; keep FN from full comparison
                final_state_comparison.true_positives = delta_tp
                final_state_comparison.false_positives = delta_fp
                final_state_comparison.mismatches = delta_mm
                final_state_comparison.total_properties_predicted = (
                    delta_tp + delta_fp + delta_mm
                )

        # Calculate metrics
        metrics = self._calculate_metrics(
            gt_ops,
            pred_ops,
            match_stats,
            lookahead_window,
            lookahead_stats,
            final_state_comparison,
            input_tokens,
            output_tokens,
            total_tokens,
        )

        result = EvaluationResult(
            metrics=metrics,
            ground_truth_operations=gt_ops,
            predicted_operations=pred_ops,
            matched_pairs=matched_pairs,
            ground_truth_workbook="N/A (state-based)",
            predicted_workbook="N/A (state-based)",
            execution_metadata={"mode": "state_based"},
            final_state_comparison=final_state_comparison,
        )

        logger.debug(
            f"Evaluation complete: precision={metrics.final_state_precision:.2%}, "
            f"recall={metrics.final_state_recall:.2%}, "
            f"ops_saved={metrics.final_state_ops_saved}"
        )

        return result

    def _to_operations(
        self, operations: List[Union[Operation, str]]
    ) -> List[Operation]:
        """Convert to operations if needed."""
        if not operations:
            return []
        if isinstance(operations[0], str):
            return symbolic_to_operations(operations)
        return operations

    def _compare_operations(
        self, gt_ops: List[Operation], pred_ops: List[Operation]
    ) -> Tuple[
        List[Tuple[Optional[Operation], Optional[Operation], str]], Dict[str, int]
    ]:
        """
        Compare operations directly to find matches.

        Returns (matched_pairs, statistics).
        """
        matched_pairs = []
        match_stats = {
            "exact_matches": 0,
            "attribute_mismatch": 0,
            "correct_op_wrong_range": 0,
            "wrong_op": 0,
        }

        gt_matched = set()
        pred_matched = set()

        # First pass: exact matches
        for i, gt_op in enumerate(gt_ops):
            for j, pred_op in enumerate(pred_ops):
                if j in pred_matched:
                    continue

                if self._operations_match(gt_op, pred_op, strict=True, exact=True):
                    matched_pairs.append((gt_op, pred_op, "exact_match"))
                    match_stats["exact_matches"] += 1
                    gt_matched.add(i)
                    pred_matched.add(j)
                    break

        # Second pass: attribute mismatch
        for i, gt_op in enumerate(gt_ops):
            if i in gt_matched:
                continue

            # Skip input operations - must match exactly
            if type(gt_op).__name__ in ("SetValue", "SetFormula", "SetInput"):
                continue

            for j, pred_op in enumerate(pred_ops):
                if j in pred_matched:
                    continue

                if (
                    type(gt_op) == type(pred_op)
                    and gt_op.cell_range == pred_op.cell_range
                ):
                    matched_pairs.append((gt_op, pred_op, "attribute_mismatch"))
                    match_stats["attribute_mismatch"] += 1
                    gt_matched.add(i)
                    pred_matched.add(j)
                    break

        # Third pass: correct operation type, wrong range
        for i, gt_op in enumerate(gt_ops):
            if i in gt_matched:
                continue

            for j, pred_op in enumerate(pred_ops):
                if j in pred_matched:
                    continue

                if type(gt_op) == type(pred_op):
                    matched_pairs.append((gt_op, pred_op, "correct_op_wrong_range"))
                    match_stats["correct_op_wrong_range"] += 1
                    gt_matched.add(i)
                    pred_matched.add(j)
                    break

        # Fourth pass: wrong operation
        for i, gt_op in enumerate(gt_ops):
            if i in gt_matched:
                continue

            for j, pred_op in enumerate(pred_ops):
                if j in pred_matched:
                    continue

                matched_pairs.append((gt_op, pred_op, "wrong_op"))
                match_stats["wrong_op"] += 1
                gt_matched.add(i)
                pred_matched.add(j)
                break

        # Add unmatched operations
        for i, gt_op in enumerate(gt_ops):
            if i not in gt_matched:
                matched_pairs.append((gt_op, None, "missing_in_prediction"))

        for j, pred_op in enumerate(pred_ops):
            if j not in pred_matched:
                matched_pairs.append((None, pred_op, "extra_in_prediction"))

        return matched_pairs, match_stats

    def _prepare_lookahead_ops(
        self,
        gt_ops: List[Operation],
        all_future_operations: Optional[List[Union[Operation, str]]],
        lookahead_window: Optional[int],
    ) -> List[Operation]:
        """Prepare lookahead operations for matching."""
        if lookahead_window == 0 or all_future_operations is None:
            return []

        all_ops = self._to_operations(all_future_operations)

        if lookahead_window is None:
            return all_ops
        else:
            region_size = len(gt_ops)
            return all_ops[: region_size + lookahead_window]

    def _perform_lookahead_matching(
        self,
        pred_ops: List[Operation],
        lookahead_ops: List[Operation],
        lookahead_window: Optional[int],
    ) -> Dict[str, Any]:
        """Perform lookahead matching."""
        if not lookahead_ops:
            return {
                "exact_matches": 0,
                "flexible_matches": 0,
                "match_positions": [],
                "total_lookahead_ops": 0,
            }

        exact_matches = 0
        flexible_matches = 0
        match_positions = []
        matched_gt_indices = set()

        for pred_op in pred_ops:
            best_match_pos = None
            best_match_type = None

            for i, gt_op in enumerate(lookahead_ops):
                if i in matched_gt_indices:
                    continue

                if self._operations_match(gt_op, pred_op, strict=True):
                    best_match_pos = i
                    best_match_type = "exact"
                    break
                elif self._operations_match(gt_op, pred_op, strict=False):
                    if best_match_type != "exact":
                        best_match_pos = i
                        best_match_type = "flexible"

            if best_match_pos is not None:
                matched_gt_indices.add(best_match_pos)
                match_positions.append(best_match_pos)
                if best_match_type == "exact":
                    exact_matches += 1
                    flexible_matches += 1
                else:
                    flexible_matches += 1

        return {
            "exact_matches": exact_matches,
            "flexible_matches": flexible_matches,
            "match_positions": match_positions,
            "total_lookahead_ops": len(lookahead_ops),
        }

    def _operations_match(
        self,
        op1: Operation,
        op2: Operation,
        exact: bool = True,
        strict: bool = True,
    ) -> bool:
        """Check if two operations match."""
        if op1 == op2:
            return True

        if type(op1) != type(op2):
            return False

        # Check ranges intersect
        if op1.cell_range.sheet != op2.cell_range.sheet:
            return False

        # Simple range overlap check
        try:
            c1 = op1.cell_range.get_coordinates()
            c2 = op2.cell_range.get_coordinates()
            if not (c1[0] <= c2[2] and c2[0] <= c1[2] and c1[1] <= c2[3] and c2[1] <= c1[3]):
                return False
        except (TypeError, ValueError):
            return False

        if strict:
            return op1.value == op2.value
        else:
            return True

    def _collect_operation_cells(
        self, operations: List[Operation]
    ) -> set:
        """Return all (sheet, cell) pairs touched by operations."""
        cells = set()
        if not operations:
            return cells

        for op in operations:
            cell_range = getattr(op, "cell_range", None)
            if cell_range is None:
                continue
            try:
                for cell_addr in get_cells_in_range(cell_range):
                    cells.add((cell_range.sheet, cell_addr))
            except Exception:
                pass

        return cells

    def _restrict_state_to_cells(
        self, state: Dict[str, Any], allowed_cells: set
    ) -> Dict[str, Any]:
        """Create a copy of state containing only allowed cells."""
        if not state or not allowed_cells:
            return {"worksheets": {}}

        worksheets = state.get("worksheets", {})
        grouped = {}
        for sheet_name, cell in allowed_cells:
            grouped.setdefault(sheet_name, set()).add(cell)

        filtered = {"worksheets": {}}
        for sheet_name, cells in grouped.items():
            sheet = worksheets.get(sheet_name)
            if not sheet:
                continue

            cell_entries = sheet.get("cells", {})
            filtered_cells = {
                cell: cell_entries[cell] for cell in cells if cell in cell_entries
            }
            if not filtered_cells:
                continue

            new_sheet = {"cells": filtered_cells}
            if "worksheetProperties" in sheet:
                new_sheet["worksheetProperties"] = sheet["worksheetProperties"]
            filtered["worksheets"][sheet_name] = new_sheet

        return filtered if filtered["worksheets"] else {"worksheets": {}}

    def _calculate_metrics(
        self,
        gt_ops: List[Operation],
        pred_ops: List[Operation],
        match_stats: Dict[str, int],
        lookahead_window: Optional[int],
        lookahead_stats: Optional[Dict[str, Any]],
        final_state_comparison: Optional[ComparisonResult],
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ) -> EvaluationMetrics:
        """Calculate evaluation metrics."""
        total_gt = len(gt_ops)
        total_pred = len(pred_ops)

        exact = match_stats["exact_matches"]
        attr_mismatch = match_stats["attribute_mismatch"]
        correct_op_wrong_range = match_stats["correct_op_wrong_range"]
        wrong_op = match_stats["wrong_op"]

        # Lookahead metrics
        if lookahead_stats:
            lookahead_exact = lookahead_stats["exact_matches"]
            lookahead_flex = lookahead_stats["flexible_matches"]
            lookahead_positions = lookahead_stats["match_positions"]
            total_lookahead = lookahead_stats["total_lookahead_ops"]

            lookahead_accuracy = lookahead_exact / total_pred if total_pred > 0 else 0.0
            lookahead_recall = (
                lookahead_exact / total_lookahead if total_lookahead > 0 else 0.0
            )
            lookahead_flex_accuracy = (
                lookahead_flex / total_pred if total_pred > 0 else 0.0
            )
            lookahead_flex_recall = (
                lookahead_flex / total_lookahead if total_lookahead > 0 else 0.0
            )
        else:
            lookahead_exact = 0
            lookahead_flex = 0
            lookahead_positions = []
            total_lookahead = 0
            lookahead_accuracy = 0.0
            lookahead_recall = 0.0
            lookahead_flex_accuracy = 0.0
            lookahead_flex_recall = 0.0

        # Final state metrics (primary)
        if final_state_comparison:
            final_state_tp = final_state_comparison.true_positives
            final_state_fp = final_state_comparison.false_positives
            final_state_fn = final_state_comparison.false_negatives
            final_state_mm = final_state_comparison.mismatches
            final_state_precision = final_state_comparison.precision()
            final_state_recall = final_state_comparison.recall()
            final_state_f1 = final_state_comparison.f1_score()
            final_state_ops_diff = final_state_comparison.ops_diff
            total_final_ops = (
                lookahead_stats["total_lookahead_ops"]
                if lookahead_stats
                else total_gt
            )
            final_state_ops_saved = total_final_ops - final_state_ops_diff
        else:
            final_state_tp = final_state_fp = final_state_fn = 0
            final_state_mm = 0
            final_state_precision = final_state_recall = final_state_f1 = 0.0
            final_state_ops_diff = final_state_ops_saved = 0

        # Missing and extra
        missing = total_gt - (exact + attr_mismatch + correct_op_wrong_range + wrong_op)
        extra = total_pred - (exact + attr_mismatch + correct_op_wrong_range + wrong_op)

        return EvaluationMetrics(
            total_ground_truth=total_gt,
            total_predicted=total_pred,
            exact_matches=exact,
            correct_op_wrong_range=correct_op_wrong_range,
            wrong_op=wrong_op,
            attribute_mismatch_matches=attr_mismatch,
            lookahead_window=lookahead_window,
            lookahead_matches=lookahead_exact,
            lookahead_flex_matches=lookahead_flex,
            lookahead_accuracy=lookahead_accuracy,
            lookahead_recall=lookahead_recall,
            lookahead_flex_accuracy=lookahead_flex_accuracy,
            lookahead_flex_recall=lookahead_flex_recall,
            match_positions=lookahead_positions,
            final_state_tp=final_state_tp,
            final_state_fp=final_state_fp,
            final_state_fn=final_state_fn,
            final_state_mm=final_state_mm,
            final_state_ops_diff=final_state_ops_diff,
            final_state_ops_saved=final_state_ops_saved,
            final_state_precision=final_state_precision,
            final_state_recall=final_state_recall,
            final_state_f1_score=final_state_f1,
            missing_operations=max(0, missing),
            extra_operations=max(0, extra),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            details={
                "match_stats": match_stats,
                "final_state_comparison": {
                    "property_type_stats": (
                        final_state_comparison.property_type_stats
                        if final_state_comparison
                        else {}
                    ),
                    "total_differences": (
                        len(final_state_comparison.differences)
                        if final_state_comparison
                        else 0
                    ),
                },
            },
        )


__all__ = [
    "StepEvaluator",
    "EvaluationResult",
]
