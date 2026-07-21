"""
Evaluation Orchestrator Module
Brings together solver and evaluator to run end-to-end evaluation experiments.

Loop semantics::

    t = 0                 # step pointer in GT
    user_steps = 0        # actual user actions

    while t < len(gt):
        t += 1
        user_steps += 1                     # user always does one step

        pred = make_prediction(gt[:t], ...)  # history = gt[:t]

        if pred is accepted:
            t += len(pred)                   # skip predicted ops
"""

import json
import logging
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.state import StateBuilder
from next_action_pred_eval.core.symbolic import (
    operations_to_symbolic,
    symbolic_to_operations,
)
from next_action_pred_eval.evaluation.solver import ISolver, PredictionResult
from next_action_pred_eval.evaluation.stride import StrideConfig, StrideMode
from next_action_pred_eval.evaluation.metrics import EvaluationMetrics
from next_action_pred_eval.evaluation.acceptance import AcceptanceHeuristic
from next_action_pred_eval.evaluation.evaluator import StepEvaluator, EvaluationResult
from next_action_pred_eval.evaluation.future_edits import (
    FutureEditsManager,
    FutureEditsConfig,
)
from next_action_pred_eval.evaluation.state_comparator import StateComparator
from next_action_pred_eval.evaluation.experiment_recorder import ExperimentRecorder

logger = logging.getLogger(__name__)
# Only warnings/errors from orchestrator go to terminal;
# detailed per-prediction info goes to trajectory.log via ExperimentRecorder
logger.setLevel(logging.WARNING)


@dataclass
class ExperimentSummary:
    """Summary of the entire experiment."""

    total_steps: int
    """Total steps (operations) in the original action stream."""

    total_predictions: int
    """Number of non-empty predictions made."""

    total_accepted: int
    """Number of accepted predictions."""

    total_rejected: int
    """Number of rejected predictions."""

    acceptance_rate: float
    """Accepted / total predictions."""

    # Aggregate metrics
    total_tp: int
    total_fp: int
    total_fn: int
    total_ops_saved: int

    # Additional coverage
    total_mm: int = 0
    total_inverse_ops_added: int = 0
    total_ops_removed_via_dedup: int = 0

    # User effort
    user_steps_taken: int = 0
    uas_pct: float = 0.0
    """(initial_length - user_steps) / initial_length."""

    # Formatting actions saved (UFAS)
    total_formatting_ops: int = 0
    """Total formatting ops in original trajectory (non-INPUT/VALUE/FORMULA)."""
    user_formatting_steps: int = 0
    """Formatting ops the user had to manually execute."""
    ufas: int = 0
    """User Formatting Actions Saved = total_formatting_ops - user_formatting_steps."""
    ufas_pct: float = 0.0
    """UFAS / total_formatting_ops."""

    # Timing
    total_time: float = 0.0
    avg_prediction_time: float = 0.0

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    # Final trajectory length (after online edits)
    final_sequence_length: int = 0

    # Per-prediction details
    prediction_details: List[Dict[str, Any]] = field(default_factory=list)

    # Property-level breakdown
    property_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Per-heuristic acceptance stats (offline multi-heuristic evaluation)
    per_heuristic_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Maps heuristic name → {accepted, rejected, acceptance_rate, total_ops_saved}."""

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def summary_str(self) -> str:
        """Return human-readable summary."""
        return (
            f"Experiment Summary:\n"
            f"  Total steps: {self.total_steps}\n"
            f"  Predictions: {self.total_predictions} "
            f"(accepted={self.total_accepted}, rejected={self.total_rejected})\n"
            f"  Acceptance rate: {self.acceptance_rate:.2%}\n"
            f"  User steps: {self.user_steps_taken} | UAS: {self.uas_pct:.2%}\n"
            f"  Total ops saved: {self.total_ops_saved}\n"
            f"  Time: {self.total_time:.2f}s "
            f"(avg prediction: {self.avg_prediction_time:.2f}s)\n"
            f"  Tokens: {self.total_tokens} "
            f"(input={self.total_input_tokens}, output={self.total_output_tokens})"
        )


@dataclass
class WorkbookStateTracker:
    """Tracks workbook states during evaluation."""

    initial_state: Dict[str, Any]
    """State at start of evaluation."""

    current_state: Dict[str, Any]
    """Current state (updated as operations are applied)."""

    final_state: Dict[str, Any]
    """Target final state."""

    @classmethod
    def initialize(cls, all_operations: List[Operation]) -> "WorkbookStateTracker":
        """Initialize tracker from operations."""
        base_builder = StateBuilder()
        initial_state = base_builder.get_state()
        final_builder = StateBuilder(initial_state)
        final_state = final_builder.apply_operations(all_operations)
        return cls(
            initial_state=deepcopy(initial_state),
            current_state=deepcopy(initial_state),
            final_state=final_state,
        )

    def apply_operations(
        self, operations: List[Operation], raise_on_error: bool = True,
    ) -> Dict[str, Any]:
        """Apply operations and update current state."""
        if not operations:
            return self.current_state
        builder = StateBuilder(self.current_state)
        self.current_state = builder.apply_operations(
            operations, raise_on_error=raise_on_error,
        )
        return self.current_state

    def snapshot_current(self) -> Dict[str, Any]:
        """Return a copy of current state."""
        return deepcopy(self.current_state)


class Orchestrator:
    """
    Orchestrates the entire prediction and evaluation workflow.

    The main loop follows the corrected semantics where the user always
    takes one step first, then a prediction is made on the updated history.
    """

    def __init__(
        self,
        solver: ISolver,
        stride_config: Optional[StrideConfig] = None,
        acceptance_heuristics: Optional[List[AcceptanceHeuristic]] = None,
        output_dir: Union[str, Path] = "outputs/evaluation",
        future_edits_config: Optional[FutureEditsConfig] = None,
        save_prediction_folders: bool = False,
        skip_simulation_on_low_precision: bool = False,
        repredict_after_accept: bool = False,
        max_predictions_per_step: Optional[int] = None,
        buffered_writes: bool = False,
    ):
        self.solver = solver
        self.stride_config = stride_config or StrideConfig()
        self.acceptance_heuristics = acceptance_heuristics or []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_prediction_folders = save_prediction_folders
        self.skip_simulation_on_low_precision = skip_simulation_on_low_precision
        self.repredict_after_accept = repredict_after_accept
        self.max_predictions_per_step = max_predictions_per_step
        self.buffered_writes = buffered_writes

        # Components
        self.evaluator = StepEvaluator()
        self.future_edits_manager = FutureEditsManager(
            config=future_edits_config or FutureEditsConfig()
        )
        self.state_comparator = StateComparator(ignore_defaults=True)

        logger.info(
            "Orchestrator initialized: stride=%s, heuristics=%d, prediction_folders=%s",
            self.stride_config, len(self.acceptance_heuristics),
            self.save_prediction_folders,
        )

    def run_experiment(
        self,
        action_stream: List[Union[Operation, str]],
        experiment_name: str = "experiment",
        max_context_ops: Optional[int] = None,
        online_mode: bool = False,
        max_steps: Optional[int] = None,
    ) -> ExperimentSummary:
        """
        Run a prediction evaluation experiment.

        Args:
            action_stream: Full sequence of operations to evaluate.
            experiment_name: Name for this experiment run.
            max_context_ops: Max operations to include in prediction context.
            online_mode: If True, update ground truth on accepted predictions.
            max_steps: Maximum steps to process (None = all).

        Returns:
            ExperimentSummary with aggregated results.
        """
        start_time = time.time()

        # Convert to operations if needed
        if action_stream and isinstance(action_stream[0], str):
            all_operations = symbolic_to_operations(action_stream)
        else:
            all_operations = list(action_stream)

        initial_length = len(all_operations)

        # Extract sheet name from first operation (single-sheet trajectories)
        trajectory_sheet_name = None
        if all_operations:
            trajectory_sheet_name = all_operations[0].cell_range.sheet

        state_tracker = WorkbookStateTracker.initialize(all_operations)

        # Create recorder for per-trajectory output files
        recorder = ExperimentRecorder(
            self.output_dir, experiment_name,
            buffered_writes=self.buffered_writes,
        )
        if self.save_prediction_folders:
            recorder.enable_prediction_folders()
        recorder.save_target_state(state_tracker.final_state)

        # Reset solver
        self.solver.reset()

        # Tracking variables
        current_gt = list(all_operations)
        all_gt_symbolic = operations_to_symbolic(all_operations)
        prediction_details: List[Dict[str, Any]] = []
        total_predictions = 0
        total_accepted = 0
        total_rejected = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_mm = 0
        total_ops_saved = 0
        total_inverse_ops_added = 0
        total_ops_removed_via_dedup = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        prediction_times: List[float] = []
        prediction_index = 0
        divergence_count = 0

        # UFAS tracking: count formatting ops (non-INPUT/VALUE/FORMULA)
        _CONTENT_OP_TYPES = {"INPUT", "VALUE", "FORMULA"}
        total_formatting_ops = sum(
            1 for s in all_gt_symbolic
            if s.split("|")[0].strip() not in _CONTENT_OP_TYPES
        )
        user_formatting_steps = 0

        # Property-level accumulation
        property_breakdown: Dict[str, Dict[str, int]] = {}

        # Per-heuristic tracking (offline multi-heuristic evaluation)
        per_heuristic_counters: Dict[str, Dict[str, Any]] = {}
        if not online_mode and self.acceptance_heuristics:
            for h in self.acceptance_heuristics:
                per_heuristic_counters[h.name] = {
                    "accepted": 0,
                    "rejected": 0,
                    "total_ops_saved": 0,
                }

        # Attribution tracking: list of (op_symbolic, source, prediction_index|None)
        attribution: List[Dict[str, Any]] = []

        # --- Main loop ---
        # t = step pointer, user_steps = actual user actions
        t = 0
        user_steps = 0
        last_accepted = False
        step_limit = max_steps if max_steps else len(current_gt)

        while t < len(current_gt) and user_steps < step_limit:
            # When repredict_after_accept is on and the last prediction was
            # accepted, skip the user pop and let the solver predict again
            # immediately (chained prediction).
            skip_pop = self.repredict_after_accept and last_accepted

            if not skip_pop:
                # User always executes gt[t]
                user_op = current_gt[t]
                user_op_symbolic = operations_to_symbolic([user_op])[0]
                t += 1
                user_steps += 1

                # Track formatting step
                if user_op_symbolic.split("|")[0].strip() not in _CONTENT_OP_TYPES:
                    user_formatting_steps += 1

                # Record user step
                history_len = t
                future_len = len(current_gt) - t
                recorder.record_user_step(
                    t=t, user_step=user_steps,
                    op_symbolic=user_op_symbolic,
                    history_len=history_len, future_len=future_len,
                )

                # Track attribution for user step
                attribution.append({
                    "index": len(attribution),
                    "op": user_op_symbolic,
                    "source": "user",
                    "user_step": user_steps,
                })

                # Apply user's step to state
                state_tracker.apply_operations([user_op])

            # Reset last_accepted — will be set again only if this
            # iteration's prediction is accepted.
            last_accepted = False

            # Check stride: should we predict now?
            # When skip_pop is True we are in a repredict-after-accept
            # iteration — always attempt a prediction regardless of stride.
            history = current_gt[:t]
            if not skip_pop and not self.stride_config.should_predict(
                t, history, None, total_predictions
            ):
                continue

            # Build context for prediction
            history_symbolic = operations_to_symbolic(history)
            context_ops: List[str] = history_symbolic
            if max_context_ops and len(context_ops) > max_context_ops:
                context_ops = context_ops[-max_context_ops:]

            # Make prediction
            pred_start = time.time()
            logger.debug(
                "predict() start: t=%d user_step=%d ctx_ops=%d total_preds=%d",
                t, user_steps, len(context_ops), total_predictions,
            )
            try:
                prediction_result = self.solver.predict(
                    previous_actions=context_ops,
                    workbook_state=state_tracker.snapshot_current(),
                    context={
                        "step_idx": t,
                        "experiment_name": experiment_name,
                        "user_step": user_steps,
                        "sheet_name": trajectory_sheet_name,
                    },
                )
            except RuntimeError:
                raise
            except Exception as e:
                recorder.record_error("prediction_failure", t, str(e))
                logger.warning("predict() FAILED at t=%d: %s", t, e)
                continue

            pred_time = time.time() - pred_start
            logger.debug(
                "predict() done: t=%d in %.2fs, n_pred_ops=%d",
                t, pred_time, len(prediction_result.predicted_operations),
            )
            prediction_times.append(pred_time)

            pred_ops = prediction_result.predicted_operations
            pred_symbolic = prediction_result.predicted_symbolic or operations_to_symbolic(pred_ops)

            # Truncate to max_predictions_per_step if configured
            if self.max_predictions_per_step and len(pred_ops) > self.max_predictions_per_step:
                pred_ops = pred_ops[:self.max_predictions_per_step]
                pred_symbolic = pred_symbolic[:self.max_predictions_per_step]

            tokens = {
                "input": prediction_result.input_tokens,
                "output": prediction_result.output_tokens,
                "total": prediction_result.total_tokens,
            }

            # Track parse errors from solver metadata
            parse_failures = prediction_result.metadata.get("parse_failures", [])
            if parse_failures:
                n_failed = len(parse_failures)
                n_total = len(pred_symbolic) + n_failed  # valid + failed
                recorder.record_partial_parse(
                    t=t,
                    n_failed=n_failed,
                    n_total=n_total,
                    failed_details=parse_failures,
                )

            # Empty prediction
            if not pred_ops:
                error_reason = prediction_result.metadata.get("error")
                recorder.record_empty_prediction(
                    t=t, user_step=user_steps,
                    tokens=tokens, generation_time_s=pred_time,
                    error_reason=error_reason,
                )
                total_input_tokens += prediction_result.input_tokens
                total_output_tokens += prediction_result.output_tokens
                total_tokens += prediction_result.total_tokens
                continue

            prediction_index += 1
            total_predictions += 1
            total_input_tokens += prediction_result.input_tokens
            total_output_tokens += prediction_result.output_tokens
            total_tokens += prediction_result.total_tokens

            # Evaluate prediction against GT segment, but compare state
            # against the FULL remaining future (matching reference behavior).
            gt_segment = current_gt[t: t + len(pred_ops)]
            gt_segment_symbolic = operations_to_symbolic(gt_segment)
            all_future_ops = current_gt[t:]  # all remaining ops from current pointer

            current_state_snapshot = state_tracker.snapshot_current()
            eval_result = self.evaluator.evaluate(
                ground_truth_operations=gt_segment,
                predicted_operations=pred_ops,
                lookahead_window=None,
                all_future_operations=all_future_ops,
                input_tokens=prediction_result.input_tokens,
                output_tokens=prediction_result.output_tokens,
                total_tokens=prediction_result.total_tokens,
                initial_state_cache=current_state_snapshot,
                lookahead_state_cache=state_tracker.final_state,
                skip_ops_diff=True,
            )

            # Simulate future edits (future-if-accepted simulation) to compute
            # operation-count-based ops_saved BEFORE acceptance check. This matches
            # the reference's _apply_undo_summary which overwrites
            # final_state_ops_saved with future_before_count - final_future_count
            # from simulate_future_edits.
            acceptance_sim_result = None
            acceptance_sim_new_gt = None

            # Optional early gate: skip simulation if precision already below
            # acceptance threshold (prediction will be rejected regardless of
            # ops_saved). OFF by default to preserve full future_if_accepted data.
            should_simulate = True
            if self.skip_simulation_on_low_precision:
                min_prec = self._get_min_precision_threshold()
                if eval_result.metrics.final_state_precision < min_prec:
                    should_simulate = False

            if should_simulate:
                try:
                    acceptance_sim_new_gt, acceptance_sim_result = (
                        self.future_edits_manager.simulate_future_edits(
                            current_gt=current_gt,
                            start_idx=t,
                            end_idx=t,
                            predicted_ops=pred_ops,
                            eval_result=eval_result,
                            initial_state=current_state_snapshot,
                            final_target_state=state_tracker.final_state,
                        )
                    )
                    self._apply_undo_summary(eval_result.metrics, acceptance_sim_result)
                except Exception as e:
                    logger.debug("Future-if-accepted simulation failed at t=%d: %s", t, e)

            # Check acceptance (uses overwritten ops_saved from future-if-accepted simulation)
            accepted = self._check_acceptance(eval_result.metrics)
            heuristic_details = self._get_heuristic_details(eval_result.metrics)

            # Per-heuristic tracking for offline multi-heuristic evaluation
            per_heuristic_accepted = {}
            if per_heuristic_counters:
                for h in self.acceptance_heuristics:
                    h_accepted = h.evaluate(eval_result.metrics)
                    per_heuristic_accepted[h.name] = h_accepted
                    if h_accepted:
                        per_heuristic_counters[h.name]["accepted"] += 1
                        per_heuristic_counters[h.name]["total_ops_saved"] += (
                            eval_result.metrics.final_state_ops_saved
                        )
                    else:
                        per_heuristic_counters[h.name]["rejected"] += 1

            # Extract metrics
            metrics = eval_result.metrics
            eval_metrics_dict = {
                "exact_matches": metrics.exact_matches,
                "correct_op_wrong_range": metrics.correct_op_wrong_range,
                "wrong_op": metrics.wrong_op,
                "final_state_tp": metrics.final_state_tp,
                "final_state_fp": metrics.final_state_fp,
                "final_state_fn": metrics.final_state_fn,
                "final_state_mm": metrics.final_state_mm,
                "final_state_precision": metrics.final_state_precision,
                "final_state_recall": metrics.final_state_recall,
                "final_state_ops_saved": metrics.final_state_ops_saved,
                "lookahead_matches": metrics.lookahead_matches,
                "lookahead_accuracy": metrics.lookahead_accuracy,
                "undo_gain_from_final_state": metrics.undo_gain_from_final_state,
                "undo_gain_from_dedup": metrics.undo_gain_from_dedup,
                "undo_inverse_ops": metrics.undo_inverse_ops,
                "undo_net_ops_saved": metrics.undo_net_ops_saved,
            }

            # Update tracking
            total_tp += metrics.final_state_tp
            total_fp += metrics.final_state_fp
            total_fn += metrics.final_state_fn
            total_mm += metrics.final_state_mm

            # Accumulate property breakdown from eval details
            if hasattr(eval_result, "final_state_comparison") and eval_result.final_state_comparison:
                comp = eval_result.final_state_comparison
                if hasattr(comp, "property_type_stats"):
                    for prop_type, stats in comp.property_type_stats.items():
                        if prop_type not in property_breakdown:
                            property_breakdown[prop_type] = {"tp": 0, "fp": 0, "fn": 0, "mm": 0}
                        property_breakdown[prop_type]["tp"] += stats.get("TP", 0)
                        property_breakdown[prop_type]["fp"] += stats.get("FP", 0)
                        property_breakdown[prop_type]["fn"] += stats.get("FN", 0)
                        property_breakdown[prop_type]["mm"] += stats.get("MISMATCH", 0)

            # Build future_if_accepted and handle acceptance.
            # Start with the simulation data from the preview run (used for
            # ops_saved).  Accepted+online predictions will enrich this below.
            future_if_accepted = None
            if acceptance_sim_result and isinstance(acceptance_sim_result, dict):
                future_if_accepted = {
                    "metadata": acceptance_sim_result.get("metadata", {}),
                    "summary": acceptance_sim_result.get("summary", {}),
                    "ops_removed": operations_to_symbolic(
                        acceptance_sim_result.get("operations_removed", [])
                    ) if acceptance_sim_result.get("operations_removed") else [],
                    "inverse_ops_added": operations_to_symbolic(
                        acceptance_sim_result.get("inverse_ops_added", [])
                    ) if acceptance_sim_result.get("inverse_ops_added") else [],
                    "old_length": acceptance_sim_result.get("old_length"),
                    "new_length": acceptance_sim_result.get("new_length"),
                }
            gt_len_before = None
            gt_len_after = None

            if accepted:
                total_accepted += 1
                total_ops_saved += metrics.final_state_ops_saved

                if online_mode:
                    gt_len_before = len(current_gt)

                    # Reuse the pre-computed GT from simulate_future_edits
                    # (since end_idx is ignored by v3, both compute the same result)
                    if acceptance_sim_new_gt is not None and acceptance_sim_result is not None:
                        new_gt = acceptance_sim_new_gt
                        changes = acceptance_sim_result
                    else:
                        # Fallback: run apply_future_edits if simulation didn't run
                        try:
                            end_idx = min(t + len(pred_ops), len(current_gt))
                            new_gt, changes = self.future_edits_manager.apply_future_edits(
                                current_gt=current_gt,
                                start_idx=t,
                                end_idx=end_idx,
                                predicted_ops=pred_ops,
                                eval_result=eval_result,
                                initial_state=state_tracker.snapshot_current(),
                                final_target_state=state_tracker.final_state,
                            )
                        except Exception as e:
                            logger.error("Future edits failed at t=%d: %s", t, e)
                            # Treat as divergence — reject and continue
                            recorder.record_divergence(
                                prediction_index=prediction_index, t=t,
                                description=f"Future edits error: {e}",
                                action="rejected",
                            )
                            divergence_count += 1
                            total_accepted -= 1
                            total_ops_saved -= metrics.final_state_ops_saved
                            total_rejected += 1
                        # Record prediction as rejected
                        recorder.record_prediction(
                            prediction_index=prediction_index, t=t, user_step=user_steps,
                            pred_ops_symbolic=pred_symbolic, gt_segment_symbolic=gt_segment_symbolic,
                            eval_metrics=eval_metrics_dict, accepted=False,
                            heuristic_details=heuristic_details, tokens=tokens,
                            generation_time_s=pred_time,
                            history_tail=history_symbolic[-10:],
                            future_head=operations_to_symbolic(current_gt[t:t + 10]),
                            prompt_messages=prediction_result.metadata.get("prompt_messages"),
                            raw_response=prediction_result.metadata.get("raw_response"),
                            response_metadata=prediction_result.metadata.get("response_metadata"),
                            property_breakdown=self._extract_property_breakdown(eval_result),
                            matched_pairs_summary=self._extract_matched_pairs(eval_result),
                            full_history_context=history_symbolic if self.save_prediction_folders else None,
                        )
                        continue

                    current_gt = new_gt
                    gt_len_after = len(current_gt)

                    # Enrich future_if_accepted with actual apply_future_edits changes
                    changes_summary = changes.get("summary", {})
                    actual_changes = {
                        "gt_len_before": gt_len_before,
                        "gt_len_after": gt_len_after,
                        "ops_removed": operations_to_symbolic(
                            changes.get("operations_removed", [])
                        ) if changes.get("operations_removed") else [],
                        "inverse_ops_added": operations_to_symbolic(
                            changes.get("inverse_ops_added", [])
                        ) if changes.get("inverse_ops_added") else [],
                        "dedup_gain": changes_summary.get("dedup_gain", 0),
                        "inverse_cost": changes_summary.get("inverse_cost", 0),
                        "net_gain": changes_summary.get("net_gain", 0),
                    }
                    if future_if_accepted is None:
                        future_if_accepted = {}
                    future_if_accepted.update(actual_changes)
                    total_inverse_ops_added += changes_summary.get("inverse_cost", 0)
                    total_ops_removed_via_dedup += changes_summary.get("dedup_gain", 0)

                    # Track attribution for predicted ops
                    for op_s in pred_symbolic:
                        attribution.append({
                            "index": len(attribution),
                            "op": op_s,
                            "source": "predicted",
                            "prediction_index": prediction_index,
                            "step_t": t,
                        })

                    # Track attribution for inverse ops
                    inverse_ops = changes.get("inverse_ops_added", [])
                    if inverse_ops:
                        for inv_op in operations_to_symbolic(inverse_ops):
                            attribution.append({
                                "index": len(attribution),
                                "op": inv_op,
                                "source": "inverse",
                                "prediction_index": prediction_index,
                                "step_t": t,
                            })

                    # Apply predicted ops to state and advance pointer
                    state_tracker.apply_operations(pred_ops, raise_on_error=False)
                    t += len(pred_ops)
                # Offline mode: no GT mutation, no pointer jump beyond user step

                last_accepted = True
            else:
                total_rejected += 1

            # Record prediction
            future_ops_for_context = current_gt[t: t + 10]
            recorder.record_prediction(
                prediction_index=prediction_index, t=t, user_step=user_steps,
                pred_ops_symbolic=pred_symbolic,
                gt_segment_symbolic=gt_segment_symbolic,
                eval_metrics=eval_metrics_dict, accepted=accepted,
                heuristic_details=heuristic_details, tokens=tokens,
                generation_time_s=pred_time,
                history_tail=history_symbolic[-10:],
                future_head=operations_to_symbolic(future_ops_for_context),
                future_if_accepted=future_if_accepted,
                gt_len_before=gt_len_before, gt_len_after=gt_len_after,
                prompt_messages=prediction_result.metadata.get("prompt_messages"),
                raw_response=prediction_result.metadata.get("raw_response"),
                response_metadata=prediction_result.metadata.get("response_metadata"),
                property_breakdown=self._extract_property_breakdown(eval_result),
                matched_pairs_summary=self._extract_matched_pairs(eval_result),
                full_history_context=history_symbolic if self.save_prediction_folders else None,
            )

            # Record detail for summary
            detail = {
                "step_t": t,
                "prediction_index": prediction_index,
                "predicted_count": len(pred_ops),
                "accepted": accepted,
                "precision": metrics.final_state_precision,
                "recall": metrics.final_state_recall,
                "ops_saved": metrics.final_state_ops_saved,
                "prediction_time": pred_time,
                "tokens": prediction_result.total_tokens,
            }
            if per_heuristic_accepted:
                detail["per_heuristic_accepted"] = per_heuristic_accepted
            prediction_details.append(detail)

        # --- End of main loop ---

        # Track remaining original ops (not executed by user or prediction)
        for remaining_idx in range(t, len(current_gt)):
            remaining_op_sym = operations_to_symbolic([current_gt[remaining_idx]])[0]
            attribution.append({
                "index": len(attribution),
                "op": remaining_op_sym,
                "source": "original",
            })

        # Write final trajectory with attribution
        recorder.record_final_trajectory(attribution)

        # Compute initial operation breakdown
        initial_op_counts = dict(Counter(
            op_s.split("|")[0].strip() for op_s in all_gt_symbolic
        ))
        final_op_counts = dict(Counter(
            op_s.split("|")[0].strip() for op_s in operations_to_symbolic(current_gt)
        ))

        # Calculate summary metrics
        total_time = time.time() - start_time
        avg_pred_time = (
            sum(prediction_times) / len(prediction_times) if prediction_times else 0.0
        )
        acceptance_rate = (
            total_accepted / total_predictions if total_predictions > 0 else 0.0
        )
        uas_pct = (
            (initial_length - user_steps) / initial_length
            if initial_length > 0 else 0.0
        )

        # UFAS (User Formatting Actions Saved)
        ufas = total_formatting_ops - user_formatting_steps
        ufas_pct = (
            ufas / total_formatting_ops
            if total_formatting_ops > 0 else 0.0
        )

        # Coverage
        coverage_total = total_tp + total_fp + total_fn
        coverage_pct_tp = total_tp / coverage_total if coverage_total > 0 else 0.0

        # Per-heuristic summary stats
        per_heuristic_stats: Dict[str, Dict[str, Any]] = {}
        for h_name, counters in per_heuristic_counters.items():
            h_total = counters["accepted"] + counters["rejected"]
            h_ops_saved = counters["total_ops_saved"]
            per_heuristic_stats[h_name] = {
                "accepted": counters["accepted"],
                "rejected": counters["rejected"],
                "total_predictions": h_total,
                "acceptance_rate": (
                    counters["accepted"] / h_total if h_total > 0 else 0.0
                ),
                "total_ops_saved": h_ops_saved,
                "uas_pct": (
                    h_ops_saved / initial_length if initial_length > 0 else 0.0
                ),
            }

        summary = ExperimentSummary(
            total_steps=initial_length,
            total_predictions=total_predictions,
            total_accepted=total_accepted,
            total_rejected=total_rejected,
            acceptance_rate=acceptance_rate,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            total_ops_saved=total_ops_saved,
            total_mm=total_mm,
            total_inverse_ops_added=total_inverse_ops_added,
            total_ops_removed_via_dedup=total_ops_removed_via_dedup,
            user_steps_taken=user_steps,
            uas_pct=uas_pct,
            total_formatting_ops=total_formatting_ops,
            user_formatting_steps=user_formatting_steps,
            ufas=ufas,
            ufas_pct=ufas_pct,
            total_time=total_time,
            avg_prediction_time=avg_pred_time,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_tokens,
            final_sequence_length=len(current_gt),
            prediction_details=prediction_details,
            property_breakdown=property_breakdown,
            per_heuristic_stats=per_heuristic_stats,
            metadata={
                "experiment_name": experiment_name,
                "stride_config": str(self.stride_config),
                "heuristics": [h.name for h in self.acceptance_heuristics],
                "online_mode": online_mode,
                "solver_config": self.solver.get_config(),
                "user_step_limit_reached": max_steps is not None and user_steps >= max_steps,
                "max_steps_config": max_steps,
                "divergences": divergence_count,
            },
        )

        # Finalize recorder — writes per-trajectory experiment_summary.json
        summary_dict = {
            "file_label": experiment_name,
            "status": "success",
            "initial_sequence_length": initial_length,
            "final_sequence_length": len(current_gt),
            "user_steps_taken": user_steps,
            "predictions_attempted": total_predictions,
            "predictions_accepted": total_accepted,
            "predictions_rejected": total_rejected,
            "operations_predicted": sum(d["predicted_count"] for d in prediction_details),
            "operations_accepted": sum(
                d["predicted_count"] for d in prediction_details if d["accepted"]
            ),
            "operations_rejected": sum(
                d["predicted_count"] for d in prediction_details if not d["accepted"]
            ),
            "net_operations_saved": total_ops_saved,
            "uas_pct": uas_pct,
            "total_formatting_ops": total_formatting_ops,
            "user_formatting_steps": user_formatting_steps,
            "ufas": ufas,
            "ufas_pct": ufas_pct,
            "acceptance_rate": acceptance_rate,
            "coverage": {
                "tp": total_tp, "fp": total_fp,
                "fn": total_fn, "mm": total_mm,
                "total": coverage_total, "pct_tp": coverage_pct_tp,
            },
            "property_breakdown": property_breakdown,
            "operation_breakdown": {
                "initial": initial_op_counts,
                "final": final_op_counts,
            },
            "inverse_ops_added": total_inverse_ops_added,
            "ops_removed_via_dedup": total_ops_removed_via_dedup,
            "tokens": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_tokens,
            },
            "total_time_s": total_time,
            "user_step_limit_reached": max_steps is not None and user_steps >= max_steps,
            "max_steps_config": max_steps,
        }
        if per_heuristic_stats:
            summary_dict["per_heuristic_stats"] = per_heuristic_stats
        recorder.finalize(summary_dict)

        logger.info("Experiment complete: %s", summary.summary_str())
        return summary

    def _check_acceptance(self, metrics: EvaluationMetrics) -> bool:
        """Check if prediction should be accepted based on heuristics."""
        if not self.acceptance_heuristics:
            return True
        for heuristic in self.acceptance_heuristics:
            if heuristic.evaluate(metrics):
                return True
        return False

    def _get_heuristic_details(self, metrics: EvaluationMetrics) -> Dict[str, Any]:
        """Get detailed heuristic check results for recording."""
        if not self.acceptance_heuristics:
            return {"name": "none", "accepted": True, "checks": []}

        # Use first heuristic for recording (main heuristic)
        heuristic = self.acceptance_heuristics[0]
        accepted, checks = heuristic.evaluate_with_details(metrics)
        result = {
            "name": heuristic.name,
            "accepted": accepted,
            "checks": [
                {
                    "metric": c.metric,
                    "value": c.value,
                    "min": c.minimum,
                    "max": c.maximum,
                    "passed": c.passed,
                }
                for c in checks
            ],
        }

        # Include all heuristics' acceptance decisions when multiple are present
        if len(self.acceptance_heuristics) > 1:
            result["all_heuristics"] = {
                h.name: h.evaluate(metrics)
                for h in self.acceptance_heuristics
            }

        return result

    def _get_min_precision_threshold(self) -> float:
        """Get the minimum precision threshold from acceptance heuristics."""
        if not self.acceptance_heuristics:
            return 0.0
        min_prec = 1.0
        for heuristic in self.acceptance_heuristics:
            for metric_name, (lo, _hi) in heuristic.constraints.items():
                if metric_name == "final_state_precision" and lo is not None:
                    min_prec = min(min_prec, lo)
        return min_prec

    @staticmethod
    def _apply_undo_summary(
        metrics: EvaluationMetrics,
        changes: Optional[Dict[str, Any]],
    ) -> None:
        """Apply future-if-accepted simulation operation counts to metrics.

        Mirrors reference ``_apply_undo_summary``: overwrites
        ``final_state_ops_saved`` and ``final_state_ops_diff`` with
        operation-count-based values from ``simulate_future_edits``.
        """
        if not metrics or not changes:
            return
        summary = changes.get("summary") if isinstance(changes, dict) else None
        if not summary:
            return

        metrics.undo_gain_from_final_state = summary.get("final_state_gain", 0)
        metrics.undo_gain_from_dedup = summary.get("dedup_gain", 0)
        metrics.undo_inverse_ops = summary.get("inverse_cost", 0)
        metrics.undo_net_ops_saved = summary.get("net_gain", 0)

        metadata = changes.get("metadata", {})
        future_before_count = metadata.get("future_before_count", 0)
        final_future_count = metadata.get("final_future_count", 0)
        metrics.final_state_ops_saved = future_before_count - final_future_count
        metrics.final_state_ops_diff = final_future_count

    @staticmethod
    def _extract_property_breakdown(eval_result: EvaluationResult) -> Optional[Dict]:
        """Extract property-level stats from an evaluation result."""
        comp = getattr(eval_result, "final_state_comparison", None)
        if comp and hasattr(comp, "property_type_stats"):
            return comp.property_type_stats
        return None

    @staticmethod
    def _extract_matched_pairs(eval_result: EvaluationResult) -> Optional[List[Dict]]:
        """Extract matched pair summaries from an evaluation result (capped at 50)."""
        if not eval_result.matched_pairs:
            return None
        result = []
        for gt_op, pred_op, match_type in eval_result.matched_pairs[:50]:
            result.append({
                "gt": operations_to_symbolic([gt_op])[0] if gt_op else None,
                "pred": operations_to_symbolic([pred_op])[0] if pred_op else None,
                "match_type": match_type,
            })
        return result


__all__ = [
    "Orchestrator",
    "ExperimentSummary",
    "WorkbookStateTracker",
]
