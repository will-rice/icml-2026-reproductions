"""
Experiment Recorder Module

Central class for recording all per-trajectory experiment data.
The orchestrator calls recorder methods to write timeline events,
prediction details, final trajectory attribution, and summaries.

Output files per trajectory::

    {file_label}/
      experiment_summary.json     # high-level metrics
      predictions.jsonl           # one line per prediction attempt
      timeline.jsonl              # one line per event (user steps + predictions)
      final_trajectory.jsonl      # final GT with source attribution
      target_state.json           # final target workbook state
      divergence_log.jsonl        # divergences (if any)
      trajectory.log              # detailed per-trajectory log
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from next_action_pred_eval.core.symbolic import operations_to_symbolic
from next_action_pred_eval.evaluation.prediction_saver import PredictionFolderSaver

logger = logging.getLogger(__name__)

# Number of history/future ops to include in timeline snapshots
TIMELINE_CONTEXT_SIZE = 10


class ExperimentRecorder:
    """
    Records all experiment data for a single trajectory.

    Created by the orchestrator at the start of each trajectory evaluation.
    Methods are called during the evaluation loop to stream events to disk.
    """

    def __init__(self, output_dir: Path, file_label: str, buffered_writes: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_label = file_label
        self._buffered_writes = buffered_writes

        # File handles (opened lazily)
        self._timeline_path = self.output_dir / "timeline.jsonl"
        self._predictions_path = self.output_dir / "predictions.jsonl"
        self._divergence_path = self.output_dir / "divergence_log.jsonl"

        # In-memory buffers for buffered_writes mode
        self._buffers: Dict[Path, List[str]] = {}

        # Per-trajectory logger — file only, not propagated to terminal
        self._traj_logger = logging.getLogger(f"trajectory.{file_label}")
        self._traj_logger.propagate = False  # don't show in terminal
        log_path = self.output_dir / "trajectory.log"
        if not self._traj_logger.handlers:
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self._traj_logger.addHandler(handler)
            self._traj_logger.setLevel(logging.DEBUG)

        # Counters for finalization
        self._prediction_count = 0
        self._empty_prediction_count = 0
        self._errored_empty_count = 0
        self._divergence_count = 0

        # Accumulate precision values for macro average
        self._precision_values: List[float] = []

        # Operation counts for breakdown
        self._ops_predicted_by_type: Dict[str, int] = {}
        self._ops_accepted_by_type: Dict[str, int] = {}
        self._ops_rejected_by_type: Dict[str, int] = {}

        # Error tracking
        self._error_counts: Dict[str, int] = {}
        self._error_details: List[Dict[str, Any]] = []
        self._MAX_ERROR_DETAILS = 50

        # Partial parse tracking
        self._partial_parse_predictions = 0
        self._total_parse_failures = 0

        # Per-prediction folder saver (None until enabled)
        self._prediction_saver: Optional[PredictionFolderSaver] = None

    def _append_jsonl(self, path: Path, data: Dict[str, Any]) -> None:
        """Append a single JSON line to a file (or in-memory buffer)."""
        line = json.dumps(data, default=str) + "\n"
        if self._buffered_writes:
            self._buffers.setdefault(path, []).append(line)
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

    def _flush_buffers(self) -> None:
        """Write all buffered JSONL lines to disk at once."""
        for path, lines in self._buffers.items():
            with open(path, "a", encoding="utf-8") as f:
                f.writelines(lines)
        self._buffers.clear()

    def _count_ops_by_type(self, symbolic_ops: List[str]) -> Dict[str, int]:
        """Count operations by type from symbolic strings."""
        counts: Dict[str, int] = {}
        for op_str in symbolic_ops:
            op_type = op_str.split("|")[0].strip() if "|" in op_str else "UNKNOWN"
            counts[op_type] = counts.get(op_type, 0) + 1
        return counts

    def _merge_counts(self, target: Dict[str, int], source: Dict[str, int]) -> None:
        """Merge source counts into target."""
        for k, v in source.items():
            target[k] = target.get(k, 0) + v

    def enable_prediction_folders(self) -> None:
        """Enable per-prediction folder saving for detailed artifact inspection."""
        self._prediction_saver = PredictionFolderSaver(self.output_dir)

    # ----------------------------------------------------------------
    # Event recording methods
    # ----------------------------------------------------------------

    def record_user_step(
        self,
        t: int,
        user_step: int,
        op_symbolic: str,
        history_len: int,
        future_len: int,
    ) -> None:
        """Record a user step event in the timeline."""
        event = {
            "event": "user_step",
            "t": t,
            "user_step": user_step,
            "op": op_symbolic,
            "history_len": history_len,
            "future_len": future_len,
        }
        self._append_jsonl(self._timeline_path, event)
        self._traj_logger.debug(
            "User step t=%d: %s (history=%d, future=%d)",
            t, op_symbolic[:80], history_len, future_len,
        )

    def record_prediction(
        self,
        prediction_index: int,
        t: int,
        user_step: int,
        pred_ops_symbolic: List[str],
        gt_segment_symbolic: List[str],
        eval_metrics: Dict[str, Any],
        accepted: bool,
        heuristic_details: Dict[str, Any],
        tokens: Dict[str, int],
        generation_time_s: float,
        history_tail: List[str],
        future_head: List[str],
        future_if_accepted: Optional[Dict[str, Any]] = None,
        gt_len_before: Optional[int] = None,
        gt_len_after: Optional[int] = None,
        # Additional data for prediction folder saving
        prompt_messages: Optional[List[Dict[str, str]]] = None,
        raw_response: Optional[str] = None,
        response_metadata: Optional[Dict[str, Any]] = None,
        property_breakdown: Optional[Dict[str, Any]] = None,
        matched_pairs_summary: Optional[List[Dict[str, Any]]] = None,
        full_history_context: Optional[List[str]] = None,
    ) -> None:
        """Record a prediction event in both predictions.jsonl and timeline.jsonl."""
        self._prediction_count += 1

        # Track precision for macro average
        precision = eval_metrics.get("final_state_precision", 0.0)
        self._precision_values.append(precision)

        # Track operation counts by type
        pred_counts = self._count_ops_by_type(pred_ops_symbolic)
        self._merge_counts(self._ops_predicted_by_type, pred_counts)
        if accepted:
            self._merge_counts(self._ops_accepted_by_type, pred_counts)
        else:
            self._merge_counts(self._ops_rejected_by_type, pred_counts)

        # Write to predictions.jsonl
        prediction_entry = {
            "prediction_index": prediction_index,
            "step_t": t,
            "user_step": user_step,
            "generation_time_s": generation_time_s,
            "tokens": tokens,
            "predicted_ops": pred_ops_symbolic,
            "predicted_count": len(pred_ops_symbolic),
            "gt_segment": gt_segment_symbolic,
            "eval_metrics": eval_metrics,
            "accepted": accepted,
            "heuristic": heuristic_details,
        }
        if future_if_accepted is not None:
            prediction_entry["future_if_accepted"] = future_if_accepted
        self._append_jsonl(self._predictions_path, prediction_entry)

        # Write to timeline.jsonl
        timeline_event = {
            "event": "prediction",
            "t": t,
            "user_step": user_step,
            "prediction_index": prediction_index,
            "history_tail": history_tail[-TIMELINE_CONTEXT_SIZE:],
            "future_head": future_head[:TIMELINE_CONTEXT_SIZE],
            "predicted_ops": pred_ops_symbolic,
            "accepted": accepted,
            "tp": eval_metrics.get("final_state_tp", 0),
            "fp": eval_metrics.get("final_state_fp", 0),
            "fn": eval_metrics.get("final_state_fn", 0),
            "precision": precision,
            "mm": eval_metrics.get("final_state_mm", 0),
            "ops_saved": eval_metrics.get("final_state_ops_saved", 0),
            "tokens": tokens.get("total", 0),
            "time_s": generation_time_s,
        }
        if accepted and gt_len_before is not None:
            timeline_event["gt_len_before"] = gt_len_before
            timeline_event["gt_len_after"] = gt_len_after
            timeline_event["t_after"] = t + len(pred_ops_symbolic)
        self._append_jsonl(self._timeline_path, timeline_event)

        # Log
        status = "ACCEPTED" if accepted else "REJECTED"
        self._traj_logger.info(
            "Prediction #%d at t=%d: %s | %d ops | precision=%.3f | ops_saved=%d | tokens=%d",
            prediction_index, t, status, len(pred_ops_symbolic),
            precision, eval_metrics.get("final_state_ops_saved", 0),
            tokens.get("total", 0),
        )

        # Save per-prediction folder artifacts (if enabled)
        if self._prediction_saver is not None:
            try:
                self._prediction_saver.save_prediction(
                    prediction_index=prediction_index,
                    prompt_messages=prompt_messages,
                    raw_response=raw_response,
                    response_metadata=response_metadata,
                    predicted_ops_symbolic=pred_ops_symbolic,
                    gt_segment_symbolic=gt_segment_symbolic,
                    history_context=full_history_context or history_tail,
                    eval_metrics=eval_metrics,
                    property_breakdown=property_breakdown,
                    matched_pairs_summary=matched_pairs_summary,
                    accepted=accepted,
                    heuristic_details=heuristic_details,
                    future_if_accepted=future_if_accepted,
                    gt_len_before=gt_len_before,
                    gt_len_after=gt_len_after,
                    generation_time_s=generation_time_s,
                    tokens=tokens,
                )
            except Exception as e:
                self._traj_logger.warning(
                    "Failed to save prediction folder for #%d: %s",
                    prediction_index, e,
                )

    def record_empty_prediction(
        self,
        t: int,
        user_step: int,
        tokens: Dict[str, int],
        generation_time_s: float,
        error_reason: Optional[str] = None,
    ) -> None:
        """Record an empty or errored prediction in the timeline.

        If *error_reason* is set the prediction is counted as an
        **errored** prediction (separate from genuinely empty ones).
        """
        if error_reason:
            self._errored_empty_count += 1
            self._error_counts["errored_empty_prediction"] = (
                self._error_counts.get("errored_empty_prediction", 0) + 1
            )
        else:
            self._empty_prediction_count += 1

        event = {
            "event": "errored_prediction" if error_reason else "empty_prediction",
            "t": t,
            "user_step": user_step,
            "prediction_index": None,
            "tokens": tokens.get("total", 0),
            "time_s": generation_time_s,
        }
        if error_reason:
            event["error_reason"] = error_reason
        self._append_jsonl(self._timeline_path, event)
        if error_reason:
            self._traj_logger.warning(
                "Errored prediction at t=%d (tokens=%d): %s",
                t, tokens.get("total", 0), error_reason,
            )
        else:
            self._traj_logger.debug(
                "Empty prediction at t=%d (tokens=%d)", t, tokens.get("total", 0)
            )

    def record_divergence(
        self,
        prediction_index: int,
        t: int,
        description: str,
        action: str = "rejected",
    ) -> None:
        """Record a divergence event."""
        self._divergence_count += 1

        entry = {
            "prediction_index": prediction_index,
            "step_t": t,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "action": action,
        }
        self._append_jsonl(self._divergence_path, entry)

        timeline_event = {
            "event": "divergence",
            "t": t,
            "prediction_index": prediction_index,
            "description": description,
            "action": action,
        }
        self._append_jsonl(self._timeline_path, timeline_event)

        self._traj_logger.warning(
            "DIVERGENCE at t=%d prediction #%d: %s -> %s",
            t, prediction_index, description, action,
        )

    def record_partial_parse(
        self,
        t: int,
        n_failed: int,
        n_total: int,
        failed_details: List[Dict[str, str]],
    ) -> None:
        """Record a prediction where some operations failed to parse.

        Args:
            t: Step pointer when the parse failure occurred.
            n_failed: Number of operations that failed to parse.
            n_total: Total number of raw operations (valid + failed).
            failed_details: List of dicts with 'symbolic' and 'reason' keys.
        """
        self._partial_parse_predictions += 1
        self._total_parse_failures += n_failed

        # Also record as a standard error for the error summary
        self._error_counts["parse_error"] = (
            self._error_counts.get("parse_error", 0) + 1
        )
        if len(self._error_details) < self._MAX_ERROR_DETAILS:
            self._error_details.append({
                "type": "parse_error",
                "t": t,
                "description": f"{n_failed}/{n_total} predicted ops failed to parse",
                "failed_ops": failed_details,
            })
        self._traj_logger.warning(
            "Partial parse at t=%d: %d/%d ops failed to parse",
            t, n_failed, n_total,
        )
        for detail in failed_details:
            self._traj_logger.debug(
                "  Failed op: '%s' reason: %s",
                detail.get("symbolic", "")[:80],
                detail.get("reason", "unknown"),
            )

    def record_error(
        self,
        error_type: str,
        t: int,
        description: str,
    ) -> None:
        """Record an error for the error summary.

        Args:
            error_type: Category (e.g., "parse_error", "prediction_failure").
            t: Step pointer when error occurred.
            description: Short description of the error.
        """
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        if len(self._error_details) < self._MAX_ERROR_DETAILS:
            self._error_details.append({
                "type": error_type,
                "t": t,
                "description": description,
            })
        self._traj_logger.debug("Error [%s] at t=%d: %s", error_type, t, description)

    def record_final_trajectory(
        self,
        attribution_list: List[Dict[str, Any]],
    ) -> None:
        """
        Write final_trajectory.jsonl with source attribution.

        Each entry has: index, op, source, and optional prediction_index/user_step/step_t.
        """
        path = self.output_dir / "final_trajectory.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in attribution_list:
                f.write(json.dumps(entry, default=str) + "\n")
        self._traj_logger.info(
            "Wrote final trajectory: %d operations", len(attribution_list)
        )

    def save_target_state(self, state_dict: Dict[str, Any]) -> None:
        """Save the final target workbook state."""
        path = self.output_dir / "target_state.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, default=str)

    def finalize(self, summary_dict: Dict[str, Any]) -> Path:
        """
        Write experiment_summary.json for this trajectory.

        The caller provides the summary dict; this method enriches it
        with recorder-tracked data (precision values, op counts, etc.).
        """
        # Enrich with recorder-tracked metrics
        summary_dict["empty_predictions"] = self._empty_prediction_count
        summary_dict["errored_empty_predictions"] = self._errored_empty_count
        summary_dict["divergences"] = self._divergence_count

        # Macro average precision
        if self._precision_values:
            summary_dict["avg_precision"] = (
                sum(self._precision_values) / len(self._precision_values)
            )
        else:
            summary_dict["avg_precision"] = 0.0

        # Ops saved per prediction
        attempted = summary_dict.get("predictions_attempted", 0)
        saved = summary_dict.get("net_operations_saved", 0)
        summary_dict["ops_saved_per_prediction"] = (
            saved / attempted if attempted > 0 else 0.0
        )

        # Operation breakdown
        summary_dict["operation_breakdown"] = {
            "predicted": self._ops_predicted_by_type,
            "accepted": self._ops_accepted_by_type,
            "rejected": self._ops_rejected_by_type,
        }

        # Partial parse tracking
        summary_dict["partial_parse"] = {
            "predictions_with_parse_failures": self._partial_parse_predictions,
            "total_ops_failed_to_parse": self._total_parse_failures,
        }

        # Error report
        total_errors = sum(self._error_counts.values())
        summary_dict["errors"] = {
            "total": total_errors,
            "counts": self._error_counts,
            "sample_details": self._error_details,
        }

        # Flush any buffered JSONL writes before writing the summary
        if self._buffered_writes:
            self._flush_buffers()

        path = self.output_dir / "experiment_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2, default=str)

        self._traj_logger.info("Finalized experiment summary: %s", path)
        return path

    @property
    def prediction_count(self) -> int:
        return self._prediction_count

    @property
    def avg_precision(self) -> float:
        if not self._precision_values:
            return 0.0
        return sum(self._precision_values) / len(self._precision_values)


__all__ = [
    "ExperimentRecorder",
    "TIMELINE_CONTEXT_SIZE",
]
