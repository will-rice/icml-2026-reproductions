"""
Output Layout Module

Manages the 2-level output directory hierarchy and summary file writing.

Directory structure::

    experiment_dir/
      run_config.yaml
      batch_summary.csv
      experiment_summary.json
      run.log
      failed_runs.jsonl
      {file_label}/
        experiment_summary.json
        predictions.jsonl
        timeline.jsonl
        final_trajectory.jsonl
        target_state.json
        divergence_log.jsonl
        trajectory.log
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# CSV column order for batch_summary.csv
CSV_COLUMNS = [
    "file_label",
    "config_variant",
    "status",
    "net_operations_saved",
    "predictions_attempted",
    "predictions_accepted",
    "empty_predictions",
    "errored_empty_predictions",
    "initial_sequence_length",
    "final_sequence_length",
    "user_steps_taken",
    "uas_pct",
    "total_formatting_ops",
    "ufas",
    "ufas_pct",
    "acceptance_rate",
    "avg_precision",
    "coverage_pct_tp",
    "ops_saved_per_prediction",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "total_time",
    "inverse_ops_added",
    "user_step_limit_reached",
    "error_message",
]


@dataclass
class TrajectoryResult:
    """
    Result for one trajectory run — maps to a single CSV row.
    """

    file_label: str
    config_variant: str
    status: str  # "success" | "error" | "skipped"
    net_operations_saved: int = 0
    predictions_attempted: int = 0
    predictions_accepted: int = 0
    empty_predictions: int = 0
    errored_empty_predictions: int = 0
    initial_sequence_length: int = 0
    final_sequence_length: int = 0
    user_steps_taken: int = 0
    uas_pct: float = 0.0
    total_formatting_ops: int = 0
    ufas: int = 0
    ufas_pct: float = 0.0
    acceptance_rate: float = 0.0
    avg_precision: float = 0.0
    coverage_pct_tp: float = 0.0
    ops_saved_per_prediction: float = 0.0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_time: float = 0.0
    inverse_ops_added: int = 0
    user_step_limit_reached: bool = False
    error_message: str = ""

    # Per-heuristic acceptance stats (offline multi-heuristic evaluation)
    per_heuristic_stats: Optional[Dict[str, Dict[str, Any]]] = None

    def to_csv_row(self) -> Dict[str, Any]:
        """Return dict keyed by CSV_COLUMNS plus per-heuristic columns."""
        d = asdict(self)
        d["user_step_limit_reached"] = str(d["user_step_limit_reached"])
        row = {col: d.get(col, "") for col in CSV_COLUMNS}
        # Add per-heuristic acceptance rate columns
        if self.per_heuristic_stats:
            for h_name, h_stats in self.per_heuristic_stats.items():
                row[f"accept_rate_{h_name}"] = h_stats.get("acceptance_rate", 0.0)
                row[f"ops_saved_{h_name}"] = h_stats.get("total_ops_saved", 0)
                row[f"uas_{h_name}"] = h_stats.get("uas_pct", 0.0)
        return row


class OutputLayout:
    """
    Manages the experiment output directory structure and summary files.

    Usage::

        layout = OutputLayout(base_dir="outputs/my_experiment")

        # Get path for a specific trajectory
        run_dir = layout.get_run_dir("0000afae")

        # Check if already completed (for auto-resume)
        if layout.is_completed("0000afae"):
            continue

        # After each trajectory completes, checkpoint to CSV
        layout.append_csv_row(result)

        # After all runs, write final summaries
        layout.write_csv(results)
        layout.write_experiment_summary(results, config_dict)
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.base_dir / "batch_summary.csv"
        self._summary_path = self.base_dir / "experiment_summary.json"
        self._failed_path = self.base_dir / "failed_runs.jsonl"

    def get_run_dir(self, file_label: str) -> Path:
        """Return and create ``experiment_dir/{file_label}/``."""
        run_dir = self.base_dir / file_label
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def is_completed(self, file_label: str) -> bool:
        """Check if experiment_summary.json exists for this trajectory."""
        return (self.base_dir / file_label / "experiment_summary.json").exists()

    def _get_csv_columns(self, results: List[TrajectoryResult]) -> List[str]:
        """Get CSV columns including dynamic per-heuristic columns."""
        columns = list(CSV_COLUMNS)
        # Collect per-heuristic column names from results
        seen_heuristics: List[str] = []
        for r in results:
            if r.per_heuristic_stats:
                for h_name in r.per_heuristic_stats:
                    if h_name not in seen_heuristics:
                        seen_heuristics.append(h_name)
        for h_name in seen_heuristics:
            columns.append(f"accept_rate_{h_name}")
            columns.append(f"ops_saved_{h_name}")
            columns.append(f"uas_{h_name}")
        return columns

    def write_csv(self, results: List[TrajectoryResult]) -> Path:
        """Write (or overwrite) batch_summary.csv with all results."""
        columns = self._get_csv_columns(results)
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_csv_row())
        logger.info("Wrote %d rows to %s", len(results), self._csv_path)
        return self._csv_path

    def append_csv_row(self, result: TrajectoryResult) -> None:
        """Append a single row to batch_summary.csv (incremental checkpoint)."""
        file_exists = self._csv_path.exists()
        # For append, use base columns + any per-heuristic columns from this result
        columns = self._get_csv_columns([result])
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(result.to_csv_row())

    def write_experiment_summary(
        self,
        results: List[TrajectoryResult],
        config_dict: Dict[str, Any],
        wall_time: float = 0.0,
    ) -> Path:
        """Write aggregate experiment_summary.json."""
        successful = [r for r in results if r.status == "success"]
        failed = [r for r in results if r.status == "error"]

        # Compute aggregate metrics
        total_attempted = sum(r.predictions_attempted for r in successful)
        total_accepted = sum(r.predictions_accepted for r in successful)
        total_saved = sum(r.net_operations_saved for r in successful)
        total_empty = sum(r.empty_predictions for r in successful)
        total_errored_empty = sum(r.errored_empty_predictions for r in successful)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "wall_time_seconds": wall_time,
            "config": config_dict,
            "counts": {
                "total": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "skipped": len(results) - len(successful) - len(failed),
            },
            "aggregate_metrics": {
                "total_predictions_attempted": total_attempted,
                "total_predictions_accepted": total_accepted,
                "total_net_ops_saved": total_saved,
                "total_empty_predictions": total_empty,
                "total_errored_empty_predictions": total_errored_empty,
                "overall_acceptance_rate": (
                    total_accepted / total_attempted if total_attempted > 0 else 0.0
                ),
                "mean_uas_pct": (
                    sum(r.uas_pct for r in successful) / len(successful)
                    if successful
                    else 0.0
                ),
                "mean_ufas_pct": (
                    sum(r.ufas_pct for r in successful) / len(successful)
                    if successful
                    else 0.0
                ),
                "mean_avg_precision": (
                    sum(r.avg_precision for r in successful) / len(successful)
                    if successful
                    else 0.0
                ),
                "total_tokens": sum(r.total_tokens for r in successful),
                "total_input_tokens": sum(r.input_tokens for r in successful),
                "total_output_tokens": sum(r.output_tokens for r in successful),
                "total_time": sum(r.total_time for r in successful),
            },
        }

        # Aggregate per-heuristic stats across trajectories
        per_h_agg: Dict[str, Dict[str, Any]] = {}
        for r in successful:
            if r.per_heuristic_stats:
                for h_name, h_stats in r.per_heuristic_stats.items():
                    if h_name not in per_h_agg:
                        per_h_agg[h_name] = {"accepted": 0, "rejected": 0, "total_ops_saved": 0}
                    per_h_agg[h_name]["accepted"] += h_stats.get("accepted", 0)
                    per_h_agg[h_name]["rejected"] += h_stats.get("rejected", 0)
                    per_h_agg[h_name]["total_ops_saved"] += h_stats.get("total_ops_saved", 0)
        if per_h_agg:
            # Compute total initial_length across successful runs for aggregate UAS
            total_initial_length = sum(r.initial_sequence_length for r in successful)
            for h_name, agg in per_h_agg.items():
                h_total = agg["accepted"] + agg["rejected"]
                agg["total_predictions"] = h_total
                agg["acceptance_rate"] = agg["accepted"] / h_total if h_total > 0 else 0.0
                agg["uas_pct"] = (
                    agg["total_ops_saved"] / total_initial_length
                    if total_initial_length > 0 else 0.0
                )
            summary["per_heuristic_stats"] = per_h_agg

        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Wrote experiment summary to %s", self._summary_path)
        return self._summary_path

    def append_failed_run(
        self,
        file_label: str,
        config_variant: str,
        error: str,
        traceback_str: str = "",
    ) -> None:
        """Append a line to failed_runs.jsonl."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "file_label": file_label,
            "config_variant": config_variant,
            "error": error,
            "traceback": traceback_str,
        }
        with open(self._failed_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def save_run_config(self, config_dict: Dict[str, Any]) -> Path:
        """Save run_config.yaml in experiment_dir."""
        import yaml

        config_path = self.base_dir / "run_config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        return config_path

    def load_completed_result(self, file_label: str, config_variant: str) -> Optional[TrajectoryResult]:
        """Load a TrajectoryResult from a previously completed run's experiment_summary.json."""
        summary_path = self.base_dir / file_label / "experiment_summary.json"
        if not summary_path.exists():
            return None
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TrajectoryResult(
                file_label=data.get("file_label", file_label),
                config_variant=config_variant,
                status=data.get("status", "success"),
                net_operations_saved=data.get("net_operations_saved", 0),
                predictions_attempted=data.get("predictions_attempted", 0),
                predictions_accepted=data.get("predictions_accepted", 0),
                empty_predictions=data.get("empty_predictions", 0),
                errored_empty_predictions=data.get("errored_empty_predictions", 0),
                initial_sequence_length=data.get("initial_sequence_length", 0),
                final_sequence_length=data.get("final_sequence_length", 0),
                user_steps_taken=data.get("user_steps_taken", 0),
                uas_pct=data.get("uas_pct", 0.0),
                total_formatting_ops=data.get("total_formatting_ops", 0),
                ufas=data.get("ufas", 0),
                ufas_pct=data.get("ufas_pct", 0.0),
                acceptance_rate=data.get("acceptance_rate", 0.0),
                avg_precision=data.get("avg_precision", 0.0),
                coverage_pct_tp=data.get("coverage", {}).get("pct_tp", 0.0),
                ops_saved_per_prediction=data.get("ops_saved_per_prediction", 0.0),
                total_tokens=data.get("tokens", {}).get("total", 0),
                input_tokens=data.get("tokens", {}).get("input", 0),
                output_tokens=data.get("tokens", {}).get("output", 0),
                total_time=data.get("total_time_s", 0.0),
                inverse_ops_added=data.get("inverse_ops_added", 0),
                user_step_limit_reached=data.get("user_step_limit_reached", False),
                per_heuristic_stats=data.get("per_heuristic_stats"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load completed result for %s: %s", file_label, e)
            return None


__all__ = [
    "CSV_COLUMNS",
    "TrajectoryResult",
    "OutputLayout",
]
