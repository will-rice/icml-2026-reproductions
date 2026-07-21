"""
Prediction Folder Saver Module

Writes per-prediction artifact folders to disk for detailed inspection.
Each non-empty prediction gets its own numbered directory containing
prompt, response, metrics, and other artifacts.

Directory structure::

    predictions/
      prediction_001/
        prompt.txt              # Raw LLM prompt (system + user)
        response.txt            # Raw LLM response text
        response_meta.json      # Model, tokens, timing, retries
        predicted_ops.txt       # Parsed operations (symbolic, one per line)
        gt_segment.txt          # Ground truth segment (symbolic, one per line)
        history_context.txt     # History ops used as context (one per line)
        evaluation.json         # Full eval metrics + property breakdown + matched pairs
        acceptance.json         # Heuristic decision + per-check pass/fail
        future_edits.json       # Online mode: future_if_accepted + GT length changes
      prediction_002/
        ...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PredictionFolderSaver:
    """Writes per-prediction artifact folders to disk.

    Created by ExperimentRecorder when ``save_prediction_folders`` is enabled.
    Each call to :meth:`save_prediction` creates a numbered directory with all
    artifacts for that prediction.
    """

    def __init__(self, base_dir: Path):
        """
        Args:
            base_dir: The trajectory output directory
                      (e.g. ``experiment_dir/{file_label}/``).
        """
        self.predictions_dir = Path(base_dir) / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_prediction(
        self,
        prediction_index: int,
        *,
        # Prompt / response data (from PredictionResult.metadata)
        prompt_messages: Optional[List[Dict[str, str]]] = None,
        raw_response: Optional[str] = None,
        response_metadata: Optional[Dict[str, Any]] = None,
        # Prediction data
        predicted_ops_symbolic: Optional[List[str]] = None,
        # Ground truth
        gt_segment_symbolic: Optional[List[str]] = None,
        # Context
        history_context: Optional[List[str]] = None,
        # Evaluation
        eval_metrics: Optional[Dict[str, Any]] = None,
        property_breakdown: Optional[Dict[str, Any]] = None,
        matched_pairs_summary: Optional[List[Dict[str, Any]]] = None,
        # Acceptance
        accepted: Optional[bool] = None,
        heuristic_details: Optional[Dict[str, Any]] = None,
        # Online mode
        future_if_accepted: Optional[Dict[str, Any]] = None,
        gt_len_before: Optional[int] = None,
        gt_len_after: Optional[int] = None,
        # Timing and tokens
        generation_time_s: float = 0.0,
        tokens: Optional[Dict[str, int]] = None,
    ) -> Path:
        """Write all artifacts for a single prediction to a numbered folder.

        Returns:
            Path to the created prediction folder.
        """
        folder = self.predictions_dir / f"prediction_{prediction_index:03d}"
        folder.mkdir(parents=True, exist_ok=True)

        # 1. Prompt
        if prompt_messages is not None:
            self._write_prompt(folder / "prompt.txt", prompt_messages)

        # 2. Raw response
        if raw_response is not None:
            self._write_text(folder / "response.txt", raw_response)

        # 3. Response metadata
        meta: Dict[str, Any] = {}
        if response_metadata:
            meta.update(response_metadata)
        meta["generation_time_s"] = generation_time_s
        if tokens:
            meta["tokens"] = tokens
        self._write_json(folder / "response_meta.json", meta)

        # 4. Predicted operations
        if predicted_ops_symbolic is not None:
            self._write_lines(folder / "predicted_ops.txt", predicted_ops_symbolic)

        # 5. Ground truth segment
        if gt_segment_symbolic is not None:
            self._write_lines(folder / "gt_segment.txt", gt_segment_symbolic)

        # 6. History context
        if history_context is not None:
            self._write_lines(folder / "history_context.txt", history_context)

        # 7. Evaluation
        eval_data: Dict[str, Any] = {}
        if eval_metrics:
            eval_data["metrics"] = eval_metrics
        if property_breakdown:
            eval_data["property_breakdown"] = property_breakdown
        if matched_pairs_summary:
            eval_data["matched_pairs"] = matched_pairs_summary
        if eval_data:
            self._write_json(folder / "evaluation.json", eval_data)

        # 8. Acceptance
        if accepted is not None:
            acceptance_data: Dict[str, Any] = {"accepted": accepted}
            if heuristic_details:
                acceptance_data["heuristic"] = heuristic_details
            self._write_json(folder / "acceptance.json", acceptance_data)

        # 9. Future edits (online mode)
        if future_if_accepted is not None or gt_len_before is not None:
            future_data: Dict[str, Any] = {}
            if future_if_accepted:
                future_data["future_if_accepted"] = future_if_accepted
            if gt_len_before is not None:
                future_data["gt_len_before"] = gt_len_before
            if gt_len_after is not None:
                future_data["gt_len_after"] = gt_len_after
            self._write_json(folder / "future_edits.json", future_data)

        logger.debug("Saved prediction folder: %s", folder)
        return folder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _write_lines(path: Path, lines: List[str]) -> None:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _write_prompt(path: Path, messages: List[Dict[str, str]]) -> None:
        """Format prompt messages into a readable text file."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            parts.append(f"[{role}]\n{content}")
        path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


__all__ = ["PredictionFolderSaver"]
