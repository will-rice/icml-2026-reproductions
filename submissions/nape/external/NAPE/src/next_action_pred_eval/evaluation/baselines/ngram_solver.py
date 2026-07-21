"""
NGram Solver — trained n-gram baseline.

Builds n-gram frequency tables from training data (JSONL) and predicts
next operations by backoff matching.

Absolute mode:
  Abstract key = (op_type, height, width, value_type)
  Range prediction via arithmetic progression on recent same-type ops.

Relative mode:
  Abstract key = (op_type, movement_class, value_type)
  Range prediction via most-common delta from training.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from next_action_pred_eval.evaluation.solver import DecodingConfig

from .feature_solver import FeatureSolver
from .featurizer import (
    FeaturizedOp,
    OperationFeaturizer,
    classify_movement,
    featurize_predicted,
)

logger = logging.getLogger(__name__)


class NGramSolver(FeatureSolver):
    """Trained n-gram solver with absolute / relative range modes.

    Args:
        training_data_path: Path to sequences JSONL.
        range_mode: ``"absolute"`` or ``"relative"``
        max_n: Maximum n-gram order (default 5).
        decoding: Decoding configuration for stopping logic.
    """

    def __init__(
        self,
        training_data_path: str,
        range_mode: str = "relative",
        max_n: int = 5,
        decoding: Optional[DecodingConfig] = None,
    ) -> None:
        super().__init__(range_mode=range_mode, decoding=decoding)
        self.max_n = max_n
        self._training_data_path = training_data_path

        self._ngram_tables: Dict[int, Dict[tuple, Counter]] = {}
        self._delta_dist: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self._shape_dist: Dict[str, Counter] = defaultdict(Counter)
        self._value_dist: Dict[str, Counter] = defaultdict(Counter)

        self._build_tables()

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "ngram",
            "range_mode": self.range_mode,
            "max_n": self.max_n,
            "decoding": self.decoding.model_dump(),
            "training_data": self._training_data_path,
        }

    # ── Table construction ────────────────────────────────────────────────

    def _build_tables(self) -> None:
        path = Path(self._training_data_path)
        if not path.exists():
            logger.warning("Training data not found: %s", path)
            return

        featurizer = OperationFeaturizer()
        seq_count = 0

        for line in path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            ops = record.get("ops", [])
            if len(ops) < 2:
                continue

            featurizer.reset()
            feats = [featurizer.featurize_one(op) for op in ops]
            abstracts = [self._abstract(f) for f in feats]

            for n in range(1, self.max_n + 1):
                if n not in self._ngram_tables:
                    self._ngram_tables[n] = defaultdict(Counter)
                table = self._ngram_tables[n]
                for i in range(n, len(abstracts)):
                    context = tuple(abstracts[i - n : i])
                    table[context][abstracts[i]] += 1

            for i in range(1, len(feats)):
                prev_f, curr_f = feats[i - 1], feats[i]
                self._delta_dist[(prev_f.parsed.op_type, curr_f.parsed.op_type)][
                    (curr_f.row_delta, curr_f.col_delta)
                ] += 1
                self._shape_dist[curr_f.parsed.op_type][
                    (curr_f.parsed.height, curr_f.parsed.width)
                ] += 1
                self._value_dist[curr_f.parsed.op_type][
                    curr_f.parsed.raw_value
                ] += 1

            seq_count += 1

        n_entries = sum(
            sum(c.total() for c in t.values())
            for t in self._ngram_tables.values()
        )
        logger.info(
            "Built n-gram tables from %d sequences (%d total entries)",
            seq_count, n_entries,
        )

    # ── Abstract key ──────────────────────────────────────────────────────

    def _abstract(self, f: FeaturizedOp) -> tuple:
        if self.range_mode == "absolute":
            return (f.parsed.op_type, f.parsed.height, f.parsed.width, f.parsed.value_type)
        else:
            return (f.parsed.op_type, f.movement_class, f.parsed.value_type)

    # ── Single-step prediction ────────────────────────────────────────────

    def _predict_single(self, history: List[FeaturizedOp]) -> Optional[str]:
        if not history or not self._ngram_tables:
            return None

        abstracts = [self._abstract(f) for f in history]
        predicted_abstract = self._ngram_lookup(abstracts)
        if predicted_abstract is None:
            return None

        pred_op_type = predicted_abstract[0]
        sr, sc, h, w = self._predict_range(
            pred_op_type, predicted_abstract, history
        )
        sheet = history[-1].parsed.sheet

        pred_vtype = (
            predicted_abstract[3]
            if self.range_mode == "absolute"
            else predicted_abstract[2]
        )
        return self._reconstruct_op(
            pred_op_type, pred_vtype, sheet, sr, sc, h, w, history
        )

    def _ngram_lookup(self, abstracts: List[tuple]) -> Optional[tuple]:
        for n in range(min(self.max_n, len(abstracts)), 0, -1):
            table = self._ngram_tables.get(n)
            if table is None:
                continue
            context = tuple(abstracts[-n:])
            if context in table:
                return table[context].most_common(1)[0][0]
        return None

    def _predict_range(
        self,
        op_type: str,
        predicted_abstract: tuple,
        history: List[FeaturizedOp],
    ) -> Tuple[int, int, int, int]:
        last = history[-1].parsed

        if self.range_mode == "absolute":
            _, pred_h, pred_w, _ = predicted_abstract
            pos = self._detect_progression(op_type, history)
            if pos is not None:
                return pos[0], pos[1], pred_h, pred_w
            dr, dc = self._most_common_delta(last.op_type, op_type)
            return max(1, last.start_row + dr), max(1, last.start_col + dc), pred_h, pred_w
        else:
            _, pred_mv, _ = predicted_abstract
            dr, dc = self._movement_to_delta(pred_mv, last.op_type, op_type)
            h, w = last.height, last.width
            if op_type in self._shape_dist:
                h, w = self._shape_dist[op_type].most_common(1)[0][0]
            return max(1, last.start_row + dr), max(1, last.start_col + dc), h, w

    def _detect_progression(
        self, op_type: str, history: List[FeaturizedOp], min_run: int = 2
    ) -> Optional[Tuple[int, int]]:
        same = [f for f in history if f.parsed.op_type == op_type]
        if len(same) < min_run:
            return None
        recent = same[-min(4, len(same)) :]
        rows = [f.parsed.start_row for f in recent]
        cols = [f.parsed.start_col for f in recent]
        if len(recent) >= 2:
            row_diffs = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
            col_diffs = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
            if len(set(row_diffs)) == 1 and len(set(col_diffs)) == 1:
                return rows[-1] + row_diffs[0], cols[-1] + col_diffs[0]
        return None

    def _most_common_delta(
        self, prev_op: str, next_op: str
    ) -> Tuple[int, int]:
        key = (prev_op, next_op)
        if key in self._delta_dist and self._delta_dist[key]:
            return self._delta_dist[key].most_common(1)[0][0]
        return 0, 0

    def _movement_to_delta(
        self, movement_class: str, prev_op: str, next_op: str
    ) -> Tuple[int, int]:
        dr, dc = self._most_common_delta(prev_op, next_op)
        actual_mv, _ = classify_movement(dr, dc, False)
        if actual_mv == movement_class:
            return dr, dc

        canonical = {
            "same": (0, 0), "right1": (0, 1), "left1": (0, -1),
            "down1": (1, 0), "up1": (-1, 0), "right_n": (0, 2),
            "left_n": (0, -2), "down_n": (2, 0), "up_n": (-2, 0),
            "diagonal": (1, 1), "sheet_change": (0, 0),
        }
        return canonical.get(movement_class, (0, 0))
