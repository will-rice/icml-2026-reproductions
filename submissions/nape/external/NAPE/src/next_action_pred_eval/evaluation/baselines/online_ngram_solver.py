"""
Online N-gram Solver — learns patterns within a single trajectory.

Uses a hash-indexed suffix table for O(1) suffix matching, plus n-gram
fallback and arithmetic progression detection.

Overrides ``_predict_multi`` because suffix matching natively produces
multi-step predictions (the continuation of a matched pattern).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from next_action_pred_eval.evaluation.solver import DecodingConfig

from .feature_solver import FeatureSolver
from .featurizer import (
    FeaturizedOp,
    featurize_predicted,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Suffix Index — O(1) suffix matching via hash table                        #
# ═══════════════════════════════════════════════════════════════════════════ #


class SuffixIndex:
    """Hash-indexed suffix table for fast longest-match lookup."""

    def __init__(self, max_len: int = 5) -> None:
        self.max_len = max_len
        self._history: List[tuple] = []
        self._index: Dict[int, List[int]] = defaultdict(list)

    def reset(self) -> None:
        self._history.clear()
        self._index.clear()

    def add(self, abstract_op: tuple) -> None:
        self._history.append(abstract_op)
        pos = len(self._history) - 1
        for length in range(1, min(self.max_len + 1, pos + 2)):
            start = pos - length + 1
            if start < 0:
                break
            key = self._hash_slice(start, pos + 1)
            self._index[key].append(start)

    def find_longest_match(
        self, min_len: int = 2
    ) -> Optional[Tuple[int, int]]:
        n = len(self._history)
        for length in range(min(self.max_len, n - 1), min_len - 1, -1):
            query_start = n - length
            key = self._hash_slice(query_start, n)
            if key not in self._index:
                continue
            for start in reversed(self._index[key]):
                if start + length > query_start:
                    continue
                if self._history[start : start + length] == self._history[query_start:n]:
                    return (start, length)
        return None

    def _hash_slice(self, start: int, end: int) -> int:
        return hash(tuple(self._history[start:end]))


# ═══════════════════════════════════════════════════════════════════════════ #
#  OnlineNGramSolver                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #


class OnlineNGramSolver(FeatureSolver):
    """Online n-gram solver: learns patterns within the current trajectory.

    Primary: suffix matching (find repeating pattern, predict continuation).
    Fallback: n-gram frequency + arithmetic progression.

    Overrides ``_predict_multi`` because suffix matching produces multi-step
    predictions natively and can predict across operation type boundaries.

    Args:
        range_mode: ``"absolute"`` or ``"relative"``
        max_ngram_n: Maximum n-gram order (default 5).
        min_match_length: Minimum suffix length (default 2).
        decoding: Decoding configuration for stopping logic.
    """

    def __init__(
        self,
        range_mode: str = "relative",
        max_ngram_n: int = 5,
        min_match_length: int = 2,
        decoding: Optional[DecodingConfig] = None,
    ) -> None:
        super().__init__(range_mode=range_mode, decoding=decoding)
        self.max_ngram_n = max_ngram_n
        self.min_match_length = min_match_length

        self._suffix_idx = SuffixIndex(max_len=max_ngram_n)
        self._ngram_table: Dict[int, Dict[tuple, Counter]] = {}
        self._abstract_count: int = 0

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "online_ngram",
            "range_mode": self.range_mode,
            "max_ngram_n": self.max_ngram_n,
            "min_match_length": self.min_match_length,
            "decoding": self.decoding.model_dump(),
        }

    def _on_reset(self) -> None:
        self._suffix_idx.reset()
        self._ngram_table.clear()
        self._abstract_count = 0

    # ── Abstract key ──────────────────────────────────────────────────────

    def _abstract(self, f: FeaturizedOp) -> tuple:
        if self.range_mode == "absolute":
            return (f.parsed.op_type, f.parsed.height, f.parsed.width, f.parsed.value_type)
        return (f.parsed.op_type, f.movement_class, f.parsed.value_type)

    # ── Incremental index update ──────────────────────────────────────────

    def _ingest_new(self, history: List[FeaturizedOp]) -> None:
        for i in range(self._abstract_count, len(history)):
            ab = self._abstract(history[i])
            self._suffix_idx.add(ab)

            all_abs = self._suffix_idx._history
            pos = len(all_abs) - 1
            for n in range(1, min(self.max_ngram_n + 1, pos + 1)):
                if n not in self._ngram_table:
                    self._ngram_table[n] = defaultdict(Counter)
                ctx = tuple(all_abs[pos - n : pos])
                self._ngram_table[n][ctx][ab] += 1

        self._abstract_count = len(history)

    # ── Multi-step prediction (overrides base) ────────────────────────────

    def _predict_multi(self, history: List[FeaturizedOp]) -> List[str]:
        if len(history) < 2:
            return []

        self._ingest_new(history)

        # Strategy 1: suffix matching
        match = self._suffix_idx.find_longest_match(self.min_match_length)
        if match is not None:
            return self._predict_from_suffix(match, history)

        # Strategy 2: arithmetic progression
        prog = self._predict_from_progression(history)
        if prog:
            return prog

        # Strategy 3: n-gram fallback (uses base class decode loop)
        return super()._predict_multi(history)

    def _predict_from_suffix(
        self,
        match: Tuple[int, int],
        history: List[FeaturizedOp],
    ) -> List[str]:
        """Predict from suffix match continuation.

        Suffix matching can predict across type boundaries because the
        pattern is known to have occurred in the trajectory.
        """
        match_start, match_len = match
        cont_start = match_start + match_len
        predictions: list = []
        virtual_history = list(history)

        for offset in range(self.decoding.max_predictions):
            src_idx = cont_start + offset
            if src_idx >= len(history):
                break

            src = history[src_idx]
            op_type = src.parsed.op_type

            last = virtual_history[-1].parsed
            sr = max(1, last.start_row + src.row_delta)
            sc = max(1, last.start_col + src.col_delta)
            h = max(1, last.height + src.height_delta)
            w = max(1, last.width + src.width_delta)
            sheet = last.sheet if not src.sheet_changed else src.parsed.sheet

            sym = self._reconstruct_op(
                op_type, src.parsed.value_type, sheet, sr, sc, h, w,
                virtual_history,
            )
            predictions.append(sym)

            vfeat = featurize_predicted(sym, last)
            virtual_history.append(vfeat)

        return predictions

    def _predict_from_progression(
        self, history: List[FeaturizedOp]
    ) -> List[str]:
        last = history[-1]
        op_type = last.parsed.op_type

        same = [f for f in history if f.parsed.op_type == op_type]
        if len(same) < 2:
            return []

        recent = same[-min(4, len(same)) :]
        rows = [f.parsed.start_row for f in recent]
        cols = [f.parsed.start_col for f in recent]

        row_diffs = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
        col_diffs = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]

        if not (len(set(row_diffs)) == 1 and len(set(col_diffs)) == 1):
            return []

        dr, dc = row_diffs[0], col_diffs[0]
        if dr == 0 and dc == 0:
            return []

        predictions: list = []
        virtual_history = list(history)

        for k in range(self.decoding.max_predictions):
            last_p = virtual_history[-1].parsed
            sr = max(1, last_p.start_row + dr)
            sc = max(1, last_p.start_col + dc)
            h, w = last_p.height, last_p.width
            sheet = last_p.sheet

            sym = self._reconstruct_op(
                op_type, last.parsed.value_type, sheet, sr, sc, h, w,
                virtual_history,
            )
            predictions.append(sym)

            vfeat = featurize_predicted(sym, last_p)
            virtual_history.append(vfeat)

        return predictions

    # ── Single-step for n-gram fallback ───────────────────────────────────

    def _predict_single(self, history: List[FeaturizedOp]) -> Optional[str]:
        """N-gram fallback: predict one op from inline frequency table."""
        abstracts = [self._abstract(f) for f in history]

        for n in range(min(self.max_ngram_n, len(abstracts)), 0, -1):
            table = self._ngram_table.get(n)
            if table is None:
                continue
            ctx = tuple(abstracts[-n:])
            if ctx in table:
                predicted = table[ctx].most_common(1)[0][0]
                pred_op_type = predicted[0]

                last = history[-1].parsed
                if self.range_mode == "absolute":
                    _, pred_h, pred_w, pred_vtype = predicted
                    sr = max(1, last.start_row + 1)
                    sc = last.start_col
                    return self._reconstruct_op(
                        pred_op_type, pred_vtype, last.sheet,
                        sr, sc, pred_h, pred_w, history,
                    )
                else:
                    _, pred_mv, pred_vtype = predicted
                    from .featurizer import classify_movement
                    canonical = {
                        "same": (0, 0), "right1": (0, 1), "left1": (0, -1),
                        "down1": (1, 0), "up1": (-1, 0), "right_n": (0, 2),
                        "left_n": (0, -2), "down_n": (2, 0), "up_n": (-2, 0),
                        "diagonal": (1, 1), "sheet_change": (0, 0),
                    }
                    dr, dc = canonical.get(pred_mv, (0, 0))
                    sr = max(1, last.start_row + dr)
                    sc = max(1, last.start_col + dc)
                    return self._reconstruct_op(
                        pred_op_type, pred_vtype, last.sheet,
                        sr, sc, last.height, last.width, history,
                    )

        return None
