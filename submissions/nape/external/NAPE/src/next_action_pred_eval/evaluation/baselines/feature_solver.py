"""Base class for feature-based (non-LLM) baseline solvers.

Provides shared incremental featurization, history management,
centralized multi-step decoding loop, and PredictionResult construction.

Subclasses implement ``_predict_single`` to return one predicted symbolic
string at a time; the base class handles the decode loop and stopping.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.symbolic import symbolic_to_operations_detailed
from next_action_pred_eval.evaluation.solver import (
    DecodingConfig,
    ISolver,
    PredictionResult,
)

from .featurizer import (
    FeaturizedOp,
    OperationFeaturizer,
    build_symbolic,
    featurize_predicted,
    reconstruct_value,
)

logger = logging.getLogger(__name__)


def _to_symbolic_list(actions: List[Union[str, Operation]]) -> List[str]:
    """Convert mixed list of string/Operation to symbolic strings."""
    out: list = []
    for a in actions:
        if isinstance(a, str):
            out.append(a)
        else:
            out.append(a.to_symbolic())
    return out


class FeatureSolver(ISolver, ABC):
    """Shared base for all feature-based baseline solvers.

    Handles:
    - Incremental featurization of new operations.
    - History rewind detection (length shrinks → full re-featurize).
    - Centralized multi-step decoding loop with configurable stopping.
    - PredictionResult construction with parsing of predicted symbolic strings.

    Subclasses implement ``_predict_single`` to produce one predicted symbolic
    string from the current virtual history.  For solvers that natively predict
    multiple steps (e.g., OnlineNGram suffix matching), override
    ``_predict_multi`` instead.
    """

    def __init__(
        self,
        range_mode: str = "relative",
        decoding: Optional[DecodingConfig] = None,
    ) -> None:
        if range_mode not in ("absolute", "relative"):
            raise ValueError(f"range_mode must be 'absolute' or 'relative', got {range_mode!r}")
        self.range_mode = range_mode
        self.decoding = decoding or DecodingConfig()
        self._featurizer = OperationFeaturizer()
        self._feat_count: int = 0

    # ── Public ISolver interface ──────────────────────────────────────────

    def predict(
        self,
        previous_actions: List[Union[Operation, str]],
        workbook_state: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        t0 = time.time()
        history = self._update_features(previous_actions)
        pred_symbolic = self._decode(history)
        return self._make_result(pred_symbolic, time.time() - t0)

    def reset(self) -> None:
        self._featurizer.reset()
        self._feat_count = 0
        self._on_reset()

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        ...

    # ── Decoding loop ─────────────────────────────────────────────────────

    def _decode(self, history: List[FeaturizedOp]) -> List[str]:
        """Centralized multi-step decoding with configurable stopping.

        Calls ``_predict_multi`` which by default loops over
        ``_predict_single``, applying ``DecodingConfig`` stopping rules.
        """
        return self._predict_multi(history)

    def _predict_multi(self, history: List[FeaturizedOp]) -> List[str]:
        """Default multi-step decoding: loop _predict_single with stopping.

        Override this for solvers that natively produce multi-step
        predictions (e.g., suffix matching).
        """
        if not history:
            return []

        predictions: list = []
        first_op_type: Optional[str] = None
        virtual_history = list(history)

        for k in range(self.decoding.max_predictions):
            sym = self._predict_single(virtual_history)
            if sym is None:
                break

            op_type = sym.split(" | ")[0].strip()

            # Stop on type change (if configured)
            if self.decoding.stop_on_type_change:
                if k == 0:
                    first_op_type = op_type
                elif op_type != first_op_type:
                    break

            predictions.append(sym)

            # Update virtual history for next step
            prev_parsed = virtual_history[-1].parsed
            vfeat = featurize_predicted(sym, prev_parsed)
            virtual_history.append(vfeat)

        return predictions

    # ── Subclass hooks ────────────────────────────────────────────────────

    def _predict_single(
        self, history: List[FeaturizedOp]
    ) -> Optional[str]:
        """Predict one symbolic operation from featurized history.

        Return None to signal "no more predictions".
        Override this for single-step solvers (LSTM, XGBoost, NGram).
        """
        raise NotImplementedError(
            "Subclass must implement _predict_single or override _predict_multi"
        )

    def _on_reset(self) -> None:
        """Override for solver-specific reset logic."""

    # ── Shared helpers ────────────────────────────────────────────────────

    def _update_features(
        self, previous_actions: List[Union[str, Operation]]
    ) -> List[FeaturizedOp]:
        ops = _to_symbolic_list(previous_actions)

        # Handle history rewind
        if len(ops) < self._feat_count:
            self._featurizer.reset()
            self._feat_count = 0
            self._on_reset()

        for i in range(self._feat_count, len(ops)):
            self._featurizer.featurize_one(ops[i])
        self._feat_count = len(ops)
        return self._featurizer.history

    def _make_result(
        self,
        pred_symbolic: List[str],
        gen_time: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        try:
            parse_result = symbolic_to_operations_detailed(pred_symbolic)
            pred_ops = parse_result.valid_operations
            valid_sym = parse_result.valid_symbolic
        except Exception:
            pred_ops = []
            valid_sym = []
        return PredictionResult(
            predicted_operations=pred_ops,
            predicted_symbolic=valid_sym,
            input_tokens=0,
            output_tokens=0,
            generation_time=gen_time,
            metadata=metadata or {},
        )

    def _reconstruct_op(
        self,
        op_type: str,
        value_type: str,
        sheet: str,
        start_row: int,
        start_col: int,
        height: int,
        width: int,
        history: List[FeaturizedOp],
    ) -> str:
        """Reconstruct a full symbolic string from predicted features."""
        value = reconstruct_value(
            op_type, value_type, history, start_row, start_col
        )
        return build_symbolic(
            op_type, sheet, start_row, start_col, height, width, value
        )
