"""
XGBoost Solver — gradient boosted tree baseline.

5-head design:
  - op_type classifier
  - 4 range regressors (absolute or relative depending on mode)

Feature vector from a window of the last K operations.
Uses ``_predict_single`` for centralized decoding in the base class.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from next_action_pred_eval.evaluation.solver import DecodingConfig

from .feature_solver import FeatureSolver
from .featurizer import (
    NUM_OP_TYPES,
    OP_TYPES,
    FeaturizedOp,
    featurize_predicted,
)

logger = logging.getLogger(__name__)


class XGBoostSolver(FeatureSolver):
    """XGBoost-based solver with absolute / relative range modes.

    Args:
        model_dir: Directory with trained XGBoost models + config.
        range_mode: ``"absolute"`` or ``"relative"``
        window_size: Number of recent ops for feature extraction (default 10).
        decoding: Decoding configuration for stopping logic.
    """

    def __init__(
        self,
        model_dir: str,
        range_mode: str = "relative",
        window_size: int = 10,
        decoding: Optional[DecodingConfig] = None,
    ) -> None:
        super().__init__(range_mode=range_mode, decoding=decoding)
        self.window_size = window_size

        mdir = Path(model_dir)
        self._op_model = joblib.load(mdir / "op_type_classifier.joblib")
        self._r1_model = joblib.load(mdir / "r1_regressor.joblib")
        self._r2_model = joblib.load(mdir / "r2_regressor.joblib")
        self._r3_model = joblib.load(mdir / "r3_regressor.joblib")
        self._r4_model = joblib.load(mdir / "r4_regressor.joblib")

        config_path = mdir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self._model_config = json.load(f)
        else:
            self._model_config = {}

        value_path = mdir / "value_lookup.json"
        if value_path.exists():
            with open(value_path) as f:
                self._value_lookup: Dict[str, str] = json.load(f)
        else:
            self._value_lookup = {}

        label_path = mdir / "label_mapping.json"
        if label_path.exists():
            with open(label_path) as f:
                mapping = json.load(f)
            self._idx_to_label = {
                int(k): v for k, v in mapping["idx_to_label"].items()
            }
        else:
            self._idx_to_label = {i: i for i in range(NUM_OP_TYPES)}

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "xgboost",
            "range_mode": self.range_mode,
            "window_size": self.window_size,
            "decoding": self.decoding.model_dump(),
        }

    # ── Feature extraction ────────────────────────────────────────────────

    def extract_features(self, history: List[FeaturizedOp]) -> np.ndarray:
        W = self.window_size
        window = history[-W:] if len(history) >= W else history
        n = len(window)

        per_pos = []
        for i in range(W):
            if i < W - n:
                per_pos.extend([0.0] * 6)
            else:
                f = window[i - (W - n)]
                per_pos.extend(self._op_features(f))

        transitions = []
        for i in range(W - 1):
            idx_a = i - (W - n)
            idx_b = idx_a + 1
            if idx_a < 0 or idx_b < 0 or idx_a >= n or idx_b >= n:
                transitions.extend([0.0, 0.0, 0.0])
            else:
                a, b = window[idx_a], window[idx_b]
                transitions.append(float(b.row_delta))
                transitions.append(float(b.col_delta))
                transitions.append(1.0 if a.parsed.op_type == b.parsed.op_type else 0.0)

        op_hist = [0.0] * NUM_OP_TYPES
        for f in window:
            idx = f.parsed.op_type_id
            if 0 <= idx < NUM_OP_TYPES:
                op_hist[idx] += 1.0

        run_len = 1
        if n >= 2:
            for i in range(n - 2, -1, -1):
                if window[i].parsed.op_type == window[-1].parsed.op_type:
                    run_len += 1
                else:
                    break

        sheet_change = 0.0
        if n >= 2:
            sheets = {f.parsed.sheet for f in window}
            sheet_change = 1.0 if len(sheets) > 1 else 0.0

        global_feats = op_hist + [float(run_len), sheet_change]
        return np.array(per_pos + transitions + global_feats, dtype=np.float32)

    def _op_features(self, f: FeaturizedOp) -> List[float]:
        op_id = float(f.parsed.op_type_id)
        if self.range_mode == "absolute":
            r1 = float(f.parsed.start_row)
            r2 = float(f.parsed.start_col)
            r3 = float(f.parsed.height)
            r4 = float(f.parsed.width)
        else:
            r1 = float(f.row_delta)
            r2 = float(f.col_delta)
            r3 = float(f.height_delta)
            r4 = float(f.width_delta)
        vtype = float(f.parsed.value_type_id)
        return [op_id, r1, r2, r3, r4, vtype]

    # ── Single-step prediction ────────────────────────────────────────────

    def _predict_single(self, history: List[FeaturizedOp]) -> Optional[str]:
        if not history:
            return None

        feat_vec = self.extract_features(history).reshape(1, -1)

        raw_pred = int(self._op_model.predict(feat_vec)[0])
        op_type_id = self._idx_to_label.get(raw_pred, -1)
        if op_type_id < 0 or op_type_id >= NUM_OP_TYPES:
            return None
        op_type = OP_TYPES[op_type_id]

        pr1 = float(self._r1_model.predict(feat_vec)[0])
        pr2 = float(self._r2_model.predict(feat_vec)[0])
        pr3 = float(self._r3_model.predict(feat_vec)[0])
        pr4 = float(self._r4_model.predict(feat_vec)[0])

        last = history[-1].parsed
        if self.range_mode == "absolute":
            sr = max(1, round(pr1))
            sc = max(1, round(pr2))
            h = max(1, round(pr3))
            w = max(1, round(pr4))
        else:
            sr = max(1, last.start_row + round(pr1))
            sc = max(1, last.start_col + round(pr2))
            h = max(1, last.height + round(pr3))
            w = max(1, last.width + round(pr4))

        vtype = self._guess_value_type(op_type, history)

        return self._reconstruct_op(
            op_type, vtype, last.sheet, sr, sc, h, w, history
        )

    def _guess_value_type(
        self, op_type: str, history: List[FeaturizedOp]
    ) -> str:
        for f in reversed(history):
            if f.parsed.op_type == op_type:
                return f.parsed.value_type
        return "other"
