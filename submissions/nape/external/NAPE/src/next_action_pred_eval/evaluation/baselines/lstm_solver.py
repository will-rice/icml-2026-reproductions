"""
LSTM/GRU Solver — wraps the GRU model for ISolver interface.

6-feature tokenization per step, auto-regressive multi-step prediction.
Uses ``_predict_single`` for centralized decoding in the base class.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from next_action_pred_eval.evaluation.solver import DecodingConfig

from .feature_solver import FeatureSolver
from .featurizer import (
    NUM_OP_TYPES,
    NUM_VALUE_TYPES,
    OP_TYPES,
    RANGE_BUCKETS,
    VALUE_TYPES,
    FeaturizedOp,
    bucket_absolute,
    bucket_relative,
    featurize_predicted,
    unbucket_absolute,
    unbucket_relative,
)
from .lstm_model import GRUPredictionModel

logger = logging.getLogger(__name__)

CONTEXT_WINDOW = 32
PAD_TOKEN = 0


class LSTMSolver(FeatureSolver):
    """GRU-based solver with absolute / relative range modes.

    Args:
        model_dir: Directory containing ``model.pt`` and ``config.json``.
        range_mode: ``"absolute"`` or ``"relative"``
        device: ``"cpu"`` or ``"cuda"``
        context_window: Number of recent ops for input (default 32).
        decoding: Decoding configuration for stopping logic.
    """

    def __init__(
        self,
        model_dir: str,
        range_mode: str = "relative",
        device: str = "cpu",
        context_window: int = CONTEXT_WINDOW,
        decoding: Optional[DecodingConfig] = None,
    ) -> None:
        super().__init__(range_mode=range_mode, decoding=decoding)
        self.device = torch.device(device)
        self.context_window = context_window

        mdir = Path(model_dir)
        with open(mdir / "config.json", "r") as f:
            self._model_config = json.load(f)

        self._model = GRUPredictionModel(
            embed_op=self._model_config.get("embed_op", 32),
            embed_range=self._model_config.get("embed_range", 16),
            embed_value=self._model_config.get("embed_value", 16),
            hidden_size=self._model_config.get("hidden_size", 128),
            num_layers=self._model_config.get("num_layers", 2),
        )
        state = torch.load(mdir / "model.pt", map_location=self.device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.to(self.device)
        self._model.eval()

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "lstm",
            "range_mode": self.range_mode,
            "context_window": self.context_window,
            "decoding": self.decoding.model_dump(),
            "model_config": self._model_config,
        }

    # ── Tokenization ──────────────────────────────────────────────────────

    def _tokenize(self, feat: FeaturizedOp) -> Tuple[int, int, int, int, int, int]:
        op_id = feat.parsed.op_type_id + 1
        if op_id < 1 or op_id > NUM_OP_TYPES:
            op_id = NUM_OP_TYPES + 1

        if self.range_mode == "absolute":
            r1 = bucket_absolute(feat.parsed.start_row) + 1
            r2 = bucket_absolute(feat.parsed.start_col) + 1
            r3 = bucket_absolute(feat.parsed.height) + 1
            r4 = bucket_absolute(feat.parsed.width) + 1
        else:
            r1 = bucket_relative(feat.row_delta) + 1
            r2 = bucket_relative(feat.col_delta) + 1
            r3 = bucket_relative(feat.height_delta) + 1
            r4 = bucket_relative(feat.width_delta) + 1

        vtype = feat.parsed.value_type_id + 1
        return op_id, r1, r2, r3, r4, vtype

    def _make_input_tensors(
        self, features: List[Tuple[int, int, int, int, int, int]]
    ) -> Tuple[torch.Tensor, ...]:
        if len(features) > self.context_window:
            features = features[-self.context_window :]
        pad_len = self.context_window - len(features)
        padded = [(PAD_TOKEN,) * 6] * pad_len + features
        t = torch.tensor(padded, dtype=torch.long, device=self.device).unsqueeze(0)
        return t[:, :, 0], t[:, :, 1], t[:, :, 2], t[:, :, 3], t[:, :, 4], t[:, :, 5]

    # ── Single-step prediction ────────────────────────────────────────────

    def _predict_single(self, history: List[FeaturizedOp]) -> Optional[str]:
        if not history:
            return None

        tokens = [self._tokenize(f) for f in history]
        with torch.no_grad():
            op_ids, r1, r2, r3, r4, vtype_ids = self._make_input_tensors(tokens)
            logits = self._model(op_ids, r1, r2, r3, r4, vtype_ids)

            pred_op_idx = logits["op_type"][0, -1].argmax().item()
            pred_r1 = logits["r1"][0, -1].argmax().item()
            pred_r2 = logits["r2"][0, -1].argmax().item()
            pred_r3 = logits["r3"][0, -1].argmax().item()
            pred_r4 = logits["r4"][0, -1].argmax().item()
            pred_vtype_idx = logits["value_type"][0, -1].argmax().item()

        op_type_id = pred_op_idx
        if op_type_id < 0 or op_type_id >= NUM_OP_TYPES:
            return None
        op_type = OP_TYPES[op_type_id]

        last = history[-1].parsed
        if self.range_mode == "absolute":
            sr = unbucket_absolute(pred_r1)
            sc = unbucket_absolute(pred_r2)
            h = max(1, unbucket_absolute(pred_r3))
            w = max(1, unbucket_absolute(pred_r4))
        else:
            dr = unbucket_relative(pred_r1)
            dc = unbucket_relative(pred_r2)
            dh = unbucket_relative(pred_r3)
            dw = unbucket_relative(pred_r4)
            sr = max(1, last.start_row + dr)
            sc = max(1, last.start_col + dc)
            h = max(1, last.height + dh)
            w = max(1, last.width + dw)

        vtype_id = pred_vtype_idx
        vtype = VALUE_TYPES[vtype_id] if 0 <= vtype_id < len(VALUE_TYPES) else "other"

        return self._reconstruct_op(
            op_type, vtype, last.sheet, sr, sc, h, w, history
        )
