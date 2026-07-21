"""
GRU model architecture for v2 LSTM solver.

6-feature input per timestep:
  [op_type_id, range_f1, range_f2, range_f3, range_f4, value_type_id]

Absolute mode: range_f = (row_bucket, col_bucket, height_bucket, width_bucket)
Relative mode: range_f = (dr_bucket, dc_bucket, dh_bucket, dw_bucket)

All range features use 101 buckets (RANGE_BUCKETS).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from next_action_pred_eval.evaluation.baselines.featurizer import (
    NUM_OP_TYPES, NUM_VALUE_TYPES, RANGE_BUCKETS, UNKNOWN_OP_ID,
)


class GRUPredictionModel(nn.Module):
    """Multi-head GRU for next-operation prediction.

    Embedding per feature → concat → projection → GRU → 6 output heads.
    """

    def __init__(
        self,
        embed_op: int = 32,
        embed_range: int = 16,
        embed_value: int = 16,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_op = embed_op
        self.embed_range = embed_range
        self.embed_value = embed_value
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # +1 for <PAD> and +1 for unknown
        num_op_emb = NUM_OP_TYPES + 2
        num_range_emb = RANGE_BUCKETS + 1   # 0..100 + padding
        num_vtype_emb = NUM_VALUE_TYPES + 1  # + padding

        # Embeddings
        self.emb_op = nn.Embedding(num_op_emb, embed_op, padding_idx=0)
        self.emb_r1 = nn.Embedding(num_range_emb, embed_range, padding_idx=0)
        self.emb_r2 = nn.Embedding(num_range_emb, embed_range, padding_idx=0)
        self.emb_r3 = nn.Embedding(num_range_emb, embed_range, padding_idx=0)
        self.emb_r4 = nn.Embedding(num_range_emb, embed_range, padding_idx=0)
        self.emb_vtype = nn.Embedding(num_vtype_emb, embed_value, padding_idx=0)

        input_dim = embed_op + 4 * embed_range + embed_value  # 32+64+16=112
        self.proj = nn.Linear(input_dim, hidden_size)

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output heads
        self.head_op = nn.Linear(hidden_size, NUM_OP_TYPES + 1)  # +1 for unknown
        self.head_r1 = nn.Linear(hidden_size, RANGE_BUCKETS)
        self.head_r2 = nn.Linear(hidden_size, RANGE_BUCKETS)
        self.head_r3 = nn.Linear(hidden_size, RANGE_BUCKETS)
        self.head_r4 = nn.Linear(hidden_size, RANGE_BUCKETS)
        self.head_vtype = nn.Linear(hidden_size, NUM_VALUE_TYPES)

    def forward(
        self,
        op_ids: torch.Tensor,    # (B, T)
        r1: torch.Tensor,        # (B, T)
        r2: torch.Tensor,        # (B, T)
        r3: torch.Tensor,        # (B, T)
        r4: torch.Tensor,        # (B, T)
        vtype_ids: torch.Tensor,  # (B, T)
    ) -> Dict[str, torch.Tensor]:
        """Forward pass, returns logits dict for all 6 heads."""
        e = torch.cat([
            self.emb_op(op_ids),
            self.emb_r1(r1),
            self.emb_r2(r2),
            self.emb_r3(r3),
            self.emb_r4(r4),
            self.emb_vtype(vtype_ids),
        ], dim=-1)  # (B, T, input_dim)

        h = torch.relu(self.proj(e))  # (B, T, hidden_size)
        out, _ = self.gru(h)          # (B, T, hidden_size)

        return {
            "op_type": self.head_op(out),     # (B, T, NUM_OP_TYPES+1)
            "r1": self.head_r1(out),          # (B, T, RANGE_BUCKETS)
            "r2": self.head_r2(out),
            "r3": self.head_r3(out),
            "r4": self.head_r4(out),
            "value_type": self.head_vtype(out),  # (B, T, NUM_VALUE_TYPES)
        }

    def get_config(self) -> Dict:
        return {
            "embed_op": self.embed_op,
            "embed_range": self.embed_range,
            "embed_value": self.embed_value,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }
