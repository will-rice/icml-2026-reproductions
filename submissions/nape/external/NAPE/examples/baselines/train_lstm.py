#!/usr/bin/env python
"""
Train a small GRU-based next-action predictor on operation sequences.

Usage:
    python examples/baselines/train_lstm.py \
        --data path/to/sequences.jsonl \
        --output-dir examples/baselines/models/lstm_rel \
        --range-mode relative \
        --epochs 15

    python examples/baselines/train_lstm.py \
        --data path/to/sequences.jsonl \
        --output-dir examples/baselines/models/lstm_abs \
        --range-mode absolute \
        --epochs 15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure package is importable
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from next_action_pred_eval.evaluation.baselines.featurizer import (
    NUM_OP_TYPES,
    NUM_VALUE_TYPES,
    RANGE_BUCKETS,
    FeaturizedOp,
    OperationFeaturizer,
    bucket_absolute,
    bucket_relative,
)
from next_action_pred_eval.evaluation.baselines.lstm_model import GRUPredictionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CONTEXT_WINDOW = 32
PAD = 0


# ═══════════════════════════════════════════════════════════════════════════ #
#  Dataset                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #


def _tokenize_feat(feat: FeaturizedOp, range_mode: str) -> Tuple[int, ...]:
    """Convert a FeaturizedOp to 6-int token tuple (+1 shifted for padding)."""
    op_id = feat.parsed.op_type_id + 1
    if op_id < 1 or op_id > NUM_OP_TYPES:
        op_id = NUM_OP_TYPES + 1

    if range_mode == "absolute":
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
    return (op_id, r1, r2, r3, r4, vtype)


def _make_target(feat: FeaturizedOp, range_mode: str) -> Tuple[int, ...]:
    """Target labels for next-step prediction (NOT +1 shifted)."""
    op_id = feat.parsed.op_type_id
    if op_id < 0 or op_id >= NUM_OP_TYPES:
        op_id = NUM_OP_TYPES  # unknown class

    if range_mode == "absolute":
        r1 = bucket_absolute(feat.parsed.start_row)
        r2 = bucket_absolute(feat.parsed.start_col)
        r3 = bucket_absolute(feat.parsed.height)
        r4 = bucket_absolute(feat.parsed.width)
    else:
        r1 = bucket_relative(feat.row_delta)
        r2 = bucket_relative(feat.col_delta)
        r3 = bucket_relative(feat.height_delta)
        r4 = bucket_relative(feat.width_delta)

    vtype = feat.parsed.value_type_id
    return (op_id, r1, r2, r3, r4, vtype)


class SequenceDataset(Dataset):
    """Windowed next-step prediction dataset."""

    def __init__(
        self,
        data_path: str,
        range_mode: str,
        context_window: int = CONTEXT_WINDOW,
        max_sequences: int = 0,
    ) -> None:
        self.context_window = context_window
        self.samples: list = []  # list of (input_tokens, target_tokens)

        featurizer = OperationFeaturizer()
        path = Path(data_path)
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
            if len(ops) < 3:
                continue

            featurizer.reset()
            feats = [featurizer.featurize_one(op) for op in ops]
            tokens = [_tokenize_feat(f, range_mode) for f in feats]
            targets = [_make_target(f, range_mode) for f in feats]

            # Create sliding windows: input[t-W:t] → target[t]
            for t in range(1, len(feats)):
                start = max(0, t - context_window)
                window = tokens[start:t]
                # Pad from left
                pad_n = context_window - len(window)
                padded = [(PAD,) * 6] * pad_n + window
                self.samples.append((padded, targets[t]))

            seq_count += 1
            if max_sequences and seq_count >= max_sequences:
                break

        logger.info(
            "Loaded %d sequences → %d training samples", seq_count, len(self.samples)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        inp, tgt = self.samples[idx]
        inp_t = torch.tensor(inp, dtype=torch.long)  # (W, 6)
        tgt_t = torch.tensor(tgt, dtype=torch.long)  # (6,)
        return inp_t, tgt_t


def _collate(batch):
    inputs = torch.stack([b[0] for b in batch])   # (B, W, 6)
    targets = torch.stack([b[1] for b in batch])   # (B, 6)
    return inputs, targets


# ═══════════════════════════════════════════════════════════════════════════ #
#  Training loop                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #


def train(
    data_path: str,
    output_dir: str,
    range_mode: str = "relative",
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    max_sequences: int = 0,
    val_split: float = 0.05,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset (range_mode=%s)...", range_mode)
    full_ds = SequenceDataset(
        data_path, range_mode, max_sequences=max_sequences
    )
    if len(full_ds) == 0:
        logger.error("No training samples!")
        return

    # Train/val split
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    logger.info("Train: %d, Val: %d", n_train, n_val)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    model = GRUPredictionModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for inputs, targets in train_dl:
            inputs = inputs.to(device)       # (B, W, 6)
            targets = targets.to(device)     # (B, 6)

            # Unpack inputs
            op_ids = inputs[:, :, 0]
            r1, r2, r3, r4 = inputs[:, :, 1], inputs[:, :, 2], inputs[:, :, 3], inputs[:, :, 4]
            vtype_ids = inputs[:, :, 5]

            logits = model(op_ids, r1, r2, r3, r4, vtype_ids)

            # Losses on last timestep
            loss = (
                2.0 * ce(logits["op_type"][:, -1], targets[:, 0])
                + 1.0 * ce(logits["r1"][:, -1], targets[:, 1])
                + 1.0 * ce(logits["r2"][:, -1], targets[:, 2])
                + 1.0 * ce(logits["r3"][:, -1], targets[:, 3])
                + 1.0 * ce(logits["r4"][:, -1], targets[:, 4])
                + 1.0 * ce(logits["value_type"][:, -1], targets[:, 5])
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct_op = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_dl:
                inputs = inputs.to(device)
                targets = targets.to(device)
                op_ids = inputs[:, :, 0]
                r1, r2, r3, r4 = inputs[:, :, 1], inputs[:, :, 2], inputs[:, :, 3], inputs[:, :, 4]
                vtype_ids = inputs[:, :, 5]
                logits = model(op_ids, r1, r2, r3, r4, vtype_ids)
                vloss = (
                    2.0 * ce(logits["op_type"][:, -1], targets[:, 0])
                    + 1.0 * ce(logits["r1"][:, -1], targets[:, 1])
                    + 1.0 * ce(logits["r2"][:, -1], targets[:, 2])
                    + 1.0 * ce(logits["r3"][:, -1], targets[:, 3])
                    + 1.0 * ce(logits["r4"][:, -1], targets[:, 4])
                    + 1.0 * ce(logits["value_type"][:, -1], targets[:, 5])
                )
                val_loss += vloss.item() * inputs.size(0)
                preds = logits["op_type"][:, -1].argmax(dim=1)
                val_correct_op += (preds == targets[:, 0]).sum().item()
                val_total += inputs.size(0)

        avg_val = val_loss / max(val_total, 1)
        val_acc = val_correct_op / max(val_total, 1)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_op_acc=%.3f  (%.1fs)",
            epoch, epochs, avg_train, avg_val, val_acc, elapsed,
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), out / "model.pt")

    # Save config
    config = model.get_config()
    config["range_mode"] = range_mode
    config["context_window"] = CONTEXT_WINDOW
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Done. Model saved to %s", out)


# ═══════════════════════════════════════════════════════════════════════════ #
#  CLI                                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train v2 GRU model")
    p.add_argument("--data", required=True, help="Path to sequences JSONL")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--range-mode", choices=["absolute", "relative"], default="relative")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-sequences", type=int, default=0, help="0=all")
    args = p.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output_dir,
        range_mode=args.range_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_sequences=args.max_sequences,
    )
