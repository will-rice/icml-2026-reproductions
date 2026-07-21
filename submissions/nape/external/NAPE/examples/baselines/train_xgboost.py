#!/usr/bin/env python
"""
Train XGBoost next-action predictors on operation sequences.

Trains 5 models: 1 classifier (op_type) + 4 regressors (range features).

Usage:
    python examples/baselines/train_xgboost.py \
        --data path/to/sequences.jsonl \
        --output-dir examples/baselines/models/xgboost_rel \
        --range-mode relative

    python examples/baselines/train_xgboost.py \
        --data path/to/sequences.jsonl \
        --output-dir examples/baselines/models/xgboost_abs \
        --range-mode absolute
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from next_action_pred_eval.evaluation.baselines.featurizer import (
    DEFAULT_VALUES,
    NUM_OP_TYPES,
    OP_TYPES,
    FeaturizedOp,
    OperationFeaturizer,
    bucket_absolute,
    bucket_relative,
)
from next_action_pred_eval.evaluation.baselines.xgboost_solver import XGBoostSolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

WINDOW_SIZE = 10


def _range_target(feat: FeaturizedOp, range_mode: str):
    """Returns (r1, r2, r3, r4) as regression targets."""
    if range_mode == "absolute":
        return (
            float(feat.parsed.start_row),
            float(feat.parsed.start_col),
            float(feat.parsed.height),
            float(feat.parsed.width),
        )
    return (
        float(feat.row_delta),
        float(feat.col_delta),
        float(feat.height_delta),
        float(feat.width_delta),
    )


def build_dataset(
    data_path: str,
    range_mode: str,
    window_size: int = WINDOW_SIZE,
    max_sequences: int = 0,
):
    """Build feature matrix X and target arrays from JSONL."""
    featurizer = OperationFeaturizer()
    # We need a temporary solver instance just for feature extraction
    # Create a dummy one — we only use extract_features()
    X_rows: list = []
    y_op: list = []
    y_r1: list = []
    y_r2: list = []
    y_r3: list = []
    y_r4: list = []
    value_counts: Dict[str, Counter] = {}

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

        for t in range(1, len(feats)):
            history = feats[: t]
            target = feats[t]

            feat_vec = _extract_features_static(history, range_mode, window_size)
            X_rows.append(feat_vec)

            op_id = target.parsed.op_type_id
            if op_id < 0 or op_id >= NUM_OP_TYPES:
                op_id = 0
            y_op.append(op_id)

            r1, r2, r3, r4 = _range_target(target, range_mode)
            y_r1.append(r1)
            y_r2.append(r2)
            y_r3.append(r3)
            y_r4.append(r4)

            # Track value counts per op_type
            ot = target.parsed.op_type
            if ot not in value_counts:
                value_counts[ot] = Counter()
            value_counts[ot][target.parsed.raw_value] += 1

        seq_count += 1
        if max_sequences and seq_count >= max_sequences:
            break
        if seq_count % 500 == 0:
            logger.info("  processed %d sequences, %d samples", seq_count, len(X_rows))

    X = np.array(X_rows, dtype=np.float32)
    logger.info("Dataset: %d sequences → %d samples, %d features", seq_count, X.shape[0], X.shape[1])

    # Build value lookup: op_type → most common raw_value
    value_lookup = {}
    for ot, cnts in value_counts.items():
        if cnts:
            value_lookup[ot] = cnts.most_common(1)[0][0]

    return X, np.array(y_op), np.array(y_r1), np.array(y_r2), np.array(y_r3), np.array(y_r4), value_lookup


def _extract_features_static(
    history: List[FeaturizedOp],
    range_mode: str,
    window_size: int,
) -> np.ndarray:
    """Static feature extraction (mirrors XGBoostSolver.extract_features)."""
    W = window_size
    window = history[-W:] if len(history) >= W else history
    n = len(window)

    per_pos = []
    for i in range(W):
        if i < W - n:
            per_pos.extend([0.0] * 6)
        else:
            f = window[i - (W - n)]
            op_id = float(f.parsed.op_type_id)
            if range_mode == "absolute":
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
            per_pos.extend([op_id, r1, r2, r3, r4, vtype])

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


def train(
    data_path: str,
    output_dir: str,
    range_mode: str = "relative",
    max_sequences: int = 5000,
    n_estimators: int = 200,
    max_depth: int = 6,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Building dataset (range_mode=%s, max_seq=%d)...", range_mode, max_sequences)
    X, y_op, y_r1, y_r2, y_r3, y_r4, value_lookup = build_dataset(
        data_path, range_mode, max_sequences=max_sequences
    )

    if X.shape[0] == 0:
        logger.error("No training samples!")
        return

    common_params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
    )

    # Op-type classifier — remap to contiguous labels for XGBoost
    logger.info("Training op_type classifier...")
    t0 = time.time()
    unique_labels = np.sort(np.unique(y_op))
    label_to_idx = {int(lab): i for i, lab in enumerate(unique_labels)}
    idx_to_label = {i: int(lab) for i, lab in enumerate(unique_labels)}
    y_op_mapped = np.array([label_to_idx[int(y)] for y in y_op])

    clf = XGBClassifier(
        num_class=len(unique_labels),
        objective="multi:softmax",
        eval_metric="mlogloss",
        **common_params,
    )
    clf.fit(X, y_op_mapped)
    joblib.dump(clf, out / "op_type_classifier.joblib")
    # Save label mapping
    with open(out / "label_mapping.json", "w") as f:
        json.dump({"idx_to_label": {str(k): v for k, v in idx_to_label.items()}}, f)
    logger.info("  done (%.1fs, %d classes)", time.time() - t0, len(unique_labels))

    # Range regressors
    for name, y in [("r1", y_r1), ("r2", y_r2), ("r3", y_r3), ("r4", y_r4)]:
        logger.info("Training %s regressor...", name)
        t0 = time.time()
        reg = XGBRegressor(
            objective="reg:squarederror",
            **common_params,
        )
        reg.fit(X, y)
        joblib.dump(reg, out / f"{name}_regressor.joblib")
        logger.info("  done (%.1fs)", time.time() - t0)

    # Save config & value lookup
    config = {
        "range_mode": range_mode,
        "window_size": WINDOW_SIZE,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out / "value_lookup.json", "w") as f:
        json.dump(value_lookup, f, ensure_ascii=False, indent=2)

    logger.info("All models saved to %s", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train v2 XGBoost models")
    p.add_argument("--data", required=True, help="Path to sequences JSONL")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--range-mode", choices=["absolute", "relative"], default="relative")
    p.add_argument("--max-sequences", type=int, default=5000, help="0=all")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=6)
    args = p.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output_dir,
        range_mode=args.range_mode,
        max_sequences=args.max_sequences,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
