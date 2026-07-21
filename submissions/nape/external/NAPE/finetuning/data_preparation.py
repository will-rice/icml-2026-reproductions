"""
Data Export for Finetuning
==========================

Walks a training-data directory tree, collects **raw** operation
sequences (no preprocessing), filters outliers by op-count percentile,
and exports them as a human-readable JSONL file.

**No preprocessing is applied at export time.**  Preprocessing (value
shortening, sheet-name stripping) is done lazily in the training
dataloader so that changing a config param takes effect immediately
without re-export.

Output format (one JSON object per line)::

    {"id": "folder_name", "ops": ["SetValue | A1 Sheet1 | ...", ...]}

Usage::

    # Export raw sequences (one-time, ~30 seconds)
    python finetuning/data_preparation.py

    # Export with custom percentile filter
    python finetuning/data_preparation.py --max_percentile 99

    # Preview sample training examples (applies preprocessing on the fly)
    python finetuning/data_preparation.py --preview 5 --tokenizer Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make the parent package importable when running as a script
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# Default prompt template — matches CompletionSolver's DEFAULT_COMPLETION_TEMPLATE
DEFAULT_PROMPT_TEMPLATE = (
    "Complete the sequence of actions to build the following "
    "spreadsheet by identifying and extending key patterns.\n\n{actions}"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_sequences(data_dir: Path) -> List[Dict]:
    """Walk ``data_dir`` and collect raw operation sequences.

    Returns a list of dicts with keys ``"id"`` (folder name) and
    ``"ops"`` (list of raw operation strings).
    """
    results: List[Dict] = []
    missing = 0

    subdirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    for subdir in tqdm(subdirs, desc="Loading raw sequences"):
        fw_path = subdir / "sequences" / "framework_output.txt"
        if not fw_path.exists():
            missing += 1
            continue

        raw_ops = [
            line
            for line in fw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not raw_ops:
            continue

        results.append({"id": subdir.name, "ops": raw_ops})

    logger.info(
        "Loaded %d sequences (%d folders missing framework_output.txt)",
        len(results),
        missing,
    )
    return results


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_by_percentile(
    data: List[Dict], max_percentile: float = 95.0
) -> List[Dict]:
    """Remove sequences with op count above the given percentile.

    Args:
        data: List of sequence dicts.
        max_percentile: Keep sequences at or below this percentile.

    Returns:
        Filtered list (original list is not mutated).
    """
    if not data:
        return data

    op_counts = sorted(len(d["ops"]) for d in data)
    idx = int(len(op_counts) * max_percentile / 100)
    threshold = op_counts[min(idx, len(op_counts) - 1)]

    before = len(data)
    filtered = [d for d in data if len(d["ops"]) <= threshold]

    logger.info(
        "Filtered by p%.0f (threshold=%d ops): %d -> %d sequences (removed %d)",
        max_percentile,
        threshold,
        before,
        len(filtered),
        before - len(filtered),
    )
    return filtered


# ---------------------------------------------------------------------------
# Persistence — JSONL
# ---------------------------------------------------------------------------

def save_jsonl(data: List[Dict], output_path: Path) -> None:
    """Save sequences as JSONL (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Saved %d sequences to %s (%.1f MB)", len(data), output_path, size_mb
    )


def load_jsonl(path: Path) -> List[Dict]:
    """Load sequences from a JSONL file."""
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info("Loaded %d sequences from %s", len(data), path)
    return data


# ---------------------------------------------------------------------------
# Statistics / preview
# ---------------------------------------------------------------------------

def compute_stats(
    data: List[Dict],
    max_context_ops: int,
    example_stride: int,
) -> Dict:
    """Compute dataset statistics."""
    op_counts = [len(d["ops"]) for d in data]
    total_ops = sum(op_counts)

    total_examples = 0
    for n in op_counts:
        if n < 2:
            continue
        total_examples += max(1, (n - 1) // example_stride + 1)

    return {
        "sequences": len(data),
        "total_ops": total_ops,
        "avg_ops": total_ops / max(len(data), 1),
        "min_ops": min(op_counts) if op_counts else 0,
        "max_ops": max(op_counts) if op_counts else 0,
        "max_context_ops": max_context_ops,
        "example_stride": example_stride,
        "estimated_examples": total_examples,
    }


def preview_examples(
    data: List[Dict],
    *,
    max_context_ops: int = 128,
    example_stride: int = 64,
    n: int = 5,
    tokenizer_name: Optional[str] = None,
    # Preprocessing params (applied on the fly for preview)
    enable_context_shortening: bool = True,
    context_shortening_max_chars: int = 32,
    context_shortening_corner_cells_dim: int = 2,
    remove_sheet_name: bool = True,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> None:
    """Print sample training examples with preprocessing applied on the fly.

    This shows exactly what the model will see during training, without
    requiring a pre-processed dataset.
    """
    import random

    from next_action_pred_eval.core.symbolic import compress_symbolic
    from next_action_pred_eval.evaluation.baselines.prompts import (
        shorten_symbolic_values,
    )

    def preprocess(ops: List[str]) -> List[str]:
        result = list(ops)
        if enable_context_shortening:
            result = shorten_symbolic_values(
                result,
                max_value_length=context_shortening_max_chars,
                corner_cells_dim=context_shortening_corner_cells_dim,
            )
        if remove_sheet_name:
            result = compress_symbolic(result, remove_sheet_name=True)
        return result

    def format_text(ops: List[str]) -> str:
        actions = "\n".join(f"{i}. {a}" for i, a in enumerate(ops, 1))
        return prompt_template.format(actions=actions)

    # ---- Statistics ----------------------------------------------------------
    stats = compute_stats(data, max_context_ops, example_stride)
    print(f"\n{'=' * 70}")
    print("Dataset Statistics")
    print(f"{'=' * 70}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:,.1f}")
        else:
            print(f"  {k:25s}: {v:,}")
    print(f"{'=' * 70}\n")

    # ---- Tokenizer (optional) -----------------------------------------------
    tokenizer = None
    if tokenizer_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )

    # ---- Sample examples ----------------------------------------------------
    rng = random.Random(42)
    sample_indices = rng.sample(range(len(data)), min(n, len(data)))

    for i, seq_idx in enumerate(sample_indices):
        d = data[seq_idx]
        ops = d["ops"]
        n_ops = len(ops)

        # Pick a random start position within the sequence
        start = rng.randint(0, max(0, n_ops - max_context_ops))
        window = ops[start : start + max_context_ops]

        # Preprocess + format (same as the dataloader does)
        processed = preprocess(window)
        text = format_text(processed)

        token_info = ""
        if tokenizer:
            tok_count = len(tokenizer.encode(text, add_special_tokens=True))
            token_info = f", tokens={tok_count}"

        print(
            f"--- Example {i + 1}/{n}  "
            f"(seq={seq_idx}, id={d['id']}, "
            f"start_op={start}, ops_in_window={len(window)}{token_info}) ---"
        )

        # Show truncated view for long examples
        lines = text.split("\n")
        if len(lines) <= 25:
            print(text)
        else:
            for line in lines[:12]:
                print(line)
            print(f"    ... ({len(lines) - 17} more lines) ...")
            for line in lines[-5:]:
                print(line)

        print()

    # ---- Token distribution (if tokenizer given) ----------------------------
    if tokenizer and len(data) > 0:
        print(f"{'=' * 70}")
        print("Token length distribution  (sampled 200 random examples)")
        print(f"{'=' * 70}")
        sample_size = min(200, len(data))
        sample_seqs = rng.sample(data, sample_size)
        token_lengths = []
        for d in sample_seqs:
            ops = d["ops"]
            start = rng.randint(0, max(0, len(ops) - max_context_ops))
            window = ops[start : start + max_context_ops]
            processed = preprocess(window)
            text = format_text(processed)
            token_lengths.append(
                len(tokenizer.encode(text, add_special_tokens=True))
            )
        token_lengths.sort()
        print(f"  min:    {token_lengths[0]:,}")
        print(f"  p25:    {token_lengths[len(token_lengths) // 4]:,}")
        print(f"  median: {token_lengths[len(token_lengths) // 2]:,}")
        print(f"  p75:    {token_lengths[3 * len(token_lengths) // 4]:,}")
        print(f"  max:    {token_lengths[-1]:,}")
        print(f"  mean:   {sum(token_lengths) / len(token_lengths):,.0f}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Export raw operation sequences for finetuning.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=_PROJECT_ROOT / "results" / "generation" / "train_data",
        help="Path to the generated training-data directory (per-trajectory subfolders).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=_PROJECT_ROOT / "finetuning" / "processed_data" / "sequences.jsonl",
        help="Where to save the JSONL file.",
    )
    parser.add_argument(
        "--max_percentile",
        type=float,
        default=95.0,
        help="Filter out sequences above this percentile by op count (default: 95).",
    )

    # Preview
    prev = parser.add_argument_group("preview")
    prev.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print N sample training examples (0 = skip).",
    )
    prev.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name for token-count stats in preview.",
    )
    prev.add_argument("--max_context_ops", type=int, default=128)
    prev.add_argument("--example_stride", type=int, default=64)

    args = parser.parse_args()

    # Load raw sequences from disk
    data = load_raw_sequences(args.data_dir)

    # Filter outliers
    data = filter_by_percentile(data, args.max_percentile)

    # Save as JSONL
    save_jsonl(data, args.output_path)

    # Optional preview
    if args.preview > 0:
        preview_examples(
            data,
            max_context_ops=args.max_context_ops,
            example_stride=args.example_stride,
            n=args.preview,
            tokenizer_name=args.tokenizer,
        )


if __name__ == "__main__":
    main()
