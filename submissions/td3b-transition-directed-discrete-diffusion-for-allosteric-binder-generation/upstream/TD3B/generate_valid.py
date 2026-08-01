#!/usr/bin/env python3
"""
generate_valid.py -- standalone CLI to generate peptide SMILES with a chosen
validity-boosting sampling strategy (see ``sampling_strategies.py``) and report the
fraction that pass ``utils.app.PeptideAnalyzer.is_peptide``.

Examples
--------
# Real checkpoint, long peptides, nucleus + remask self-correction (recommended default):
python generate_valid.py \
    --ckpt_path checkpoints/td3b.ckpt \
    --length 400 --num_samples 64 \
    --strategy nucleus_remask \
    --device cuda:0 --seed 42 \
    --save_path results/valid_len400.csv

# No checkpoint available -> RANDOM-init model on CPU (development / API smoke test;
# absolute yields are garbage, only the sampling machinery is exercised):
python generate_valid.py --length 200 --num_samples 32 --strategy remask --device cpu

Strategies: baseline, more_steps, top_p (nucleus), top_k, low_temp, remask,
best_of_n, nucleus_remask.  Per-strategy knobs below override the preset defaults.
"""
import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sampling_strategies import generate, build_random_model, available_strategies
from utils.app import PeptideAnalyzer

logger = logging.getLogger("generate_valid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_model(ckpt_path, device, base_path, hidden_size, n_layers, n_heads):
    """Load the real checkpoint via ``inference.load_model`` when available; otherwise
    fall back to a RANDOM-init model (imports of ``inference`` are done lazily so the
    random path has no heavy dependencies)."""
    if ckpt_path and os.path.isfile(ckpt_path):
        logger.info("Loading real checkpoint from %s", ckpt_path)
        from inference import load_model  # reuse the canonical loader
        model, tokenizer = load_model(ckpt_path, device)
        return model, tokenizer, False
    if ckpt_path:
        logger.warning("Checkpoint %s not found -- falling back to RANDOM-init model.", ckpt_path)
    else:
        logger.warning("No --ckpt_path given -- using RANDOM-init model "
                       "(yields are meaningless; API/mechanism check only).")
    model, tokenizer = build_random_model(
        device=device, hidden_size=hidden_size, n_layers=n_layers,
        n_heads=n_heads, base_path=base_path)
    return model, tokenizer, True


def build_parser():
    p = argparse.ArgumentParser(description="Generate valid peptides with a chosen sampling strategy.")
    p.add_argument("--ckpt_path", type=str, default=None,
                   help="Path to TD3B checkpoint. If missing/omitted, a random-init model is used.")
    p.add_argument("--base_path", type=str, default=ROOT_DIR, help="Repo root (for tokenizer files).")
    p.add_argument("--length", type=int, default=200, help="Target sequence length (tokens).")
    p.add_argument("--num_samples", type=int, default=64, help="Number of sequences to generate.")
    p.add_argument("--strategy", type=str, default="nucleus_remask",
                   choices=available_strategies(), help="Sampling strategy.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_path", type=str, default=None,
                   help="CSV path to save VALID sequences (default: results/valid_<strategy>_len<L>.csv).")
    # strategy knobs (None -> use the strategy preset default)
    p.add_argument("--num_steps", type=int, default=128, help="Base reverse-diffusion steps.")
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=None, help="<1 sharpens logits (low_temp).")
    p.add_argument("--top_p", type=float, default=None, help="Nucleus mass in (0,1].")
    p.add_argument("--top_k", type=int, default=None, help="Top-k tokens per position.")
    p.add_argument("--steps_per_token", type=float, default=None,
                   help="more_steps: num_steps = max(num_steps, round(steps_per_token*length)).")
    p.add_argument("--remask_rounds", type=int, default=None, help="Self-correction rounds.")
    p.add_argument("--remask_frac", type=float, default=None, help="Fraction of lowest-conf tokens to remask.")
    p.add_argument("--remask_steps", type=int, default=None, help="Re-denoise steps per remask round.")
    p.add_argument("--best_of_n", type=int, default=None, help="Oversample N per slot, keep first valid.")
    # random-fallback architecture (ignored when a real checkpoint loads)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    return p


def main():
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if (args.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")

    model, tokenizer, is_random = _load_model(
        args.ckpt_path, device, args.base_path, args.hidden_size, args.n_layers, args.n_heads)
    analyzer = PeptideAnalyzer()

    logger.info("Generating %d sequences of length %d with strategy=%s on %s",
                args.num_samples, args.length, args.strategy, device)

    tokens, sequences, valid_mask, stats = generate(
        model, tokenizer, analyzer,
        batch_size=args.num_samples, length=args.length, strategy=args.strategy,
        num_steps=args.num_steps, eps=args.eps,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        steps_per_token=args.steps_per_token,
        remask_rounds=args.remask_rounds, remask_frac=args.remask_frac,
        remask_steps=args.remask_steps, best_of_n=args.best_of_n,
        verbose=False,
    )

    valid_seqs = [s for s, v in zip(sequences, valid_mask) if v]

    save_path = args.save_path or os.path.join(
        args.base_path, "results", f"valid_{args.strategy}_len{args.length}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "sequence", "n_chars"])
        for i, s in enumerate(valid_seqs):
            w.writerow([i, s, len(s)])

    print("\n" + "=" * 66)
    print(f"  strategy         : {stats['strategy']}")
    print(f"  length           : {stats['length']}")
    print(f"  num_samples      : {stats['batch_size']}")
    print(f"  num_steps        : {stats['num_steps']}"
          f"   (temp={stats['temperature']}, top_p={stats['top_p']}, top_k={stats['top_k']})")
    print(f"  remask           : rounds={stats['remask_rounds']} frac={stats['remask_frac']} "
          f"steps={stats['remask_steps']}   best_of_n={stats['best_of_n']}")
    if len(stats.get("round_valid_counts", [])) > 1:
        print(f"  valid per round  : {stats['round_valid_counts']}  (round 0 = before remask)")
    print(f"  VALID YIELD      : {stats['valid_count']}/{stats['batch_size']} "
          f"= {stats['valid_rate']:.1%}")
    print(f"  wall time        : {stats['wall_time_s']}s")
    print(f"  saved valid seqs : {save_path}  ({len(valid_seqs)} rows)")
    if is_random:
        print("  NOTE: RANDOM-init model -- yields are meaningless; rerun with --ckpt_path "
              "<real.ckpt> for real numbers.")
    print("=" * 66)


if __name__ == "__main__":
    main()
