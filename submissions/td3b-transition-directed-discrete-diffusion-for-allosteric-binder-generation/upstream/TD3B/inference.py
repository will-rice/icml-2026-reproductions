#!/usr/bin/env python3
"""
TD3B Inference Script
Generate directional binders for target proteins using a finetuned TD3B model.

Usage:
    python inference.py \
        --ckpt_path checkpoints/td3b.ckpt \
        --val_csv data/test.csv \
        --save_path results/ \
        --seed 42
"""
import argparse
import os
import sys
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.diffusion import Diffusion
from configs.finetune_config import (
    DiffusionConfig, RoFormerConfig, NoiseConfig,
    TrainingConfig, SamplingConfig, EvalConfig, OptimConfig, MCTSConfig,
)
from training.finetune_utils import load_tokenizer
from td3b.direction_oracle import DirectionalOracle
from td3b.td3b_scoring import create_td3b_reward_function
from scoring.functions.binding import MultiTargetBindingAffinity, TargetSpecificBindingAffinity
from td3b.data_utils import peptide_seq_to_smiles, smiles_token_length
from utils.app import PeptideAnalyzer
import sampling_strategies

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    seq_length=200,
    sampling_eps=1e-3,
    total_num_steps=128,
    hidden_dim=768,
    num_layers=8,
    num_heads=8,
    alpha=0.1,
    min_affinity_threshold=0.0,
    sigmoid_temperature=0.1,
    num_pool=32,
    val_samples_per_target=8,
)


def load_model(ckpt_path: str, device: torch.device):
    """Load finetuned TD3B model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    config = ckpt.get("config") or {}

    tokenizer = load_tokenizer(ROOT_DIR)

    cfg = DiffusionConfig(
        roformer=RoFormerConfig(
            hidden_size=config.get("hidden_dim", 768),
            n_layers=config.get("num_layers", 8),
            n_heads=config.get("num_heads", 8),
        ),
        noise=NoiseConfig(),
        training=TrainingConfig(sampling_eps=1e-3),
        sampling=SamplingConfig(steps=128, sampling_eps=1e-3),
        eval_cfg=EvalConfig(),
        optim=OptimConfig(lr=3e-4),
        mcts=MCTSConfig(),
    )

    model = Diffusion(config=cfg, tokenizer=tokenizer, device=device).to(device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    # A correct TD3B checkpoint round-trips to 0 missing backbone keys; strict=False otherwise
    # silently leaves the denoiser at random init and generates garbage with no error.
    backbone_missing = [k for k in incompatible.missing_keys if k.startswith("backbone.")]
    total_backbone = sum(1 for k in model.state_dict() if k.startswith("backbone."))
    if backbone_missing:
        msg = (f"{len(backbone_missing)}/{total_backbone} backbone weights did not load from "
               f"{ckpt_path} (e.g. {backbone_missing[:3]}) — the generator would run on "
               f"partially-random weights.")
        if len(backbone_missing) == total_backbone:
            raise RuntimeError(msg + " The backbone loaded NO weights; refusing to run — "
                               "check the checkpoint format/key namespace.")
        logger.warning(msg)
    if incompatible.unexpected_keys:
        logger.warning("Checkpoint had %d unexpected keys (e.g. %s)",
                       len(incompatible.unexpected_keys), incompatible.unexpected_keys[:3])
    model.eval()
    model.tokenizer = tokenizer
    return model, tokenizer


def sample_sequences(model, batch_size: int, seq_length: int, num_steps: int, eps: float = 1e-5):
    """Sample sequences from the diffusion model."""
    x = model.sample_prior(batch_size, seq_length).to(model.device, dtype=torch.long)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = torch.tensor((1 - eps) / num_steps, device=model.device)

    for i in range(num_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=model.device)
        _, x = model.single_reverse_step(x, t=t, dt=dt)
        x = x.to(model.device)

    # Remove remaining masks
    mask_pos = (x == model.mask_index)
    if mask_pos.any():
        t = timesteps[-2] * torch.ones(x.shape[0], 1, device=model.device)
        _, x = model.single_noise_removal(x, t=t, dt=dt)
        x = x.to(model.device)

    return x


def score_sequences(reward_model, sequences: List[str]):
    """Score sequences with the TD3B reward function."""
    result = reward_model(sequences)
    if isinstance(result, tuple):
        rewards, info = result
        return (
            np.asarray(rewards),
            np.asarray(info.get("affinities", rewards)),
            np.asarray(info.get("directions", np.zeros_like(rewards))),
            np.asarray(info.get("confidences", np.ones_like(rewards))),
        )
    rewards = np.asarray(result)
    return rewards, rewards, np.zeros_like(rewards), np.ones_like(rewards)


def main():
    parser = argparse.ArgumentParser(description="TD3B Inference")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to TD3B checkpoint")
    parser.add_argument("--val_csv", type=str, required=True, help="CSV with Target_Sequence, Ligand_Sequence, label columns")
    parser.add_argument("--save_path", type=str, default="results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_pool", type=int, default=32, help="Pool size for candidate generation")
    parser.add_argument("--val_samples_per_target", type=int, default=8, help="Samples to keep per target-direction")
    parser.add_argument("--resample_alpha", type=float, default=0.1, help="Temperature for weighted resampling")
    parser.add_argument("--direction_oracle_ckpt", type=str, default=None)
    parser.add_argument("--direction_oracle_tr2d2_checkpoint", type=str, default=None)
    # ─── Opt-in validity-boosting sampler (default "baseline" == original behavior) ───
    parser.add_argument("--sampler", type=str, default="baseline",
                        choices=sampling_strategies.available_strategies(),
                        help="Candidate-pool sampling strategy (default: baseline == original behavior)")
    parser.add_argument("--num_steps", type=int, default=128,
                        help="Reverse-diffusion steps for non-baseline samplers")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Nucleus cumulative mass for top_p/nucleus/nucleus_remask samplers")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k cutoff on clean-token logits for the top_k sampler")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Softmax temperature on clean-token logits (e.g. low_temp)")
    parser.add_argument("--remask_rounds", type=int, default=None,
                        help="Remask self-correction rounds (remask/nucleus_remask samplers)")
    parser.add_argument("--remask_frac", type=float, default=None,
                        help="Fraction of lowest-confidence tokens remasked each round")
    parser.add_argument("--best_of_n", type=int, default=None,
                        help="Best-of-N validity-guided oversampling per slot (best_of_n sampler)")
    parser.add_argument("--seq_length", type=int, default=None,
                        help="Generation length in SMILES tokens (override). If unset, derived from the "
                             "reference binder. NOTE: this checkpoint only yields valid SMILES at short "
                             "lengths (~<100 tokens) — set this (e.g. 50) for non-empty output on long binders.")
    parser.add_argument("--max_seq_length", type=int, default=200,
                        help="Cap on the derived generation length (SMILES tokens)")
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    analyzer = PeptideAnalyzer()

    # Load model
    logger.info(f"Loading model from {args.ckpt_path}")
    model, tokenizer = load_model(args.ckpt_path, device)

    # Load targets
    logger.info(f"Loading targets from {args.val_csv}")
    df = pd.read_csv(args.val_csv)
    targets = []
    for _, row in df.iterrows():
        # Generation length in SMILES *tokens*: explicit --seq_length override, else the
        # reference binder's tokenized length (from a SMILES/Ligand_SMILES column, or derived
        # from Ligand_Sequence), else the default. Capped at --max_seq_length.
        if args.seq_length is not None:
            seq_len = args.seq_length
        else:
            smi = row.get("SMILES") if isinstance(row.get("SMILES"), str) else row.get("Ligand_SMILES")
            if not isinstance(smi, str):
                lig = row.get("Ligand_Sequence")
                smi = peptide_seq_to_smiles(lig) if isinstance(lig, str) else None
            seq_len = smiles_token_length(smi, tokenizer) if isinstance(smi, str) else DEFAULTS["seq_length"]
        targets.append({
            "target_seq": row["Target_Sequence"],
            "target_uid": row.get("Target_UniProt_ID", ""),
            "binder_seq": row.get("Ligand_Sequence", ""),
            "label": row.get("label", ""),
            "seq_length": max(1, min(int(seq_len) if seq_len else DEFAULTS["seq_length"], args.max_seq_length)),
        })

    # Build the shared reward components ONCE. The affinity predictor (ESM2) and the
    # Direction Oracle (ESM2 + TR2-D2) are expensive to load, so we build them a single
    # time and rebind the per-target protein via lightweight wrappers inside the loop.
    logger.info("Building reward functions...")
    oracle_ckpt = args.direction_oracle_ckpt or os.path.join(ROOT_DIR, "checkpoints", "direction_oracle.pt")
    oracle_tr2d2 = args.direction_oracle_tr2d2_checkpoint or os.path.join(ROOT_DIR, "checkpoints", "pretrained.ckpt")
    vocab_path = os.path.join(ROOT_DIR, "tokenizer", "new_vocab.txt")
    splits_path = os.path.join(ROOT_DIR, "tokenizer", "new_splits.txt")

    multi_affinity = MultiTargetBindingAffinity(
        tokenizer=tokenizer,
        base_path=ROOT_DIR,
        device=device,
        emb_model=model.backbone,
    )
    directional_oracle = DirectionalOracle(
        model_ckpt=oracle_ckpt,
        tr2d2_checkpoint=oracle_tr2d2,
        tokenizer_vocab=vocab_path,
        tokenizer_splits=splits_path,
        device=device,
    )
    directional_oracle.eval()

    records = []

    for tidx, target in enumerate(targets):
        # Bind the shared affinity predictor to this target's protein sequence.
        target_affinity = TargetSpecificBindingAffinity(multi_affinity, target["target_seq"])
        for d_star, d_name in [(1.0, "agonist"), (-1.0, "antagonist")]:
            logger.info(f"[{tidx+1}/{len(targets)}] Target {target['target_uid']} direction={d_name}")

            # Create reward function (reuses the preloaded oracle; only re-encodes the protein)
            try:
                reward_model = create_td3b_reward_function(
                    affinity_predictor=target_affinity,
                    directional_oracle=directional_oracle,
                    target_protein_seq=target["target_seq"],
                    target_direction=d_name,
                    peptide_tokenizer=tokenizer,
                    device=device,
                    min_affinity_threshold=DEFAULTS["min_affinity_threshold"],
                    temperature=DEFAULTS["sigmoid_temperature"],
                )
            except Exception as e:
                logger.warning(f"Failed to create reward for {target['target_uid']}: {e}")
                continue

            # Generate pool of candidates
            target_length = target.get("seq_length", 200)
            if args.sampler == "baseline":
                # Original behavior, unchanged (byte-for-byte).
                x_pool = sample_sequences(model, args.num_pool, target_length, 128)
                sequences = tokenizer.batch_decode(x_pool)
                # Check validity
                valid_mask = np.array([analyzer.is_peptide(seq) for seq in sequences])
            else:
                # Opt-in validity-boosting sampler. Reuses its decode + is_peptide validity.
                # None-valued knobs fall back to each strategy's preset inside generate().
                x_pool, sequences, valid_mask, _sampler_stats = sampling_strategies.generate(
                    model, tokenizer, analyzer,
                    batch_size=args.num_pool, length=target_length,
                    strategy=args.sampler,
                    num_steps=args.num_steps, top_p=args.top_p, top_k=args.top_k,
                    temperature=args.temperature, remask_rounds=args.remask_rounds,
                    remask_frac=args.remask_frac, best_of_n=args.best_of_n,
                )

            # Score all
            gated_rewards, affinities, directions, confidences = score_sequences(reward_model, sequences)
            direction_accuracy = ((directions > 0.5).astype(float) if d_star > 0
                                  else (directions < 0.5).astype(float))

            # Weighted resampling (Algorithm 2). Resample only among candidates that are BOTH
            # finite-reward AND valid peptides, and draw WITHOUT replacement so the output holds
            # up to val_samples_per_target DISTINCT binders. (With replacement + the peaked
            # softmax at alpha=0.1, multinomial repeatedly draws the argmax → duplicate rows that
            # inflate the sample count and skew every mean metric; filtering validity only AFTER
            # resampling could also drop every draw and save 0 samples despite valid candidates.)
            eligible = np.isfinite(gated_rewards) & valid_mask.astype(bool)
            if eligible.any():
                rewards_t = torch.as_tensor(gated_rewards[eligible], device=device, dtype=torch.float32)
                alpha = max(args.resample_alpha, 1e-6)
                weights = torch.softmax(rewards_t / alpha, dim=0)
                k = min(args.val_samples_per_target, int(eligible.sum()))
                idx = torch.multinomial(weights, num_samples=k, replacement=False)
                chosen = np.where(eligible)[0][idx.cpu().numpy()]
            else:
                chosen = np.array([], dtype=int)  # no valid finite-reward candidate → save nothing

            # Save resampled samples (already gated to valid + finite above)
            for i in chosen:
                records.append({
                    "target": target["target_seq"][:20],
                    "target_uid": target["target_uid"],
                    "sequence": sequences[i],
                    "target_direction": d_star,
                    "direction_name": d_name,
                    "is_valid": True,
                    "affinity": float(affinities[i]),
                    "gated_reward": float(gated_rewards[i]),
                    "direction_oracle": float(directions[i]),
                    "direction_accuracy": float(direction_accuracy[i]),
                })

    # Save results
    out_df = pd.DataFrame(records)
    out_path = os.path.join(args.save_path, f"td3b_results_seed{args.seed}.csv")
    out_df.to_csv(out_path, index=False)

    # Print summary
    if len(out_df) > 0:
        dp = out_df[out_df["target_direction"] == 1.0]
        dm = out_df[out_df["target_direction"] == -1.0]
        logger.info(f"\n{'='*60}")
        logger.info(f"Results saved to {out_path} ({len(out_df)} valid samples)")
        logger.info(f"  Aff(d*=+1) = {dp['affinity'].mean():.2f}" if len(dp) else "  No agonist samples")
        logger.info(f"  Aff(d*=-1) = {dm['affinity'].mean():.2f}" if len(dm) else "  No antagonist samples")
        logger.info(f"  DA(d*=+1)  = {dp['direction_accuracy'].mean():.3f}" if len(dp) else "")
        logger.info(f"  DA(d*=-1)  = {dm['direction_accuracy'].mean():.3f}" if len(dm) else "")
        logger.info(f"  Gated Reward = {out_df['gated_reward'].mean():.2f}")
        logger.info(f"{'='*60}")
    else:
        logger.warning("No valid samples generated.")
        logger.warning("Hint: this checkpoint only produces valid SMILES at short lengths "
                       "(~<100 tokens). Re-run with a shorter --seq_length (e.g. 50) and/or "
                       "--sampler best_of_n (or nucleus_remask) to raise the valid yield.")


if __name__ == "__main__":
    main()
