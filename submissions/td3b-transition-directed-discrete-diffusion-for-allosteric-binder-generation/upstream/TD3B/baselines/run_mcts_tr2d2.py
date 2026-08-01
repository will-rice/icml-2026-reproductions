#!/usr/bin/env python3
import argparse
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.diffusion import Diffusion
from configs.finetune_config import (
    DiffusionConfig,
    RoFormerConfig,
    NoiseConfig,
    TrainingConfig,
    SamplingConfig,
    EvalConfig,
    OptimConfig,
    MCTSConfig,
)
from training.finetune_utils import load_tokenizer
from training.distributed_utils import setup_distributed, cleanup_distributed, is_main_process
from scoring.functions.binding import MultiTargetBindingAffinity, TargetSpecificBindingAffinity
from td3b.direction_oracle import DirectionalOracle
from finetune_multi_target_tr2d2_ddp import TR2D2GatedReward, TargetDataset, create_tr2d2_mcts
from utils.app import PeptideAnalyzer


def _load_checkpoint(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")
    return ckpt


def _extract_state_and_config(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    config = ckpt.get("config") or {}
    return {"state_dict": state_dict, "config": config}


def _build_args(cfg: Dict[str, Any], cli: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "base_path": ROOT_DIR,
        "seq_length": 200,
        "sampling_eps": 1e-3,
        "total_num_steps": 128,
        "alpha": 0.1,
        "hidden_dim": 768,
        "num_layers": 8,
        "num_heads": 8,
        "min_affinity_threshold": 0.0,
        "sigmoid_temperature": 0.1,
        "val_samples_per_target": 8,
        "direction_oracle_esm_name": "facebook/esm2_t33_650M_UR50D",
        "direction_oracle_esm_cache_dir": None,
        "direction_oracle_esm_local_files_only": False,
        "direction_oracle_max_ligand_length": 768,
        "direction_oracle_max_protein_length": 1024,
        "direction_oracle_d_model": 256,
        "direction_oracle_n_heads": 4,
        "direction_oracle_n_self_attn_layers": 1,
        "direction_oracle_n_bmca_layers": 2,
        "direction_oracle_dropout": 0.3,
        "num_iter": 20,
        "num_children": 24,
        "buffer_size": 32,
        "exploration": 1.0,
    }

    merged = dict(defaults)
    merged.update(cfg or {})

    if cli.base_path is not None:
        merged["base_path"] = cli.base_path
    if cli.val_csv is not None:
        merged["val_csv"] = cli.val_csv
    if cli.save_path is not None:
        merged["save_path"] = cli.save_path
    if cli.device is not None:
        merged["device"] = cli.device
    if cli.val_samples_per_target is not None:
        merged["val_samples_per_target"] = cli.val_samples_per_target
    if cli.seq_length is not None:
        merged["seq_length"] = cli.seq_length
    if cli.total_num_steps is not None:
        merged["total_num_steps"] = cli.total_num_steps
    if cli.sampling_eps is not None:
        merged["sampling_eps"] = cli.sampling_eps
    if cli.alpha is not None:
        merged["alpha"] = cli.alpha
    if cli.num_iter is not None:
        merged["num_iter"] = cli.num_iter
    if cli.num_children is not None:
        merged["num_children"] = cli.num_children
    if cli.buffer_size is not None:
        merged["buffer_size"] = cli.buffer_size
    if cli.exploration is not None:
        merged["exploration"] = cli.exploration
    if cli.max_sequence_length is not None:
        merged["max_sequence_length"] = cli.max_sequence_length

    args = SimpleNamespace(**merged)

    base_tr2d2_path = os.path.join(args.base_path, "tr2d2-pep")
    if not getattr(args, "direction_oracle_ckpt", None):
        args.direction_oracle_ckpt = os.path.join(base_tr2d2_path, "direction_oracle.pt")
    if not getattr(args, "direction_oracle_tr2d2_checkpoint", None):
        args.direction_oracle_tr2d2_checkpoint = os.path.join(
            base_tr2d2_path, "pretrained", "peptune-pretrained.ckpt"
        )
    if not getattr(args, "direction_oracle_tokenizer_vocab", None):
        args.direction_oracle_tokenizer_vocab = os.path.join(
            base_tr2d2_path, "tokenizer", "new_vocab.txt"
        )
    if not getattr(args, "direction_oracle_tokenizer_splits", None):
        args.direction_oracle_tokenizer_splits = os.path.join(
            base_tr2d2_path, "tokenizer", "new_splits.txt"
        )

    if not getattr(args, "save_path", None):
        args.save_path = os.path.join(base_tr2d2_path, "baselines", "outputs_mcts_tr2d2")
    os.makedirs(args.save_path, exist_ok=True)
    return args


def _build_model(args: argparse.Namespace, state_dict: Dict[str, Any], device: torch.device) -> Diffusion:
    config = DiffusionConfig(
        roformer=RoFormerConfig(
            hidden_size=args.hidden_dim,
            n_layers=args.num_layers,
            n_heads=args.num_heads,
        ),
        noise=NoiseConfig(),
        training=TrainingConfig(sampling_eps=args.sampling_eps),
        sampling=SamplingConfig(
            steps=args.total_num_steps,
            sampling_eps=args.sampling_eps,
        ),
        eval_cfg=EvalConfig(),
        optim=OptimConfig(lr=getattr(args, "learning_rate", 3e-4)),
        mcts=MCTSConfig(),
    )

    tokenizer = load_tokenizer(args.base_path)
    model = Diffusion(
        config=config,
        tokenizer=tokenizer,
        device=device,
    ).to(device)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"[load] Missing keys: {len(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        print(f"[load] Unexpected keys: {len(load_result.unexpected_keys)}")
    model.eval()
    return model


def _build_oracle(args: argparse.Namespace, device: torch.device) -> DirectionalOracle:
    oracle = DirectionalOracle(
        model_ckpt=args.direction_oracle_ckpt,
        tr2d2_checkpoint=args.direction_oracle_tr2d2_checkpoint,
        tokenizer_vocab=args.direction_oracle_tokenizer_vocab,
        tokenizer_splits=args.direction_oracle_tokenizer_splits,
        esm_name=args.direction_oracle_esm_name,
        d_model=args.direction_oracle_d_model,
        n_heads=args.direction_oracle_n_heads,
        n_self_attn_layers=args.direction_oracle_n_self_attn_layers,
        n_bmca_layers=args.direction_oracle_n_bmca_layers,
        dropout=args.direction_oracle_dropout,
        max_ligand_length=args.direction_oracle_max_ligand_length,
        max_protein_length=args.direction_oracle_max_protein_length,
        device=device,
        esm_cache_dir=args.direction_oracle_esm_cache_dir,
        esm_local_files_only=args.direction_oracle_esm_local_files_only,
    )
    oracle.eval()
    return oracle


def _compute_direction_accuracy(directions: np.ndarray, d_star: float) -> np.ndarray:
    if directions.size == 0:
        return directions
    acc = np.full(directions.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(directions)
    if not valid.any():
        return acc
    if d_star > 0:
        acc[valid] = (directions[valid] >= 0.5).astype(np.float32)
    else:
        acc[valid] = (directions[valid] < 0.5).astype(np.float32)
    return acc


def _nanmean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else 0.0


def _nanstd(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    return float(np.std(finite)) if finite.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="MCTS-based TR2-D2 evaluation.")
    parser.add_argument("--ckpt_path", required=True, help="Path to finetuned checkpoint (.ckpt)")
    parser.add_argument("--val_csv", required=True, help="Validation CSV path")
    parser.add_argument("--device", default="cuda", help="Device string (e.g., cuda:0 or cpu)")
    parser.add_argument("--base_path", default=None, help="Base path for TR2-D2")
    parser.add_argument("--save_path", default=None, help="Output directory for evaluation CSV")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch number to label outputs")
    parser.add_argument("--val_samples_per_target", type=int, default=None, help="Samples per target (unused by MCTS)")
    parser.add_argument("--seq_length", type=int, default=None, help="Fallback sequence length")
    parser.add_argument("--total_num_steps", type=int, default=None, help="Diffusion steps")
    parser.add_argument("--sampling_eps", type=float, default=None, help="Sampling epsilon")
    parser.add_argument("--alpha", type=float, default=None, help="MCTS alpha temperature")
    parser.add_argument("--num_iter", type=int, default=None, help="MCTS iterations")
    parser.add_argument("--num_children", type=int, default=None, help="MCTS children per expand")
    parser.add_argument("--buffer_size", type=int, default=None, help="MCTS buffer size")
    parser.add_argument("--exploration", type=float, default=None, help="MCTS exploration constant")
    parser.add_argument("--max_sequence_length", type=int, default=1035)
    parser.add_argument("--max_attempts", type=int, default=3, help="Max MCTS attempts to reach target count")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    cli_args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        setup_distributed(rank, world_size)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cli_args.device)

    if cli_args.seed is not None:
        torch.manual_seed(cli_args.seed + rank)
        np.random.seed(cli_args.seed + rank)

    ckpt = _load_checkpoint(cli_args.ckpt_path, device)
    payload = _extract_state_and_config(ckpt)
    args = _build_args(payload["config"], cli_args)

    tokenizer = load_tokenizer(args.base_path)
    val_dataset = TargetDataset(args.val_csv, tokenizer=tokenizer)

    policy_model = _build_model(args, payload["state_dict"], device)

    multi_target_affinity = MultiTargetBindingAffinity(
        tokenizer=tokenizer,
        base_path=args.base_path,
        device=device,
        emb_model=policy_model.backbone,
    )

    directional_oracle = _build_oracle(args, device)
    analyzer = PeptideAnalyzer()

    val_targets = val_dataset.get_all_targets()
    if world_size > 1:
        my_targets = val_targets[rank::world_size]
    else:
        my_targets = val_targets

    records: List[Dict[str, Any]] = []
    protein_token_cache: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for target_seq in my_targets:
            target_tokens = protein_token_cache.get(target_seq)
            if target_tokens is None:
                target_tokens = directional_oracle.encode_protein(target_seq)
                protein_token_cache[target_seq] = target_tokens

            for direction_name, d_star in [("agonist", 1.0), ("antagonist", -1.0)]:
                target_length = val_dataset.get_sequence_length(target_seq, direction_name)
                if target_length > args.max_sequence_length:
                    target_length = args.max_sequence_length

                original_seq_length = args.seq_length
                args.seq_length = int(target_length)

                target_affinity = TargetSpecificBindingAffinity(multi_target_affinity, target_seq)
                reward_model = TR2D2GatedReward(
                    affinity_predictor=target_affinity,
                    directional_oracle=directional_oracle,
                    target_direction=d_star,
                    target_protein_tokens=target_tokens,
                    tokenizer=tokenizer,
                    device=device,
                    min_affinity_threshold=args.min_affinity_threshold,
                    temperature=args.sigmoid_temperature,
                )

                mcts = create_tr2d2_mcts(
                    args=args,
                    policy_model=policy_model,
                    reward_function=reward_model,
                    buffer_size=args.buffer_size,
                )

                target_count = int(args.val_samples_per_target)
                collected_sequences: List[str] = []
                attempt_valid_fractions: List[float] = []

                for attempt in range(max(cli_args.max_attempts, 1)):
                    try:
                        _, _, _, _, sequences = mcts.forward(resetTree=True)
                    except Exception as exc:
                        print(f"[mcts] failed for target={target_seq[:12]} dir={direction_name}: {exc}")
                        sequences = []

                    attempt_valid = float(np.mean(mcts.valid_fraction_log)) if getattr(mcts, "valid_fraction_log", None) else 0.0
                    attempt_valid_fractions.append(attempt_valid)

                    if sequences:
                        collected_sequences.extend(sequences)

                    if len(collected_sequences) >= target_count:
                        break

                args.seq_length = original_seq_length

                valid_fraction = _nanmean(np.asarray(attempt_valid_fractions, dtype=np.float32))

                if not collected_sequences:
                    records.append(
                        {
                            "target": target_seq[:20],
                            "sequence": "",
                            "target_direction": d_star,
                            "is_valid": False,
                            "valid_fraction": valid_fraction,
                            "affinity": np.nan,
                            "gated_reward": np.nan,
                            "direction_oracle": np.nan,
                            "consistency_reward": np.nan,
                            "direction_accuracy": np.nan,
                            "success_rate": np.nan,
                        }
                    )
                    continue

                if len(collected_sequences) > target_count:
                    collected_sequences = collected_sequences[:target_count]

                gated_rewards, affinities, confidences, directions = reward_model.reward_fn.compute_gated_reward(collected_sequences)
                direction_accuracy = _compute_direction_accuracy(directions, d_star)
                consistency = d_star * (directions - 0.5)
                success_rate = direction_accuracy * valid_fraction

                valid_mask = np.array([analyzer.is_peptide(seq) for seq in collected_sequences], dtype=bool)

                for idx, seq in enumerate(collected_sequences):
                    records.append(
                        {
                            "target": target_seq[:20],
                            "sequence": seq,
                            "target_direction": d_star,
                            "is_valid": bool(valid_mask[idx]) if valid_mask.size else False,
                            "valid_fraction": valid_fraction,
                            "affinity": float(affinities[idx]) if len(affinities) else np.nan,
                            "gated_reward": float(gated_rewards[idx]) if len(gated_rewards) else np.nan,
                            "direction_oracle": float(directions[idx]) if len(directions) else np.nan,
                            "consistency_reward": float(consistency[idx]) if len(consistency) else np.nan,
                            "direction_accuracy": float(direction_accuracy[idx]) if len(direction_accuracy) else np.nan,
                            "success_rate": float(success_rate[idx]) if len(success_rate) else np.nan,
                        }
                    )

    if world_size > 1:
        gathered: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, records)
        if is_main_process():
            records = [item for sub in gathered for item in sub]
        else:
            cleanup_distributed()
            return

    if is_main_process():
        df = pd.DataFrame(records)
        output_path = os.path.join(args.save_path, f"mcts_validation_epoch_{cli_args.epoch}.csv")
        df.to_csv(output_path, index=False)
        print(f"MCTS validation sequences saved to {output_path}")

        affinities = df["affinity"].to_numpy(dtype=np.float32)
        gated_rewards = df["gated_reward"].to_numpy(dtype=np.float32)
        directions = df["direction_oracle"].to_numpy(dtype=np.float32)
        target_directions = df["target_direction"].to_numpy(dtype=np.float32)
        direction_correct = df["direction_accuracy"].to_numpy(dtype=np.float32)
        valid_fractions = df["valid_fraction"].to_numpy(dtype=np.float32)

        pos_mask = target_directions == 1.0
        neg_mask = target_directions == -1.0

        print("MCTS validation summary")
        print(f"  Affinity (d*=1): {_nanmean(affinities[pos_mask]):.4f} ± {_nanstd(affinities[pos_mask]):.4f}")
        print(f"  Affinity (d*=-1): {_nanmean(affinities[neg_mask]):.4f} ± {_nanstd(affinities[neg_mask]):.4f}")
        print(f"  Direction Accuracy (d*=1): {_nanmean(direction_correct[pos_mask]):.4f} ± {_nanstd(direction_correct[pos_mask]):.4f}")
        print(f"  Direction Accuracy (d*=-1): {_nanmean(direction_correct[neg_mask]):.4f} ± {_nanstd(direction_correct[neg_mask]):.4f}")
        print(f"  Gated Reward (overall): {_nanmean(gated_rewards):.4f} ± {_nanstd(gated_rewards):.4f}")
        print(f"  Valid Fraction: {_nanmean(valid_fractions):.4f} ± {_nanstd(valid_fractions):.4f}")

    if world_size > 1:
        cleanup_distributed()


if __name__ == "__main__":
    main()
