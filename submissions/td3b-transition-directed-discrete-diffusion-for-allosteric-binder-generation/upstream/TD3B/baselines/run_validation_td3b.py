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
from training.finetune_utils import load_tokenizer, create_reward_function
from finetune_multi_target import TargetDataset
from training.distributed_utils import setup_distributed, cleanup_distributed, is_main_process
from scoring.functions.binding import MultiTargetBindingAffinity, TargetSpecificBindingAffinity
from td3b.direction_oracle import DirectionalOracle
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
    if getattr(cli, "num_pool", None) is not None:
        merged["num_pool"] = cli.num_pool
    if cli.seq_length is not None:
        merged["seq_length"] = cli.seq_length
    if cli.total_num_steps is not None:
        merged["total_num_steps"] = cli.total_num_steps
    if cli.sampling_eps is not None:
        merged["sampling_eps"] = cli.sampling_eps
    if cli.seed is not None:
        merged["seed"] = cli.seed

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
        args.save_path = os.path.join(base_tr2d2_path, "results", "validation_runs")

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


def _sample_sequences(
    model: Diffusion,
    batch_size: int,
    seq_length: int,
    total_num_steps: int,
    sampling_eps: float,
) -> torch.Tensor:
    model.backbone.eval()
    model.noise.eval()

    x_rollout = model.sample_prior(batch_size, seq_length).to(model.device, dtype=torch.long)

    timesteps = torch.linspace(1, sampling_eps, total_num_steps + 1, device=model.device)
    dt = torch.tensor((1 - sampling_eps) / total_num_steps, device=model.device)

    for i in range(total_num_steps):
        t = timesteps[i] * torch.ones(x_rollout.shape[0], 1, device=model.device)
        _, x_next = model.single_reverse_step(x_rollout, t=t, dt=dt)
        x_rollout = x_next.to(model.device)

    if (x_rollout == model.mask_index).any().item():
        _, x_next = model.single_noise_removal(x_rollout, t=t, dt=dt)
        x_rollout = x_next.to(model.device)

    return x_rollout


def _score_sequences(reward_model, sequences: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not sequences:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty

    try:
        result = reward_model(sequences)
        if isinstance(result, tuple):
            total_rewards, info = result
            affinity = np.asarray(info.get("affinities", total_rewards), dtype=np.float32)
            confidence = np.asarray(info.get("confidences", np.ones_like(affinity)), dtype=np.float32)
            directions = np.asarray(info.get("directions", np.zeros_like(affinity)), dtype=np.float32)
        else:
            total_rewards = np.asarray(result, dtype=np.float32)
            if total_rewards.ndim > 1:
                affinity = total_rewards[:, 0]
            else:
                affinity = total_rewards
            confidence = np.ones_like(affinity, dtype=np.float32)
            directions = np.zeros_like(affinity, dtype=np.float32)
        return np.asarray(total_rewards, dtype=np.float32), affinity, directions, confidence
    except Exception:
        total_rewards = np.full(len(sequences), np.nan, dtype=np.float32)
        affinity = np.full(len(sequences), np.nan, dtype=np.float32)
        directions = np.full(len(sequences), np.nan, dtype=np.float32)
        confidence = np.full(len(sequences), np.nan, dtype=np.float32)
        for idx, seq in enumerate(sequences):
            try:
                result = reward_model([seq])
                if isinstance(result, tuple):
                    rewards, info = result
                    total_rewards[idx] = float(np.asarray(rewards)[0])
                    affinity[idx] = float(np.asarray(info.get("affinities", rewards))[0])
                    confidence[idx] = float(np.asarray(info.get("confidences", [np.nan]))[0])
                    directions[idx] = float(np.asarray(info.get("directions", [np.nan]))[0])
                else:
                    reward = np.asarray(result)
                    total_rewards[idx] = float(reward[0]) if reward.size else np.nan
                    affinity[idx] = total_rewards[idx]
            except Exception:
                continue
        return total_rewards, affinity, directions, confidence


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
    return float(np.nanmean(values)) if values.size else float("nan")


def _nanstd(values: np.ndarray) -> float:
    return float(np.nanstd(values)) if values.size else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TD3B validation from a saved checkpoint.")
    parser.add_argument("--ckpt_path", required=True, help="Path to saved checkpoint (.ckpt)")
    parser.add_argument("--val_csv", required=True, help="Validation CSV path")
    parser.add_argument("--device", default="cuda", help="Device string (e.g., cuda:0 or cpu)")
    parser.add_argument("--base_path", default=None, help="Base path for TR2-D2")
    parser.add_argument("--save_path", default=None, help="Output directory for validation CSV")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch number to label outputs")
    parser.add_argument("--val_samples_per_target", type=int, default=None, help="Samples per target")
    parser.add_argument("--num_pool", type=int, default=None,
                        help="Number of candidate sequences to sample before resampling")
    parser.add_argument("--seq_length", type=int, default=None, help="Fallback sequence length")
    parser.add_argument("--total_num_steps", type=int, default=None, help="Diffusion steps")
    parser.add_argument("--sampling_eps", type=float, default=None, help="Sampling epsilon")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument("--no_resample", action="store_true", help="Disable reward-weighted resampling")
    parser.add_argument("--resample_without_replacement", action="store_true",
                        help="Resample without replacement when possible")
    parser.add_argument("--resample_alpha", type=float, default=None,
                        help="Override alpha for resampling weights")
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
    protein_token_cache: Dict[str, torch.Tensor] = {}

    resample_enabled = not cli_args.no_resample
    resample_with_replacement = not cli_args.resample_without_replacement
    resample_alpha = cli_args.resample_alpha if cli_args.resample_alpha is not None else args.alpha

    all_targets = val_dataset.get_all_targets()
    if world_size > 1:
        my_targets = all_targets[rank::world_size]
    else:
        my_targets = all_targets

    records: List[Dict[str, Any]] = []
    resampled_records: List[Dict[str, Any]] = []
    resampled_affinity_pos: List[float] = []
    resampled_affinity_neg: List[float] = []
    resampled_acc_pos: List[float] = []
    resampled_acc_neg: List[float] = []
    resampled_gated_rewards: List[float] = []

    with torch.no_grad():
        for target_seq in my_targets:
            target_protein_tokens = protein_token_cache.get(target_seq)
            if target_protein_tokens is None:
                target_protein_tokens = directional_oracle.encode_protein(target_seq)
                protein_token_cache[target_seq] = target_protein_tokens

            for direction_name, d_star in [("agonist", 1.0), ("antagonist", -1.0)]:
                target_length = val_dataset.get_sequence_length(target_seq, direction_name)
                max_len = 1035
                if target_length > max_len:
                    target_length = max_len

                target_affinity = TargetSpecificBindingAffinity(multi_target_affinity, target_seq)
                reward_model = create_reward_function(
                    affinity_predictor=target_affinity,
                    directional_oracle=directional_oracle,
                    target_direction=d_star,
                    target_protein_tokens=target_protein_tokens,
                    tokenizer=tokenizer,
                    device=device,
                    min_affinity_threshold=args.min_affinity_threshold,
                    use_confidence_weighting=True,
                    temperature=args.sigmoid_temperature,
                )

                pool_size = args.val_samples_per_target
                if getattr(args, "num_pool", None) is not None:
                    pool_size = int(args.num_pool)
                if pool_size < args.val_samples_per_target:
                    print(
                        f"[warn] num_pool ({pool_size}) < val_samples_per_target "
                        f"({args.val_samples_per_target}); using val_samples_per_target."
                    )
                    pool_size = args.val_samples_per_target

                x_eval = _sample_sequences(
                    policy_model,
                    batch_size=pool_size,
                    seq_length=target_length,
                    total_num_steps=args.total_num_steps,
                    sampling_eps=args.sampling_eps,
                )

                sequences = tokenizer.batch_decode(x_eval)
                valid_mask = np.array([analyzer.is_peptide(seq) for seq in sequences], dtype=bool)
                valid_fraction = float(valid_mask.mean()) if valid_mask.size else 0.0

                gated_rewards, affinities, directions, confidences = _score_sequences(reward_model, sequences)
                direction_accuracy = _compute_direction_accuracy(directions, d_star)
                consistency = d_star * (directions - 0.5)
                success_rate = direction_accuracy * valid_fraction

                if resample_enabled:
                    finite_rewards = np.isfinite(gated_rewards)
                    if np.any(finite_rewards):
                        rewards_t = torch.as_tensor(gated_rewards[finite_rewards], device=device)
                        alpha = max(float(resample_alpha), 1e-6)
                        weights = torch.softmax(rewards_t / alpha, dim=0)
                        if resample_with_replacement:
                            num_samples = args.val_samples_per_target
                            idx = torch.multinomial(weights, num_samples=num_samples, replacement=True)
                        else:
                            num_samples = min(args.val_samples_per_target, int(finite_rewards.sum()))
                            idx = torch.multinomial(weights, num_samples=num_samples, replacement=False)

                        valid_idx = np.where(finite_rewards)[0]
                        chosen = valid_idx[idx.detach().cpu().numpy()]
                        if d_star > 0:
                            resampled_affinity_pos.extend(affinities[chosen].tolist())
                            resampled_acc_pos.extend(direction_accuracy[chosen].tolist())
                        else:
                            resampled_affinity_neg.extend(affinities[chosen].tolist())
                            resampled_acc_neg.extend(direction_accuracy[chosen].tolist())
                        resampled_gated_rewards.extend(gated_rewards[chosen].tolist())

                        for picked in chosen.tolist():
                            resampled_records.append({
                                "target": target_seq[:20],
                                "sequence": sequences[picked],
                                "target_direction": d_star,
                                "is_valid": bool(valid_mask[picked]) if valid_mask.size else False,
                                "affinity": float(affinities[picked]) if affinities.size else np.nan,
                                "gated_reward": float(gated_rewards[picked]) if gated_rewards.size else np.nan,
                                "direction_oracle": float(directions[picked]) if directions.size else np.nan,
                                "consistency_reward": float(consistency[picked]) if consistency.size else np.nan,
                                "direction_accuracy": float(direction_accuracy[picked]) if direction_accuracy.size else np.nan,
                                "success_rate": float(success_rate[picked]) if success_rate.size else np.nan,
                            })

                for idx, seq in enumerate(sequences):
                    records.append({
                        "target": target_seq[:20],
                        "sequence": seq,
                        "target_direction": d_star,
                        "is_valid": bool(valid_mask[idx]) if valid_mask.size else False,
                        "affinity": float(affinities[idx]) if affinities.size else np.nan,
                        "gated_reward": float(gated_rewards[idx]) if gated_rewards.size else np.nan,
                        "direction_oracle": float(directions[idx]) if directions.size else np.nan,
                        "consistency_reward": float(consistency[idx]) if consistency.size else np.nan,
                        "direction_accuracy": float(direction_accuracy[idx]) if direction_accuracy.size else np.nan,
                        "success_rate": float(success_rate[idx]) if success_rate.size else np.nan,
                    })

    if world_size > 1:
        gathered: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, records)
        if is_main_process():
            all_records = [item for sub in gathered for item in sub]
        else:
            all_records = []
    else:
        all_records = records

    if world_size > 1:
        gathered_resampled_records: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_resampled_records, resampled_records)
        if is_main_process():
            all_resampled_records = [item for sub in gathered_resampled_records for item in sub]
        else:
            all_resampled_records = []
    else:
        all_resampled_records = resampled_records

    if world_size > 1:
        resampled_payload = {
            "aff_pos": resampled_affinity_pos,
            "aff_neg": resampled_affinity_neg,
            "acc_pos": resampled_acc_pos,
            "acc_neg": resampled_acc_neg,
            "gated": resampled_gated_rewards,
        }
        gathered_resampled = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_resampled, resampled_payload)
        if is_main_process():
            resampled_affinity_pos = []
            resampled_affinity_neg = []
            resampled_acc_pos = []
            resampled_acc_neg = []
            resampled_gated_rewards = []
            for payload in gathered_resampled:
                resampled_affinity_pos.extend(payload.get("aff_pos", []))
                resampled_affinity_neg.extend(payload.get("aff_neg", []))
                resampled_acc_pos.extend(payload.get("acc_pos", []))
                resampled_acc_neg.extend(payload.get("acc_neg", []))
                resampled_gated_rewards.extend(payload.get("gated", []))

    if is_main_process():
        df = pd.DataFrame(all_records)
        output_path = os.path.join(args.save_path, f"validation_epoch_{cli_args.epoch}.csv")
        df.to_csv(output_path, index=False)
        print(f"Validation sequences saved to {output_path}")

        if resample_enabled:
            if all_resampled_records:
                resampled_df = pd.DataFrame(all_resampled_records)
                resampled_path = os.path.join(args.save_path, f"validation_epoch_{cli_args.epoch}_resampled.csv")
                resampled_df.to_csv(resampled_path, index=False)
                print(f"Resampled sequences saved to {resampled_path}")
            else:
                print("Resampling enabled but no finite rewards were available to select.")

        if resample_enabled and resampled_gated_rewards:
            aff_mean_pos = _nanmean(np.asarray(resampled_affinity_pos, dtype=np.float32))
            aff_std_pos = _nanstd(np.asarray(resampled_affinity_pos, dtype=np.float32))
            acc_mean_pos = _nanmean(np.asarray(resampled_acc_pos, dtype=np.float32))
            acc_std_pos = _nanstd(np.asarray(resampled_acc_pos, dtype=np.float32))

            aff_mean_neg = _nanmean(np.asarray(resampled_affinity_neg, dtype=np.float32))
            aff_std_neg = _nanstd(np.asarray(resampled_affinity_neg, dtype=np.float32))
            acc_mean_neg = _nanmean(np.asarray(resampled_acc_neg, dtype=np.float32))
            acc_std_neg = _nanstd(np.asarray(resampled_acc_neg, dtype=np.float32))

            gated = np.asarray(resampled_gated_rewards, dtype=np.float32)
            gated_mean = _nanmean(gated)
            gated_std = _nanstd(gated)
        else:
            def _stats_for_direction(d_star: float) -> Tuple[float, float, float, float]:
                subset = df[df["target_direction"] == d_star]
                affinity = subset["affinity"].to_numpy(dtype=np.float32)
                direction_acc = subset["direction_accuracy"].to_numpy(dtype=np.float32)
                return _nanmean(affinity), _nanstd(affinity), _nanmean(direction_acc), _nanstd(direction_acc)

            aff_mean_pos, aff_std_pos, acc_mean_pos, acc_std_pos = _stats_for_direction(1.0)
            aff_mean_neg, aff_std_neg, acc_mean_neg, acc_std_neg = _stats_for_direction(-1.0)
            gated = df["gated_reward"].to_numpy(dtype=np.float32)
            gated_mean = _nanmean(gated)
            gated_std = _nanstd(gated)

        print("Validation summary")
        print(f"  Affinity (d*=1): {aff_mean_pos:.4f} ± {aff_std_pos:.4f}")
        print(f"  Affinity (d*=-1): {aff_mean_neg:.4f} ± {aff_std_neg:.4f}")
        print(f"  Direction Accuracy (d*=1): {acc_mean_pos:.4f} ± {acc_std_pos:.4f}")
        print(f"  Direction Accuracy (d*=-1): {acc_mean_neg:.4f} ± {acc_std_neg:.4f}")
        print(f"  Gated Reward (overall): {gated_mean:.4f} ± {gated_std:.4f}")

    if world_size > 1:
        cleanup_distributed()


if __name__ == "__main__":
    main()

# Running command:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29501 run_validation_td3b.py --ckpt_path To Be Added --val_csv To Be Added --device cuda:0 --save_path To Be Added --epoch 99 --val_samples_per_target 8 --seed 42 --resample_alpha 0.1
