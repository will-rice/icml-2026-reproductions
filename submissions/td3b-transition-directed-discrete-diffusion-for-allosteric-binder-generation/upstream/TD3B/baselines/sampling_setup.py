import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from models.diffusion import Diffusion
from scoring.scoring_functions import ScoringFunctions
from scoring.functions.binding import MultiTargetBindingAffinity
from td3b.direction_oracle import DirectionalOracle, resolve_device
from td3b.data_utils import peptide_seq_to_smiles, smiles_token_length

from baselines.baselines import (
    RewardInputs,
    RewardWrapper,
    classifier_guidance,
    peptune_mctg_sampling,
    sequential_monte_carlo,
    twisted_diffusion_sampler,
    unguided_sampling,
)


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class ProteinTokenizer:
    aa_to_id: Dict[str, int]
    pad_id: int = 0

    @classmethod
    def default(cls) -> "ProteinTokenizer":
        aa_to_id = {aa: idx + 1 for idx, aa in enumerate(AMINO_ACIDS)}
        return cls(aa_to_id=aa_to_id, pad_id=0)

    def encode(self, seq: str) -> torch.Tensor:
        ids = [self.aa_to_id.get(aa, self.pad_id) for aa in seq]
        return torch.tensor([ids], dtype=torch.long)


def load_base_model(
    ckpt_path: str,
    device: str,
    config_name: str = "peptune_config.yaml",
) -> Diffusion:
    GlobalHydra.instance().clear()
    config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    initialize_config_dir(config_dir=config_dir, job_name="load_model")
    cfg = compose(config_name=config_name)
    try:
        model = Diffusion.load_from_checkpoint(
            ckpt_path,
            config=cfg,
            mode="eval",
            device=device,
            map_location=device,
        )
        model.eval()
        return model
    except Exception as exc:
        print(f"[load_base_model] Lightning load failed, falling back to raw state_dict: {exc}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    model = Diffusion(
        config=cfg,
        mode="eval",
        device=device,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_base_model] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[load_base_model] Unexpected keys: {len(unexpected)}")
    model.eval()
    model.to(device)
    return model


def load_reward_models(
    prot_seq: Optional[str],
    device: str,
    base_model: Optional[Diffusion] = None,
    base_path: Optional[str] = None,
    multi_target: bool = False,
    score_func_names: Optional[List[str]] = None,
):
    if multi_target:
        if base_model is None or base_path is None:
            raise ValueError("base_model and base_path are required for multi-target affinity.")
        return MultiTargetBindingAffinity(
            tokenizer=base_model.tokenizer,
            base_path=base_path,
            device=device,
            emb_model=base_model.backbone,
        )
    if score_func_names is None:
        score_func_names = [
            "binding_affinity1",
            "solubility",
            "hemolysis",
            "nonfouling",
            "permeability",
        ]
    if prot_seq is None:
        raise ValueError("prot_seq is required for single-target scoring.")
    return ScoringFunctions(score_func_names, prot_seqs=[prot_seq], device=device)


def load_direction_oracle(args, device: str) -> DirectionalOracle:
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


def run_baseline(
    baseline: str,
    base_model: Diffusion,
    reward_fn: RewardWrapper,
    batch_size: int,
    seq_length: int,
    num_steps: int,
    guidance_scale: float,
    alpha: float,
    guidance_steps: Optional[int],
    mcts_iterations: int,
    num_children: int,
    sample_prob_weight: float,
    invalid_penalty: float,
    pareto_max_size: Optional[int],
) -> Dict[str, torch.Tensor]:
    baseline = baseline.lower()
    if baseline == "cg":
        return classifier_guidance(
            base_model,
            reward_fn,
            batch_size=batch_size,
            seq_length=seq_length,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            guidance_steps=guidance_steps,
        )
    if baseline == "unguided":
        return unguided_sampling(
            base_model,
            batch_size=batch_size,
            seq_length=seq_length,
            num_steps=num_steps,
        )
    if baseline == "smc":
        return sequential_monte_carlo(
            base_model,
            reward_fn,
            batch_size=batch_size,
            seq_length=seq_length,
            num_steps=num_steps,
            alpha=alpha,
        )
    if baseline == "tds":
        return twisted_diffusion_sampler(
            base_model,
            reward_fn,
            batch_size=batch_size,
            seq_length=seq_length,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            alpha=alpha,
            guidance_steps=guidance_steps,
        )
    if baseline == "peptune":
        return peptune_mctg_sampling(
            base_model,
            reward_fn,
            batch_size=batch_size,
            seq_length=seq_length,
            num_steps=num_steps,
            mcts_iterations=mcts_iterations,
            num_children=num_children,
            alpha=alpha,
            sample_prob_weight=sample_prob_weight,
            invalid_penalty=invalid_penalty,
            pareto_max_size=pareto_max_size,
        )
    raise ValueError(f"Unknown baseline: {baseline}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--baseline", type=str, default="cg", choices=["cg", "smc", "tds", "unguided", "peptune"])
    parser.add_argument("--prot_seq", type=str, default=None)
    parser.add_argument("--targets_csv", type=str, default=None)
    parser.add_argument("--d_star", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=200)
    parser.add_argument("--binder_seq", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--reward_alpha", type=float, default=None)
    parser.add_argument("--mcts_iterations", type=int, default=20)
    parser.add_argument("--num_children", type=int, default=24)
    parser.add_argument("--sample_prob_weight", type=float, default=0.1)
    parser.add_argument("--invalid_penalty", type=float, default=1.0)
    parser.add_argument("--pareto_max_size", type=int, default=None)
    parser.add_argument("--guidance_steps", type=int, default=None)
    parser.add_argument("--fast_direction", action="store_true", default=False)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--direction_oracle_ckpt", type=str, default=None)
    parser.add_argument("--direction_oracle_tr2d2_checkpoint", type=str, default=None)
    parser.add_argument("--direction_oracle_tokenizer_vocab", type=str, default=None)
    parser.add_argument("--direction_oracle_tokenizer_splits", type=str, default=None)
    parser.add_argument("--direction_oracle_esm_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--direction_oracle_esm_cache_dir", type=str, default=None)
    parser.add_argument("--direction_oracle_esm_local_files_only", action="store_true", default=False)
    parser.add_argument("--direction_oracle_max_ligand_length", type=int, default=768)
    parser.add_argument("--direction_oracle_max_protein_length", type=int, default=1024)
    parser.add_argument("--direction_oracle_d_model", type=int, default=256)
    parser.add_argument("--direction_oracle_n_heads", type=int, default=4)
    parser.add_argument("--direction_oracle_n_self_attn_layers", type=int, default=1)
    parser.add_argument("--direction_oracle_n_bmca_layers", type=int, default=2)
    parser.add_argument("--direction_oracle_dropout", type=float, default=0.3)
    args = parser.parse_args()

    rank_env = os.environ.get("LOCAL_RANK")
    world_env = os.environ.get("WORLD_SIZE")
    if rank_env is not None or world_env is not None:
        rank = int(rank_env or 0)
        world_size = int(world_env or 1)
    else:
        rank = int(args.shard_id) if args.shard_id is not None else 0
        world_size = int(args.num_shards) if args.num_shards is not None else 1
    if world_size < 1:
        world_size = 1
    if world_size > 1 and str(args.device).lower() in {"cuda", "cuda:0", "auto"}:
        args.device = f"cuda:{rank}"

    resolved_device = resolve_device(args.device)
    args.device = str(resolved_device)

    tr2d2_root = ROOT_DIR
    if args.direction_oracle_ckpt is None:
        args.direction_oracle_ckpt = os.path.join(
            tr2d2_root, "direction_oracle.pt"
        )
    if args.direction_oracle_tr2d2_checkpoint is None:
        args.direction_oracle_tr2d2_checkpoint = os.path.join(
            tr2d2_root, "pretrained", "peptune-pretrained.ckpt"
        )
    if args.direction_oracle_tokenizer_vocab is None:
        args.direction_oracle_tokenizer_vocab = os.path.join(
            tr2d2_root, "tokenizer", "new_vocab.txt"
        )
    if args.direction_oracle_tokenizer_splits is None:
        args.direction_oracle_tokenizer_splits = os.path.join(
            tr2d2_root, "tokenizer", "new_splits.txt"
        )

    if args.targets_csv is None and args.prot_seq is None:
        raise ValueError("--prot_seq is required when --targets_csv is not provided.")

    base_model = load_base_model(args.ckpt_path, args.device)
    base_path = ROOT_DIR
    multi_target = args.targets_csv is not None
    scoring_fn = load_reward_models(
        args.prot_seq if not multi_target else None,
        args.device,
        base_model=base_model,
        base_path=base_path,
        multi_target=multi_target,
    )
    direction_oracle = load_direction_oracle(args, args.device)
    reward_alpha = args.reward_alpha if args.reward_alpha is not None else args.alpha

    if args.targets_csv:
        import pandas as pd

        df = pd.read_csv(args.targets_csv)
        if "Target_Sequence" not in df.columns:
            raise ValueError("targets_csv must contain a 'Target_Sequence' column.")
        if "Ligand_Sequence" not in df.columns:
            raise ValueError("targets_csv must contain a 'Ligand_Sequence' column.")

        targets = []
        for row_idx, row in df.iterrows():
            target_seq = str(row["Target_Sequence"]) if pd.notna(row["Target_Sequence"]) else None
            if not target_seq:
                continue
            binder_seq = row["Ligand_Sequence"]
            if pd.isna(binder_seq):
                binder_seq = None
            else:
                binder_seq = str(binder_seq)
                if binder_seq.strip() == "":
                    binder_seq = None
            targets.append(
                {
                    "target_seq": target_seq,
                    "binder_seq": binder_seq,
                    "row_index": int(row_idx),
                }
            )
    else:
        targets = [{"target_seq": args.prot_seq, "binder_seq": args.binder_seq, "row_index": 0}]

    if world_size > 1:
        targets = [item for idx, item in enumerate(targets) if idx % world_size == rank]
        print(f"[shard] rank {rank}/{world_size}: {len(targets)} targets")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    from utils.app import PeptideAnalyzer

    analyzer = PeptideAnalyzer()
    all_rows = []
    batch_rows = []
    metrics_rows = []
    def resolve_seq_length(binder_seq: Optional[str]) -> int:
        if not binder_seq:
            return args.seq_length
        try:
            smiles = peptide_seq_to_smiles(binder_seq)
            if not smiles:
                return args.seq_length
            if base_model.tokenizer is None:
                return len(smiles)
            return smiles_token_length(smiles, base_model.tokenizer)
        except Exception as exc:
            print(f"Warning: failed to derive seq_length from binder_seq; using {args.seq_length}. Error: {exc}")
            return args.seq_length

    for target_idx, target_info in enumerate(targets):
        target_seq = target_info["target_seq"]
        binder_seq = target_info.get("binder_seq")
        row_index = target_info.get("row_index", target_idx)
        seq_length = resolve_seq_length(binder_seq)
        protein_tokens = direction_oracle.encode_protein(target_seq)
        for direction_name, d_star in [("agonist", 1.0), ("antagonist", -1.0)]:

            reward_inputs = RewardInputs(
                protein_tokens=protein_tokens,
                d_star=d_star,
                protein_seq=target_seq,
            )
            reward_fn = RewardWrapper(
                scoring_fn=scoring_fn,
                direction_oracle=direction_oracle,
                base_model=base_model,
                tokenizer=base_model.tokenizer,
                reward_inputs=reward_inputs,
                device=torch.device(args.device),
                fast_direction=args.fast_direction,
                reward_alpha=reward_alpha,
            )

            num_batches = 1 if multi_target else args.num_batches
            for batch_idx in range(num_batches):
                start = time.perf_counter()
                result = run_baseline(
                    args.baseline,
                    base_model,
                    reward_fn,
                    batch_size=args.batch_size,
                    seq_length=seq_length,
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    alpha=args.alpha,
                    guidance_steps=args.guidance_steps,
                    mcts_iterations=args.mcts_iterations,
                    num_children=args.num_children,
                    sample_prob_weight=args.sample_prob_weight,
                    invalid_penalty=args.invalid_penalty,
                    pareto_max_size=args.pareto_max_size,
                )
                elapsed = time.perf_counter() - start

                scores = reward_fn.evaluate_tokens(
                    result["tokens"],
                    torch.ones_like(result["tokens"], device=result["tokens"].device),
                )
                sequences = scores["sequences"]
                affinity = scores["affinity"].detach().cpu().numpy()
                direction = scores["direction"].detach().cpu().numpy()
                gated_reward = scores["gated_reward"].detach().cpu().numpy()
                valid_mask = np.array([analyzer.is_peptide(seq) for seq in sequences], dtype=np.float32)
                valid_fraction = float(valid_mask.mean()) if len(valid_mask) else 0.0
                consistency = d_star * (direction - 0.5)
                if d_star > 0:
                    direction_correct = (direction >= 0.5).astype(np.float32)
                else:
                    direction_correct = (direction < 0.5).astype(np.float32)
                success = direction_correct * valid_mask
                direction_mean = float(np.mean(direction))
                direction_std = float(np.std(direction))
                affinity_mean = float(np.mean(affinity))
                affinity_std = float(np.std(affinity))
                consistency_mean = float(np.mean(consistency))
                consistency_std = float(np.std(consistency))
                gated_reward_mean = float(np.mean(gated_reward))
                gated_reward_std = float(np.std(gated_reward))
                direction_acc_mean = float(np.mean(direction_correct))
                direction_acc_std = float(np.std(direction_correct))
                success_rate_mean = float(np.mean(success))
                success_rate_std = float(np.std(success))
                batch_metrics = {
                    "direction_mean": direction_mean,
                    "direction_std": direction_std,
                    "affinity_mean": affinity_mean,
                    "affinity_std": affinity_std,
                    "consistency_mean": consistency_mean,
                    "consistency_std": consistency_std,
                    "gated_reward_mean": gated_reward_mean,
                    "gated_reward_std": gated_reward_std,
                    "direction_accuracy_mean": direction_acc_mean,
                    "direction_accuracy_std": direction_acc_std,
                    "valid_fraction": valid_fraction,
                    "success_rate_mean": success_rate_mean,
                    "success_rate_std": success_rate_std,
                }

                for i, seq in enumerate(sequences):
                    all_rows.append(
                        {
                            "rank": rank,
                            "sequence": seq,
                            "affinity": float(affinity[i]),
                            "direction": float(direction[i]),
                            "d_star": float(d_star),
                            "direction_name": direction_name,
                            "target_seq": target_seq,
                            "target_index": target_idx,
                            "row_index": row_index,
                            "binder_seq": binder_seq,
                            "seq_length": seq_length,
                            "gated_reward": float(gated_reward[i]),
                            "consistency_reward": float(consistency[i]),
                            "direction_accuracy": float(direction_correct[i]),
                            "valid": float(valid_mask[i]),
                            "success": float(success[i]),
                            "batch_index": batch_idx,
                            "batch_time_sec": elapsed,
                            **batch_metrics,
                        }
                    )
                batch_rows.append(
                    {
                        "rank": rank,
                        "batch_index": batch_idx,
                        "batch_time_sec": elapsed,
                        "target_index": target_idx,
                        "row_index": row_index,
                        "binder_seq": binder_seq,
                        "seq_length": seq_length,
                        "direction_name": direction_name,
                    }
                )
                metrics_rows.append(
                    {
                        "rank": rank,
                        "target_index": target_idx,
                        "target_seq": target_seq,
                        "row_index": row_index,
                        "binder_seq": binder_seq,
                        "seq_length": seq_length,
                        "direction_name": direction_name,
                        "d_star": float(d_star),
                        "batch_index": batch_idx,
                        "num_samples": len(sequences),
                        **batch_metrics,
                    }
                )
                print(
                    f"Target {target_idx} dir {direction_name}: "
                    f"generated {len(sequences)} sequences in {elapsed:.3f}s"
                )

    import pandas as pd

    if world_size > 1:
        output_csv = os.path.join(output_dir, f"{args.baseline}_samples_rank{rank}.csv")
        batch_csv = os.path.join(output_dir, f"batch_times_rank{rank}.csv")
        metrics_csv = os.path.join(output_dir, f"{args.baseline}_metrics_rank{rank}.csv")
    else:
        output_csv = os.path.join(output_dir, f"{args.baseline}_samples.csv")
        batch_csv = os.path.join(output_dir, "batch_times.csv")
        metrics_csv = os.path.join(output_dir, f"{args.baseline}_metrics.csv")
    pd.DataFrame(all_rows).to_csv(output_csv, index=False)
    pd.DataFrame(batch_rows).to_csv(batch_csv, index=False)
    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)

    print(f"Saved samples to {output_csv}")


if __name__ == "__main__":
    main()
