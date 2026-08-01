#!/usr/bin/env python3
"""
TD3B — Finetune-on-Target entry point (BUILD FUNCTION A).

User-facing workflow: the user supplies one or more protein *targets*; the system
(1) FINETUNES the pretrained TD3B diffusion policy on ONLY those target(s), then
(2) GENERATES directional (agonist/antagonist) peptide binders for them — with a
toggle to include or exclude the validity reward.

This script does NOT reimplement the training loop. It REUSES the existing
machinery:

  * FINETUNE half  -> runs ``finetune_multi_target.py`` as a subprocess on a
    temporary CSV built from the provided target(s), with K (``--targets_per_mcts``)
    set to the number of provided targets. That script builds
    ``MultiTargetBindingAffinity`` + ``DirectionalOracle``, runs ``TD3B_MCTS`` per
    (target, direction) to search candidate binders, and distills them into the
    diffusion policy via WDCE + KL + contrastive losses.

  * GENERATE half  -> runs IN-PROCESS, reusing ``inference.py``'s
    ``load_model`` / ``sample_sequences`` / ``score_sequences`` and
    ``td3b.td3b_scoring.create_td3b_reward_function`` (gated reward
    R = g_psi * sigma(d*(f_phi - 0.5)/tau)) plus the Algorithm-2 weighted
    resampling. Running this half in-process (rather than shelling out to
    ``inference.py``) is what lets us honor ``--validity_reward off`` at the
    generation/scoring layer — see the VALIDITY TOGGLE section below.

--------------------------------------------------------------------------------
VALIDITY TOGGLE  (``--validity_reward {on,off}``)
--------------------------------------------------------------------------------
Validity never enters the reward *formula* itself (the reward is always
affinity x direction). It enters TD3B only as a FILTER: ``PeptideAnalyzer.is_peptide``
gates MCTS expansion during finetuning and gates the Algorithm-2 resampling during
generation. This flag now controls BOTH halves:

  * ``on``  (default): keep the validity gate on both halves. During finetuning,
    invalid decoded children are gated as zero-reward MCTS nodes; during
    generation only valid peptides are eligible for resampling (identical to
    ``inference.py``).
  * ``off``: drop the validity gate on both halves. FINETUNE side — forwarded as
    ``--validity_reward off`` to ``finetune_multi_target.py``, which disables the
    ``is_peptide`` gate in MCTS expansion (``mcts/peptide_mcts.py``) so invalid
    peptides are kept and scored on the pure affinity x direction reward instead
    of being zero-rewarded. GENERATION side — every finite-reward candidate is
    eligible for resampling and invalid samples are retained in the output (each
    row still records ``is_valid`` for inspection). The reward stays pure
    affinity x direction throughout.

Use ``--finetune_validity_hook {on,off}`` to override the finetune-side gate
independently of the generation-side toggle (default: follow ``--validity_reward``).

--------------------------------------------------------------------------------
Also note (reuse-without-editing limitations):
  * ``finetune_multi_target.py`` ALWAYS searches both agonist and antagonist per
    target in its MCTS phase. The training CSV therefore always seeds a length
    prior for BOTH directions (real binder length if provided, else a placeholder
    of ``--binder_length`` residues). ``--direction`` restricts only what the
    GENERATE half emits, not what finetuning trains on.

Base paths DEFAULT to the repo root (auto-detected from ``__file__``); nothing is
hardcoded to an absolute path. Heavy artifacts (checkpoints / ESM / binding
predictor) are validated up-front with a clear error if missing.
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Repo root — every default path is derived from this, never hardcoded absolute.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger("finetune_on_target")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
DIRECTIONS = {"agonist": 1.0, "antagonist": -1.0}
GDRIVE_HINT = ("These heavy artifacts ship separately (see README 'Data and "
               "Checkpoints' — ~3.4 GB on Google Drive). Unzip them at the repo "
               "root to restore checkpoints/ and scoring/functions/classifiers/.")


# ─────────────────────────────────────────────────────────────────────────────
# Target-set construction
# ─────────────────────────────────────────────────────────────────────────────
def _looks_like_protein(seq: str) -> bool:
    if not isinstance(seq, str):
        return False
    s = seq.strip().upper()
    return len(s) >= 2 and all(c in AA_SET for c in s)


def make_placeholder_binder(binder_length: int, mode: str = "polyG",
                            rng: Optional[np.random.Generator] = None) -> str:
    """Synthesize a placeholder binder purely to seed the per-direction LENGTH
    prior (never scored as a real binder). Poly-glycine by default."""
    n = max(2, int(binder_length))
    if mode == "random":
        rng = rng or np.random.default_rng(0)
        return "".join(rng.choice(list(AA_SET), size=n))
    return "G" * n


def build_target_training_frame(
    target_seqs: List[str],
    targets_csv: Optional[str],
    binder_length: int,
    placeholder_mode: str = "polyG",
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the normalized training frame consumed by ``finetune_multi_target``.

    Columns: ``Target_Sequence``, ``Ligand_Sequence``, ``label``.

    For every unique target we ensure BOTH directions have at least one row so the
    downstream ``TargetDataset`` can compute a per-direction median binder length.
    Real (Ligand_Sequence, label) rows from the CSV are kept as-is; any missing
    direction is seeded with a placeholder binder of ``binder_length`` residues.

    Returns (frame, unique_target_order).
    """
    rng = np.random.default_rng(seed)
    # target_seq -> {'agonist': [binders...], 'antagonist': [...]}
    per_target: Dict[str, Dict[str, List[str]]] = {}
    order: List[str] = []

    def _ensure(t: str) -> Dict[str, List[str]]:
        if t not in per_target:
            per_target[t] = {"agonist": [], "antagonist": []}
            order.append(t)
        return per_target[t]

    # 1) Explicit --target_seq entries (no known binders).
    for t in target_seqs or []:
        t = t.strip()
        if not t:
            continue
        if not _looks_like_protein(t):
            logger.warning("Target sequence does not look like a protein (non-AA "
                           "characters); using it anyway: %.40s...", t)
        _ensure(t)

    # 2) Optional CSV with Target_Sequence (+ optional Ligand_Sequence/label).
    if targets_csv:
        if not os.path.isfile(targets_csv):
            raise FileNotFoundError(f"--targets_csv not found: {targets_csv}")
        df = pd.read_csv(targets_csv)
        if "Target_Sequence" not in df.columns:
            raise ValueError(f"{targets_csv} must contain a 'Target_Sequence' column "
                             f"(found: {list(df.columns)})")
        for _, row in df.iterrows():
            t = str(row["Target_Sequence"]).strip()
            if not t:
                continue
            bucket = _ensure(t)
            binder = row.get("Ligand_Sequence")
            label = str(row.get("label", "")).strip().lower()
            if isinstance(binder, str) and binder.strip() and label in DIRECTIONS:
                bucket[label].append(binder.strip())

    if not order:
        raise ValueError("No targets provided. Pass at least one --target_seq or a "
                         "--targets_csv with a 'Target_Sequence' column.")

    # 3) Materialize rows, seeding placeholders for any empty direction.
    rows = []
    for t in order:
        for direction in ("agonist", "antagonist"):
            binders = per_target[t][direction]
            if not binders:
                binders = [make_placeholder_binder(binder_length, placeholder_mode, rng)]
            for b in binders:
                rows.append({"Target_Sequence": t, "Ligand_Sequence": b, "label": direction})

    frame = pd.DataFrame(rows, columns=["Target_Sequence", "Ligand_Sequence", "label"])
    return frame, order


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / artifact validation
# ─────────────────────────────────────────────────────────────────────────────
def _require_files(pairs: List[Tuple[str, str]], stage: str) -> None:
    """Raise a single clear error listing every missing artifact for ``stage``."""
    missing = [(label, path) for label, path in pairs if not os.path.isfile(path)]
    if missing:
        lines = "\n".join(f"  - {label}: {path}" for label, path in missing)
        raise FileNotFoundError(
            f"Cannot run the {stage} stage — required artifact(s) are missing:\n"
            f"{lines}\n{GDRIVE_HINT}"
        )


def finetune_artifact_checks(args) -> List[Tuple[str, str]]:
    return [
        ("pretrained_checkpoint", args.pretrained_checkpoint),
        ("direction_oracle_ckpt", args.direction_oracle_ckpt),
        ("direction_oracle_tr2d2_checkpoint", args.direction_oracle_tr2d2_checkpoint),
        ("direction_oracle_tokenizer_vocab", args.direction_oracle_tokenizer_vocab),
        ("direction_oracle_tokenizer_splits", args.direction_oracle_tokenizer_splits),
        ("binding_affinity_predictor",
         os.path.join(args.base_path, "scoring", "functions", "classifiers", "binding-affinity.pt")),
    ]


def generate_artifact_checks(args, ckpt_path: str) -> List[Tuple[str, str]]:
    return [
        ("td3b_checkpoint", ckpt_path),
        ("direction_oracle_ckpt", args.direction_oracle_ckpt),
        ("direction_oracle_tr2d2_checkpoint", args.direction_oracle_tr2d2_checkpoint),
        ("binding_affinity_predictor",
         os.path.join(args.base_path, "scoring", "functions", "classifiers", "binding-affinity.pt")),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# FINETUNE half  (subprocess reuse of finetune_multi_target.py)
# ─────────────────────────────────────────────────────────────────────────────
def run_finetune(args, train_csv: str, num_targets: int) -> str:
    """Finetune on the provided target(s) by invoking finetune_multi_target.py.

    Returns the path to the finetuned checkpoint (model_final.ckpt, else the
    newest model_epoch_*.ckpt) discovered in the run's output directory.
    """
    _require_files(finetune_artifact_checks(args), "finetune")

    # Validity gate forwarded to the finetune-side MCTS. Follows --validity_reward
    # unless --finetune_validity_hook explicitly overrides it (decoupling the
    # finetune-side gate from the generation-side toggle).
    finetune_validity = args.finetune_validity_hook or args.validity_reward

    results_root = os.path.join(args.base_path, "results")
    os.makedirs(results_root, exist_ok=True)
    # finetune_multi_target appends a timestamp to run_name -> record pre-existing
    # dirs so we can identify the NEW one it creates.
    before = set(glob.glob(os.path.join(results_root, f"{args.run_name}_*")))

    # K = number of provided targets (train on ONLY these).
    cmd = [
        sys.executable, os.path.join(args.base_path, "finetune_multi_target.py"),
        "--base_path", args.base_path,
        "--train_csv", train_csv,
        "--pretrained_checkpoint", args.pretrained_checkpoint,
        "--run_name", args.run_name,
        "--device", args.device,
        "--targets_per_mcts", str(num_targets),
        "--resample_targets_every", "1",
        "--num_epochs", str(args.num_epochs),
        "--learning_rate", str(args.learning_rate),
        "--train_batch_size", str(args.train_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--resample_every_n_step", str(args.resample_every_n_step),
        "--save_every_n_epochs", str(args.save_every_n_epochs),
        "--reset_every_n_step", str(max(1, args.num_epochs)),  # reset tree once, at epoch 0
        "--num_iter", str(args.num_iter),
        "--num_children", str(args.num_children),
        "--buffer_size", str(args.buffer_size),
        "--validity_reward", finetune_validity,  # forwards the validity gate to finetune-side MCTS
        "--alpha", str(args.alpha),
        "--min_affinity_threshold", str(args.min_affinity_threshold),
        "--sigmoid_temperature", str(args.sigmoid_temperature),
        "--seq_length", str(args.seq_length),
        "--wandb_project", args.wandb_project,
        # oracle wiring (pass our repo-root defaults through explicitly so the
        # subprocess does not fall back to its own legacy default paths)
        "--direction_oracle_ckpt", args.direction_oracle_ckpt,
        "--direction_oracle_tr2d2_checkpoint", args.direction_oracle_tr2d2_checkpoint,
        "--direction_oracle_tokenizer_vocab", args.direction_oracle_tokenizer_vocab,
        "--direction_oracle_tokenizer_splits", args.direction_oracle_tokenizer_splits,
        "--direction_oracle_esm_name", args.direction_oracle_esm_name,
        "--direction_oracle_d_model", str(args.direction_oracle_d_model),
        "--direction_oracle_n_heads", str(args.direction_oracle_n_heads),
        "--direction_oracle_n_self_attn_layers", str(args.direction_oracle_n_self_attn_layers),
        "--direction_oracle_n_bmca_layers", str(args.direction_oracle_n_bmca_layers),
        "--direction_oracle_dropout", str(args.direction_oracle_dropout),
    ]
    if args.direction_oracle_esm_cache_dir:
        cmd += ["--direction_oracle_esm_cache_dir", args.direction_oracle_esm_cache_dir]
    if args.direction_oracle_esm_local_files_only:
        cmd += ["--direction_oracle_esm_local_files_only"]
    if args.grad_clip:
        cmd += ["--grad_clip", "--gradnorm_clip", str(args.gradnorm_clip)]

    env = dict(os.environ)
    env.setdefault("WANDB_MODE", args.wandb_mode)  # default: disabled (offline, no prompt)

    logger.info("Launching finetune subprocess (K=%d target(s)):\n  %s",
                num_targets, " ".join(cmd))
    proc = subprocess.run(cmd, cwd=args.base_path, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"finetune_multi_target.py exited with code {proc.returncode}. "
            f"See the subprocess log above for the underlying cause (e.g. a missing "
            f"checkpoint, OOM, or oracle/ESM load failure)."
        )

    # Locate the checkpoint produced by this run.
    after = set(glob.glob(os.path.join(results_root, f"{args.run_name}_*")))
    new_dirs = sorted(after - before, key=os.path.getmtime)
    search_dirs = new_dirs or sorted(after, key=os.path.getmtime)
    if not search_dirs:
        raise RuntimeError(
            f"Finetuning finished but no results dir matching "
            f"{results_root}/{args.run_name}_* was found; cannot locate the "
            f"finetuned checkpoint.")
    run_dir = search_dirs[-1]
    final_ckpt = os.path.join(run_dir, "model_final.ckpt")
    if os.path.isfile(final_ckpt):
        ckpt = final_ckpt
    else:
        epoch_ckpts = sorted(glob.glob(os.path.join(run_dir, "model_epoch_*.ckpt")),
                             key=os.path.getmtime)
        if not epoch_ckpts:
            raise RuntimeError(
                f"Finetuning finished but no checkpoint (model_final.ckpt or "
                f"model_epoch_*.ckpt) was found in {run_dir}.")
        ckpt = epoch_ckpts[-1]
    logger.info("Finetuned checkpoint: %s", ckpt)
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE half  (in-process reuse of inference.py + td3b scoring)
# ─────────────────────────────────────────────────────────────────────────────
# These three thin seams are what the smoke test monkeypatches to inject a
# random-init tiny model + stub reward (so the generation control flow can run on
# CPU without the 3.4 GB artifacts / ESM).
def build_generation_model(ckpt_path: str, device):
    """Load the finetuned TD3B model. Reuses inference.load_model (which refuses
    to run on an all-random backbone, guarding against silent garbage output)."""
    from inference import load_model
    return load_model(ckpt_path, device)


def build_reward_components(args, device, tokenizer, model):
    """Build the shared (expensive) reward components ONCE: the multi-target
    affinity predictor (ESM2 + binding-affinity.pt) and the Direction Oracle."""
    from scoring.functions.binding import MultiTargetBindingAffinity
    from td3b.direction_oracle import DirectionalOracle

    multi_affinity = MultiTargetBindingAffinity(
        tokenizer=tokenizer, base_path=args.base_path, device=device,
        emb_model=model.backbone,
    )
    directional_oracle = DirectionalOracle(
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
    directional_oracle.eval()
    return multi_affinity, directional_oracle


def make_reward_function(multi_affinity, directional_oracle, target_seq, direction,
                         tokenizer, device, args):
    """Bind the shared reward components to (target, direction). Reuses the exact
    gated reward from td3b.td3b_scoring."""
    from scoring.functions.binding import TargetSpecificBindingAffinity
    from td3b.td3b_scoring import create_td3b_reward_function

    target_affinity = TargetSpecificBindingAffinity(multi_affinity, target_seq)
    return create_td3b_reward_function(
        affinity_predictor=target_affinity,
        directional_oracle=directional_oracle,
        target_protein_seq=target_seq,
        target_direction=direction,
        peptide_tokenizer=tokenizer,
        device=device,
        min_affinity_threshold=args.min_affinity_threshold,
        temperature=args.sigmoid_temperature,
    )


def generate_binders(args, model, tokenizer, multi_affinity, directional_oracle,
                     targets, length_provider, device) -> pd.DataFrame:
    """Sample -> score -> Algorithm-2 resample, for every (target, requested
    direction). Honors the validity toggle at the resampling gate.

    Args:
        targets: ordered list of unique target protein sequences.
        length_provider: fn(target_seq, direction_name) -> int generation length.
    Returns a results DataFrame (also written to disk by the caller).
    """
    import torch
    from inference import sample_sequences, score_sequences
    from utils.app import PeptideAnalyzer

    analyzer = PeptideAnalyzer()
    validity_on = (args.validity_reward == "on")
    directions = (["agonist", "antagonist"] if args.direction == "both" else [args.direction])
    columns = ["target", "target_full", "sequence", "direction_name", "target_direction",
               "is_valid", "validity_reward", "affinity", "gated_reward",
               "direction_oracle", "direction_accuracy", "gen_length"]
    records = []

    for tidx, target_seq in enumerate(targets):
        for d_name in directions:
            d_star = DIRECTIONS[d_name]
            logger.info("[%d/%d] target=%.20s... direction=%s",
                        tidx + 1, len(targets), target_seq, d_name)

            reward_model = make_reward_function(
                multi_affinity, directional_oracle, target_seq, d_name,
                tokenizer, device, args)

            gen_len = int(max(1, length_provider(target_seq, d_name)))
            x_pool = sample_sequences(model, args.num_pool, gen_len, args.total_num_steps)
            sequences = tokenizer.batch_decode(x_pool)

            valid_mask = np.array([analyzer.is_peptide(s) for s in sequences], dtype=bool)
            gated_rewards, affinities, oracle_dirs, _conf = score_sequences(reward_model, sequences)
            direction_accuracy = ((oracle_dirs > 0.5).astype(float) if d_star > 0
                                  else (oracle_dirs < 0.5).astype(float))

            # ── VALIDITY TOGGLE ───────────────────────────────────────────────
            # on : eligible = finite reward AND valid peptide (== inference.py)
            # off: eligible = finite reward only (validity gate skipped; invalid
            #      samples retained; reward stays pure affinity x direction)
            finite = np.isfinite(gated_rewards)
            eligible = (finite & valid_mask) if validity_on else finite

            if eligible.any():
                rewards_t = torch.as_tensor(gated_rewards[eligible], device=device,
                                            dtype=torch.float32)
                alpha = max(args.resample_alpha, 1e-6)
                weights = torch.softmax(rewards_t / alpha, dim=0)
                k = min(args.gen_samples_per_target, int(eligible.sum()))
                idx = torch.multinomial(weights, num_samples=k, replacement=False)
                chosen = np.where(eligible)[0][idx.cpu().numpy()]
            else:
                logger.warning("No eligible candidates for target=%.20s dir=%s "
                               "(validity_reward=%s).", target_seq, d_name, args.validity_reward)
                chosen = np.array([], dtype=int)

            for i in chosen:
                records.append({
                    "target": target_seq[:20],
                    "target_full": target_seq,
                    "sequence": sequences[i],
                    "direction_name": d_name,
                    "target_direction": d_star,
                    "is_valid": bool(valid_mask[i]),
                    "validity_reward": args.validity_reward,
                    "affinity": float(affinities[i]),
                    "gated_reward": float(gated_rewards[i]),
                    "direction_oracle": float(oracle_dirs[i]),
                    "direction_accuracy": float(direction_accuracy[i]),
                    "gen_length": gen_len,
                })

    # Always return the defined columns so an empty result still writes a header
    # row (clearer than a 0-byte file — never silently produce empty output).
    return pd.DataFrame(records, columns=columns)


def run_generation(args, ckpt_path: str, train_csv: str, targets: List[str]) -> str:
    """Orchestrate the generate half and write the results CSV. Returns its path."""
    import torch
    _require_files(generate_artifact_checks(args, ckpt_path), "generate")

    device = torch.device(args.device if (args.device != "auto") else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; using CPU.")
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Loading finetuned model: %s", ckpt_path)
    model, tokenizer = build_generation_model(ckpt_path, device)

    logger.info("Building reward components (affinity predictor + Direction Oracle)...")
    multi_affinity, directional_oracle = build_reward_components(args, device, tokenizer, model)

    # Per-direction generation length prior, reusing finetune_multi_target's
    # TargetDataset (SMILES-token median length) for consistency with training.
    length_provider = build_length_provider(train_csv, tokenizer, args.binder_length)

    df = generate_binders(args, model, tokenizer, multi_affinity, directional_oracle,
                          targets, length_provider, device)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"binders_{args.direction}_validity-{args.validity_reward}_seed{args.seed}.csv")
    df.to_csv(out_path, index=False)

    if len(df):
        logger.info("=" * 60)
        logger.info("Wrote %d binder(s) -> %s", len(df), out_path)
        for d_name in (["agonist", "antagonist"] if args.direction == "both" else [args.direction]):
            sub = df[df["direction_name"] == d_name]
            if len(sub):
                logger.info("  %-10s n=%d  affinity=%.3f  gated=%.3f  dir_acc=%.3f  valid_frac=%.3f",
                            d_name, len(sub), sub["affinity"].mean(), sub["gated_reward"].mean(),
                            sub["direction_accuracy"].mean(), sub["is_valid"].mean())
        logger.info("=" * 60)
    else:
        logger.warning("No binders generated (empty results). Wrote header-only CSV -> %s", out_path)
    return out_path


def build_length_provider(train_csv: str, tokenizer, fallback_binder_length: int):
    """Return fn(target_seq, direction_name) -> generation length, reusing
    finetune_multi_target.TargetDataset so the generated length matches the length
    prior used during training. Falls back to a token-length estimate of a
    placeholder of ``fallback_binder_length`` residues if a target is unknown."""
    from finetune_multi_target import TargetDataset
    from td3b.data_utils import peptide_seq_to_smiles, smiles_token_length

    dataset = TargetDataset(train_csv, tokenizer=tokenizer)
    fallback_len = smiles_token_length(
        peptide_seq_to_smiles(make_placeholder_binder(fallback_binder_length)), tokenizer)

    def provider(target_seq: str, direction_name: str) -> int:
        try:
            return dataset.get_sequence_length(target_seq, direction_name)
        except KeyError:
            return fallback_len

    return provider


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Finetune TD3B on user-provided protein target(s), then "
                    "generate directional binders (with a validity-reward toggle).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    tgt = p.add_argument_group("Targets (provide at least one)")
    tgt.add_argument("--target_seq", action="append", default=None, metavar="SEQ",
                     help="Target protein amino-acid sequence. Repeatable for multiple targets.")
    tgt.add_argument("--targets_csv", type=str, default=None,
                     help="CSV with at least 'Target_Sequence' (optional 'Ligand_Sequence'/'label').")

    ctrl = p.add_argument_group("Control")
    ctrl.add_argument("--direction", choices=["agonist", "antagonist", "both"], default="both",
                      help="Which directional binders to GENERATE (finetuning always trains both).")
    ctrl.add_argument("--validity_reward", choices=["on", "off"], default="on",
                      help="on: keep the is_peptide validity gate during BOTH finetuning "
                           "(MCTS expansion) and generation; off: pure affinity x direction, "
                           "no validity gate in either phase (see module docstring).")
    ctrl.add_argument("--binder_length", type=int, default=20,
                      help="Placeholder binder length (residues) used to seed the length prior "
                           "when a target has no known binder.")
    ctrl.add_argument("--placeholder_mode", choices=["polyG", "random"], default="polyG",
                      help="Placeholder binder composition (length-prior seed only).")

    stages = p.add_argument_group("Stages")
    stages.add_argument("--skip_finetune", action="store_true",
                        help="Skip finetuning and generate from --td3b_checkpoint directly.")
    stages.add_argument("--skip_generate", action="store_true",
                        help="Finetune only; do not generate.")

    train = p.add_argument_group("Training knobs (small sane defaults)")
    train.add_argument("--num_epochs", type=int, default=20)
    train.add_argument("--num_iter", type=int, default=10, help="MCTS iterations per resample.")
    train.add_argument("--num_children", type=int, default=16, help="Children per MCTS expansion.")
    train.add_argument("--learning_rate", type=float, default=3e-4)
    train.add_argument("--train_batch_size", type=int, default=8)
    train.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train.add_argument("--resample_every_n_step", type=int, default=10)
    train.add_argument("--save_every_n_epochs", type=int, default=20)
    train.add_argument("--buffer_size", type=int, default=50)
    train.add_argument("--alpha", type=float, default=0.1, help="Importance-weight temperature.")
    train.add_argument("--min_affinity_threshold", type=float, default=0.0)
    train.add_argument("--sigmoid_temperature", type=float, default=0.1)
    train.add_argument("--seq_length", type=int, default=200, help="Max sequence length.")
    train.add_argument("--total_num_steps", type=int, default=128, help="Diffusion steps for generation.")
    train.add_argument("--grad_clip", action="store_true")
    train.add_argument("--gradnorm_clip", type=float, default=1.0)
    train.add_argument("--finetune_validity_hook", choices=["on", "off"], default=None,
                       help="Optional override for the FINETUNE-side validity gate forwarded to "
                            "finetune_multi_target.py's --validity_reward. Default (unset): follow "
                            "--validity_reward. Set to decouple the finetune-side gate from the "
                            "generation-side toggle.")

    gen = p.add_argument_group("Generation knobs")
    gen.add_argument("--num_pool", type=int, default=32, help="Candidates sampled per (target, direction).")
    gen.add_argument("--gen_samples_per_target", type=int, default=8, help="Binders kept per (target, direction).")
    gen.add_argument("--resample_alpha", type=float, default=0.1, help="Algorithm-2 resampling temperature.")

    paths = p.add_argument_group("Paths (default to repo root, auto-detected)")
    paths.add_argument("--base_path", type=str, default=REPO_ROOT,
                       help="Repo root; source of scoring/, tokenizer/, checkpoints/.")
    paths.add_argument("--pretrained_checkpoint", type=str,
                       default=os.path.join(REPO_ROOT, "checkpoints", "pretrained.ckpt"))
    paths.add_argument("--td3b_checkpoint", type=str, default=None,
                       help="Finetuned checkpoint for --skip_finetune generation. If omitted and "
                            "finetuning ran, the produced checkpoint is used.")
    paths.add_argument("--output_dir", type=str,
                       default=os.path.join(REPO_ROOT, "results", "finetune_on_target"))
    paths.add_argument("--run_name", type=str, default="finetune_on_target")
    paths.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', 'cuda', 'cuda:N'.")
    paths.add_argument("--seed", type=int, default=42)
    paths.add_argument("--wandb_project", type=str, default="TD3B-finetune-on-target")
    paths.add_argument("--wandb_mode", type=str, default="disabled",
                       help="WANDB_MODE for the finetune subprocess (disabled/offline/online).")

    orc = p.add_argument_group("Directional oracle")
    orc.add_argument("--direction_oracle_ckpt", type=str,
                     default=os.path.join(REPO_ROOT, "checkpoints", "direction_oracle.pt"))
    orc.add_argument("--direction_oracle_tr2d2_checkpoint", type=str,
                     default=os.path.join(REPO_ROOT, "checkpoints", "pretrained.ckpt"))
    orc.add_argument("--direction_oracle_tokenizer_vocab", type=str,
                     default=os.path.join(REPO_ROOT, "tokenizer", "new_vocab.txt"))
    orc.add_argument("--direction_oracle_tokenizer_splits", type=str,
                     default=os.path.join(REPO_ROOT, "tokenizer", "new_splits.txt"))
    orc.add_argument("--direction_oracle_esm_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    orc.add_argument("--direction_oracle_esm_cache_dir", type=str, default=None)
    orc.add_argument("--direction_oracle_esm_local_files_only", action="store_true")
    orc.add_argument("--direction_oracle_max_ligand_length", type=int, default=768)
    orc.add_argument("--direction_oracle_max_protein_length", type=int, default=1024)
    orc.add_argument("--direction_oracle_d_model", type=int, default=256)
    orc.add_argument("--direction_oracle_n_heads", type=int, default=4)
    orc.add_argument("--direction_oracle_n_self_attn_layers", type=int, default=1)
    orc.add_argument("--direction_oracle_n_bmca_layers", type=int, default=2)
    orc.add_argument("--direction_oracle_dropout", type=float, default=0.3)

    args = p.parse_args(argv)

    if not args.target_seq and not args.targets_csv:
        p.error("Provide at least one --target_seq or a --targets_csv.")
    if args.skip_finetune and args.skip_generate:
        p.error("--skip_finetune and --skip_generate together leave nothing to do.")
    # Normalize base-path-derived defaults if the user overrode --base_path only.
    return args


def write_temp_training_csv(frame: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="td3b_target_train_", suffix=".csv", dir=output_dir)
    os.close(fd)
    frame.to_csv(path, index=False)
    return path


def main(argv=None):
    args = parse_args(argv)
    logger.info("=" * 80)
    logger.info("TD3B finetune-on-target  |  direction=%s  validity_reward=%s",
                args.direction, args.validity_reward)
    logger.info("=" * 80)

    # 1) Build + normalize the target training set (seeds length priors).
    frame, targets = build_target_training_frame(
        args.target_seq, args.targets_csv, args.binder_length,
        placeholder_mode=args.placeholder_mode, seed=args.seed)
    logger.info("Prepared %d unique target(s); %d training row(s).", len(targets), len(frame))

    train_csv = write_temp_training_csv(frame, args.output_dir)
    logger.info("Temp training CSV: %s", train_csv)

    if args.finetune_validity_hook is not None:
        logger.info("--finetune_validity_hook=%s overrides the finetune-side validity gate "
                    "(generation-side stays validity_reward=%s).",
                    args.finetune_validity_hook, args.validity_reward)

    # 2) FINETUNE half.
    ckpt_path = args.td3b_checkpoint
    if not args.skip_finetune:
        ckpt_path = run_finetune(args, train_csv, num_targets=len(targets))
    elif not ckpt_path:
        raise ValueError("--skip_finetune requires --td3b_checkpoint (nothing to generate from).")

    # 3) GENERATE half.
    if not args.skip_generate:
        out_path = run_generation(args, ckpt_path, train_csv, targets)
        logger.info("Done. Results: %s", out_path)
    else:
        logger.info("Done (finetune only). Checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
