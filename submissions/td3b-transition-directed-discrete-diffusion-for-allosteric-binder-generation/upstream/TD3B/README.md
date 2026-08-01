<h1 align="center">
  TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation
</h1>

<div align="center">
 <a href="https://arxiv.org/abs/2605.09810"><img src="https://img.shields.io/badge/Arxiv-2605.09810-red?style=for-the-badge&logo=Arxiv" alt="arXiv"/></a>

</div>

![Screenshot 2026-05-10 at 6.52.29 PM](https://cdn-uploads.huggingface.co/production/uploads/64cd5b3f0494187a9e8b7c69/38-6PQ83pPraF7KAs3g3E.png)

TD3B is a sequence-based generative framework that designs peptide binders with specified agonist or antagonist behavior. It combines a Direction Oracle, a soft binding-affinity gate, and amortized fine-tuning of a pre-trained discrete diffusion model (MDLM).

> **Development release.** This repository (`ChatterjeeLab/TD3B-dev`) carries the full code. The heavy artifacts — trained checkpoints, training/test data, and generated binders — are distributed as a single archive on Google Drive (see [Data and Checkpoints](#data-and-checkpoints)). For the clean code-only release see [`ChatterjeeLab/TD3B`](https://huggingface.co/ChatterjeeLab/TD3B).

## Installation

```bash
conda env create -f env.yml
conda activate td3b
pip install -e .
```

## Demo

An interactive inference demo is provided in [`notebooks/TD3B_Inference_Demo.ipynb`](notebooks/TD3B_Inference_Demo.ipynb) (configured for a Colab T4 GPU).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

To run it: download the notebook from this repository, open [Google Colab](https://colab.research.google.com/), upload it via **File → Upload notebook**, and select a GPU runtime (**Runtime → Change runtime type → T4 GPU**).

## Data and Checkpoints

The checkpoints, datasets, and generated binders are bundled in a single archive on Google Drive:

**Download:** [`td3b_dev_artifacts.zip`](https://drive.google.com/file/d/1EU0-pcF-UoEb0OEMuO4qhodZ5RjMU4n9/view?usp=sharing) (~3.4 GB)

Unzip it at the repository root to restore the full layout:

```bash
unzip td3b_dev_artifacts.zip -d .
```

```
TD3B-dev/
├── checkpoints/
│   ├── pretrained.ckpt              # Pre-trained MDLM weights (1.4 GB)
│   ├── td3b.ckpt                    # Fine-tuned TD3B model (231 MB)
│   └── direction_oracle.pt          # Direction Oracle weights (2.85 GB)
├── scoring/functions/classifiers/
│   ├── binding-affinity.pt          # (from archive)
│   ├── permeability-xgboost.json    # (from archive)
│   ├── hemolysis-xgboost.json       # (in repo)
│   ├── nonfouling-xgboost.json      # (in repo)
│   └── solubility-xgboost.json      # (in repo)
├── data/
│   ├── train.csv                    # Training set (target-binder pairs)
│   ├── test.csv                     # Test set
│   └── td3b_data_new.csv            # Full labeled target/ligand pairs (agonist/antagonist)
└── generated_binders/
    ├── agonist/                     # 1106 generated agonist binders (.pdb + .trb), 249 targets
    └── antagonist.tar.gz            # Generated antagonist binders
```

## Code Structure

```
TD3B/
├── inference.py                 # Generate binders (main inference entry point)
├── finetune_on_target.py        # Finetune on your own target(s) + generate (Function A)
├── generate_valid.py            # Validity-boosting sampling CLI (Function B)
├── sampling_strategies.py       # Validity-boosting sampler library (Function B)
├── finetune_multi_target.py     # Multi-target TD3B training
├── launch_multi_target.sh       # Training launcher script
├── models/
│   ├── diffusion.py             # MDLM backbone (TR2-D2)
│   ├── roformer.py              # RoFormer wrapper
│   └── noise_schedule.py        # Noise schedules
├── training/
│   ├── finetune_utils.py        # Training utilities
│   └── distributed_utils.py     # Distributed training helpers
├── mcts/
│   └── peptide_mcts.py          # MCTS tree search
├── td3b/
│   ├── direction_oracle.py      # Direction Oracle (f_φ)
│   ├── td3b_scoring.py          # Gated reward R = g_ψ · σ(d*·(f_φ−0.5)/τ)
│   ├── td3b_losses.py           # L_WDCE + λ·L_ctr + β·L_KL
│   ├── td3b_mcts.py             # TD3B-extended MCTS
│   ├── td3b_finetune.py         # Training loop
│   └── data_utils.py            # Data loading utilities
├── scoring/                     # Affinity predictor (g_ψ) and property classifiers
├── baselines/                   # CG, SMC, TDS, PepTune, Unguided baselines
├── tokenizer/                   # SMILES tokenizer (vocab + splits)
├── configs/                     # Model and training configs
└── utils/                       # Misc utilities
```

## Inference

Generate agonist/antagonist binders for target proteins:

```bash
python inference.py \
    --ckpt_path checkpoints/td3b.ckpt \
    --val_csv data/test.csv \
    --save_path results/ \
    --seed 42 \
    --num_pool 32 \
    --val_samples_per_target 8 \
    --resample_alpha 0.1
```

This generates 32 candidates per (target, direction), scores them with the Direction Oracle and affinity predictor, applies Algorithm 2 weighted resampling, and saves only valid peptide samples.

> **Generation length matters.** The length (in SMILES tokens) comes from `--seq_length` if given, else it is
> derived from each row's reference binder (correctly tokenized). This released checkpoint reliably produces
> **valid** SMILES only at **short lengths (≲100 tokens)**; long binders (e.g. 25–30-residue peptides ≈ 150–210
> tokens) yield few or no valid samples regardless of step count. For non-empty output on such targets, pass a
> shorter `--seq_length` and a validity-boosting sampler (see below), e.g.:
> ```bash
> python inference.py --ckpt_path checkpoints/td3b.ckpt --val_csv data/test.csv \
>     --seq_length 50 --sampler best_of_n --best_of_n 6 --num_pool 32 --val_samples_per_target 4
> ```
> On a real run this yields valid, oracle-scored binders in both directions (antagonist direction accuracy ≈ 1.0).

Output: `results/td3b_results_seed42.csv` with columns: target, target_uid, sequence, direction_name, target_direction, is_valid, affinity, gated_reward, direction_oracle, direction_accuracy.

## Finetune on your own target(s)

`finetune_on_target.py` takes protein target(s) you supply, finetunes the pretrained TD3B policy on **only those target(s)**, then generates directional (agonist/antagonist) binders for them. It reuses the existing machinery: the finetune half subprocess-invokes `finetune_multi_target.py` (with `K` set to the number of targets), and the generation half runs in-process with the same reward, oracle, and Algorithm 2 resampling as `inference.py`.

```bash
python finetune_on_target.py \
    --target_seq MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ \
    --target_seq GSHMKELVLALYDYQEKSPREVTMKKGDILTLL \
    --direction both \
    --validity_reward on \
    --num_epochs 20 \
    --num_pool 32 \
    --gen_samples_per_target 8 \
    --device cuda:0 --seed 42
```

Provide targets by repeating `--target_seq` and/or via `--targets_csv` (a CSV with a `Target_Sequence` column; optional `Ligand_Sequence`/`label` rows seed the per-direction length prior):

```bash
python finetune_on_target.py --targets_csv my_targets.csv --direction agonist --binder_length 20
```

Key flags:
- `--target_seq SEQ` (repeatable) and/or `--targets_csv` — provide at least one target.
- `--direction {agonist,antagonist,both}` (default `both`) — which binders to **generate**. Finetuning always searches both directions.
- `--validity_reward {on,off}` (default `on`) — validity gate applied to **both** halves. `on` keeps the `is_peptide` filter (invalid MCTS children are zero-rewarded during finetuning; only valid peptides are eligible for resampling during generation). `off` drops the gate on both halves (invalid samples are retained, each row keeps `is_valid`); the reward itself is always affinity × direction — validity is only ever a filter, never part of the reward formula. `--finetune_validity_hook {on,off}` overrides just the finetune-side gate.
- `--binder_length` (default 20) — placeholder binder length (residues) used to seed the generation-length prior for any target with no known binder.
- `--skip_finetune` (generate directly from `--td3b_checkpoint`) / `--skip_generate` (finetune only).
- Training knobs: `--num_epochs` (20), `--num_iter` (10) / `--num_children` (16) (MCTS), `--learning_rate` (3e-4), `--seq_length` (200). Generation knobs: `--num_pool` (32), `--gen_samples_per_target` (8), `--resample_alpha` (0.1), `--total_num_steps` (128).
- Paths default to the repo root; override `--pretrained_checkpoint`, `--direction_oracle_ckpt`, `--output_dir`, `--device` (`auto`/`cpu`/`cuda:N`), `--seed` as needed.

Output: `results/finetune_on_target/binders_<direction>_validity-<on|off>_seed<seed>.csv` with columns target, sequence, direction_name, target_direction, is_valid, affinity, gated_reward, direction_oracle, direction_accuracy, gen_length. The finetuned checkpoint is written under `results/<run_name>_<timestamp>/` (used automatically for the generation half).

## Higher-validity sampling for long binders

As the target length grows, the fraction of decoded SMILES that pass the RDKit peptide check drops. `generate_valid.py` (backed by `sampling_strategies.py`) applies **sampling-time** strategies — no retraining — to raise the valid-peptide yield, especially at large length, and reports the valid fraction. The strategies only change token selection (temperature / top-k / top-p) and add a remask self-correction loop and best-of-N rejection on top of the model's existing diffusion primitives.

```bash
python generate_valid.py \
    --ckpt_path checkpoints/td3b.ckpt \
    --length 400 --num_samples 64 \
    --strategy nucleus_remask \
    --device cuda:0 --seed 42 \
    --save_path results/valid_len400.csv
```

Key flags:
- `--strategy` (default `nucleus_remask`) — one of `baseline`, `more_steps`, `top_p` (a.k.a. `nucleus`), `top_k`, `low_temp`, `remask`, `best_of_n`, `nucleus_remask`. On random-init benchmarks the self-correcting strategies (`remask`, `best_of_n`, `nucleus_remask`) give the largest relative validity gains at length ≥ 200; `nucleus_remask` is the recommended default.
- `--length` (default 200, in tokens) and `--num_samples` (default 64).
- `--ckpt_path` — TD3B checkpoint. If omitted or missing, a **random-init** model is used (exercises the sampling machinery only; yields are meaningless — pass a real checkpoint for real numbers).
- Per-strategy overrides (else the preset default is used): `--top_p`, `--top_k`, `--temperature`, `--steps_per_token`, `--remask_rounds`, `--remask_frac`, `--remask_steps`, `--best_of_n`, `--num_steps`.

Output: prints the valid yield (valid / `num_samples`) with per-round counts, and writes the valid sequences to `--save_path` (default `results/valid_<strategy>_len<L>.csv`) with columns idx, sequence, n_chars.

## Training

### Multi-target TD3B

1. Edit `launch_multi_target.sh` only if needed — `BASE_PATH` auto-detects the repo root (this script's own directory), and the checkpoint, data, and oracle paths derive from it:

```bash
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # auto-detects the repo root
PRETRAINED_CHECKPOINT="${BASE_PATH}/checkpoints/pretrained.ckpt"
TRAIN_CSV="${BASE_PATH}/data/train.csv"
ORACLE_CKPT="${BASE_PATH}/checkpoints/direction_oracle.pt"
```

`finetune_multi_target.py --base_path` also defaults to the repo root, so you only need to override these when your checkpoints or data live elsewhere.

2. Launch training:

```bash
bash launch_multi_target.sh
```

Key hyperparameters (in `launch_multi_target.sh`):
- `CONTRASTIVE_WEIGHT=0.1` — λ for L_ctr
- `KL_BETA=0.1` — β for L_KL
- `SIGMOID_TEMPERATURE=0.1` — τ for gated reward
- `NUM_ITER=20` — MCTS iterations per round
- `NUM_CHILDREN=16` — Children per MCTS expansion

### Baselines

Run baseline methods (CG, SMC, TDS, PepTune, Unguided):

```bash
cd baselines/
bash run.sh --baseline cg --device cuda:0
bash run.sh --baseline smc --device cuda:0
bash run.sh --baseline tds --device cuda:0
```

## Citation

```bibtex
@inproceedings{
cao2026tdb,
title={{TD}3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation},
author={Hanqun Cao and Aastha Pal and Sophia Tang and Yinuo Zhang and Jingjie Zhang and Pheng-Ann Heng and Pranam Chatterjee},
booktitle={Learning Meaningful Representations of Life (LMRL) Workshop at ICLR 2026},
year={2026},
url={https://openreview.net/forum?id=uMUicLylVp}
}
```
