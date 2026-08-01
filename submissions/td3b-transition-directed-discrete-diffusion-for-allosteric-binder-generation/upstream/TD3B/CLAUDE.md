# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TD3B (Transition-Directed Discrete Diffusion for Allosteric Binder Generation) is a sequence-based
generative framework that designs peptide binders with a **specified direction** — agonist or
antagonist — for a target protein. It extends **TR2-D2** (a masked discrete-diffusion peptide
generator: MDLM backbone + MCTS amortized finetuning) with three additions:

1. a **Direction Oracle** `f_φ` that predicts agonist vs. antagonist behavior,
2. a **soft binding-affinity gate** `g_ψ`, and
3. a **gated reward** `R = g_ψ · σ(d*·(f_φ − 0.5)/τ)` that steers generation toward direction `d* ∈ {+1 agonist, −1 antagonist}`.

Finetuning distills MCTS-discovered high-reward sequences into the diffusion policy via a total loss
`L = L_WDCE + λ·L_ctr + β·L_KL` (weighted denoising CE + directional contrastive + KL-to-reference).

Paper: arXiv:2605.09810 (LMRL Workshop, ICLR 2026).

## Repo layout: dev vs. OSS, and where the artifacts are

- This repo (`ChatterjeeLab/TD3B-dev`) is the **full/dev code**. The clean code-only public release is
  `ChatterjeeLab/TD3B` on HuggingFace. Keep changes runnable against that OSS release.
- **No checkpoints or data are in git.** Trained checkpoints, train/test CSVs, and generated binders
  ship as a single ~3.4 GB archive `td3b_dev_artifacts.zip` on Google Drive (link in README). Unzip at
  repo root to populate `checkpoints/`, `data/`, `scoring/functions/classifiers/`, `generated_binders/`.
- `.gitattributes` routes all weight/data extensions (`*.ckpt *.pt *.pth *.npy *.csv? *.zip ...`) through
  Git LFS. Only three XGBoost classifier JSONs (`hemolysis`, `nonfouling`, `solubility`) are in-repo;
  `binding-affinity.pt` and `permeability-xgboost.json` come from the archive.

## Environment & commands

```bash
conda env create -f env.yml   # creates env "td3b" (python 3.10, pytorch-cuda 12.1)
conda activate td3b
pip install -e .              # installs the `td3b` package (setup.py)
```

Core deps: PyTorch + Lightning 2.5.5, HuggingFace `transformers` 4.56.2, `fair-esm` 2.0.0 (ESM2),
`rdkit`, `SmilesPE`, `xgboost`, `wandb`, `hydra-core`. There is **no test suite, no linter config, and
no Makefile** — do not look for `pytest`/`tox`/`ruff`. Verify changes by actually running inference.

**Inference** (primary OSS entry point — generate directional binders):
```bash
python inference.py \
    --ckpt_path checkpoints/td3b.ckpt \
    --val_csv data/test.csv \
    --save_path results/ \
    --seed 42 --num_pool 32 --val_samples_per_target 8 --resample_alpha 0.1
```
For each target row it generates for **both** directions (agonist `d*=+1`, antagonist `d*=-1`), scores
with the oracle + affinity gate, applies softmax(reward/α) weighted resampling (Algorithm 2), keeps only
RDKit-valid peptides, and writes `results/td3b_results_seed{seed}.csv`.

**Training** (multi-target): edit paths in `launch_multi_target.sh` (`BASE_PATH`, checkpoints, data,
oracle, and `WANDB_ENTITY` — blank by default; set your own), then `bash launch_multi_target.sh`. It calls
`finetune_multi_target.py`. Key knobs live in the launch script: `CONTRASTIVE_WEIGHT`(λ), `KL_BETA`(β),
`SIGMOID_TEMPERATURE`(τ), `NUM_ITER`/`NUM_CHILDREN` (MCTS), `TARGETS_PER_MCTS`(K), and the cadence flags
below.

**Baselines** (CG, SMC, TDS, PepTune, Unguided): `cd baselines && ./run.sh <csv> <baseline> <device>`.
Multi-GPU via a 5th/6th arg (`torchrun`). Note this script loads `../pretrained/peptune-pretrained.ckpt`,
**not** `checkpoints/pretrained.ckpt` (see path landmines below).

**Demo**: `notebooks/TD3B_Inference_Demo.ipynb` (Colab T4).

## Architecture — the cross-file big picture

The pieces below only make sense together; reading any one file in isolation misses the flow.

### Diffusion backbone — `models/`
- `Diffusion(L.LightningModule)` in `models/diffusion.py` is the MDLM core (absorbing-state / masked
  discrete diffusion). The denoiser is `models/roformer.py::Roformer`, a thin wrapper over HuggingFace
  `RoFormerForMaskedLM` (rotary embeddings). Tokens are **SMILES** via `tokenizer/my_tokenizers.py::SMILES_SPE_Tokenizer`.
- Absorbing state = the tokenizer's `[MASK]` id (`self.mask_index`), not a constant. Generation starts
  fully masked (`sample_prior` → all-mask) and reverse-diffuses. SUBS parameterization
  (`subs_parameterization`) forbids predicting MASK and pins already-unmasked positions ("carry-over
  unmasking"). Callers use `single_reverse_step` / `single_noise_removal` (final step guarantees no
  surviving MASK); MCTS uses the `batch_mcts_reverse_step` / `mcts_reverse_step` variants which also
  return per-step policy vs. pretrained log-probs for the importance log-ratio `log_rnd`.
- Config is built by `configs/finetune_config.py::DiffusionConfig`, a **shim** that synthesizes
  duck-typed attribute objects (`type(...)()`) for backward compat. It fixes `parameterization='subs'`,
  `T=0` (continuous-time MDLM loss), `time_conditioning=False`, and `max_position_embeddings=1035`.
  It is a *partial* interface — only the finetune/eval/MCTS fields exist; pure-training paths
  (`antithetic_sampling`, `noise.state_dependent`, `vocab`, `model.length`) are absent and would
  `AttributeError` under it.
- `models/noise_schedule.py`: `get_noise(config)` supports geometric/loglinear/cosine/linear, but every
  reverse/sampling step **asserts `loglinear`**. A second hardcoded `LogPolyNoise` (cubic) masks
  peptide-bond tokens more slowly.

### Directional reward — `td3b/td3b_scoring.py`
`TD3BRewardFunction.__call__(List[str] of peptide SMILES) -> (rewards, info)`. Internals:
`g_ψ` = `scoring/functions/binding.py::BindingAffinity` (magnitude); `f_φ` = the `DirectionalOracle`
(direction prob ∈[0,1] + confidence κ); reward = `g_ψ · σ(d*·(f_φ−0.5)/τ)`. `create_td3b_reward_function`
is the factory that builds/loads the oracle, caches the encoded protein tokens, maps `'agonist'/'antagonist'`
→ `d* = +1/−1`, and returns the configured reward. `TD3BConfidenceWeighting` provides the
confidence-weighted importance weights used by MCTS (`w = κ·exp(S/α)`).

### Direction Oracle — `td3b/direction_oracle.py`
`DirectionalOracle` wraps `ESM_TR2D2_GPCRClassifier`: **frozen ESM2** (`facebook/esm2_t33_650M_UR50D`,
1280-d — downloads from HF unless `esm_cache_dir`/`esm_local_files_only` set) encodes the **protein**;
a **frozen TR2-D2 RoFormer** encodes the **ligand/peptide SMILES**; both project to `d_model=256`, pass
1 self-attention layer each, then **2 stacked bidirectional cross-attention (BMCA) layers**, mean-pool,
concat, MLP → 2 logits. `predict_with_confidence` returns `f_φ = p_agonist` and `κ = max(softmax)`.
Loading needs four assets: the oracle `.pt`, the TR2-D2 ligand checkpoint, and the SMILES tokenizer
vocab+splits. The RoFormer config (768/8/8/1035) is hardcoded and must match the checkpoint.

### Losses — `td3b/td3b_losses.py`
`TD3BTotalLoss` = `L_WDCE + λ·L_ctr + β·L_KL` (λ=`contrastive_weight`, β=`kl_beta`, both default 0.1).
`L_WDCE` is computed **externally** by `training/finetune_utils.py::loss_wdce` and passed in — it is the
policy-distillation term that reweights MCTS samples by `softmax(log_rnd)`. `L_ctr` is `ContrastiveLoss`
(margin, default) or `InfoNCELoss` over agonist/antagonist embeddings from `extract_embeddings_from_mdlm`
(reaches into `model.backbone.model`, RoFormer last hidden state, **must not** be under `no_grad`).
`L_KL` is per-position categorical KL to a **frozen deepcopy reference model** (the pretrained weights).

### MCTS — `mcts/peptide_mcts.py` + `td3b/td3b_mcts.py`
Base `MCTS` does Pareto/multi-objective tree search: root is fully masked; `select` descends via a
non-dominated (not scalar-UCB) set with `rd.choice`; `expand` samples `num_children` one-step
unmaskings, rolls each to a full sequence, filters with `PeptideAnalyzer.is_peptide`, scores valid ones,
and maintains a **Pareto buffer** of finished trajectories (each storing `x_final`, `log_rnd`,
reward, score vector). `TD3B_MCTS` subclasses it: injects the gated `TD3BRewardFunction`, pads the (N,2)
directional score vector to (N,5) so the base Pareto machinery works, folds confidence into `log_rnd`,
and `forward`/`consolidateBuffer` return **7** values (adding `directional_labels`, `confidences`).

### Training loop — `finetune_multi_target.py`
Self-contained inline loop (it does **not** call `td3b/td3b_finetune.py::td3b_finetune`, which is a
legacy/unused single-target loop). Per epoch it alternates:
- **MCTS generation phase** (every `resample_every_n_step` epochs, default 10): for each sampled target ×
  each direction, build a per-(target,direction) reward, run a fresh `TD3B_MCTS`, push Pareto survivors
  into a replay buffer with `directional_label` **forced to the intended `d*`** (not the oracle guess).
- **Gradient phase** (every epoch): shuffle the buffer, pad variable-length `x` to max-len with
  `mask_index` + build an attention mask, then WDCE + KL every batch, and contrastive only when a batch
  mixes both directions.

Three independent epoch-indexed cadence knobs: `resample_targets_every` (redraw K targets),
`resample_every_n_step` (MCTS phase), `reset_every_n_step` (reset vs. reuse the search tree).
`add_td3b_sampling_to_model(policy)` monkey-patches `sample_finetuned_td3b` onto the Diffusion instance —
required before validation/eval works (that method is not native to `Diffusion`).

### Data schema — `td3b/data_utils.py`
CSVs use columns `Target_Sequence`, `Ligand_Sequence` (peptide AA string), and `label`
(`agonist`/`antagonist`, mapped to `d*`); `TD3BDataset` also reads `Action`, `Target_UniProt_ID`,
`Ligand_UniProt_ID`. Binders are converted AA→SMILES via RDKit `MolFromSequence`. The multi-target
script uses its own in-file `TargetDataset` (groups by target, stores per-direction **median binder
length** used to set generation length), reading only `Target_Sequence`/`Ligand_Sequence`/`label`.
`inference.py` also reads `Target_UniProt_ID`, `Ligand_SMILES` (for length).

### New entry points (added 2026-07-12) — `finetune_on_target.py`, `generate_valid.py`
- **`finetune_on_target.py` (Function A)** — user-facing "bring your own target" wrapper; it does **not** reimplement training. It normalizes `--target_seq` (repeatable) / `--targets_csv` into a temp CSV (seeding a placeholder poly-G binder of `--binder_length` residues for any missing direction so `TargetDataset` can set a length prior), then **subprocess-invokes `finetune_multi_target.py`** with `--targets_per_mcts=<#targets>` + `--resample_targets_every 1` (finetune on ONLY those targets), and finally **generates in-process** reusing `inference.load_model`/`sample_sequences`/`score_sequences` + `create_td3b_reward_function` + Algorithm-2 resampling. `--direction` restricts only what is generated (finetuning always searches both). Writes `results/finetune_on_target/binders_<dir>_validity-<on|off>_seed<seed>.csv` + the finetuned ckpt under `results/<run_name>_<ts>/`. Paths default to repo root; missing heavy artifacts fail fast with a Google-Drive hint.
- **`generate_valid.py` + `sampling_strategies.py` (Function B)** — sampling-time validity boosters (no retraining) that reuse the model's own `sample_prior`/`single_reverse_step`/`single_noise_removal`/`forward` and change only token SELECTION (temperature → top-k → softmax → top-p), plus a **remask self-correction loop** (remask the lowest-confidence K% of invalid sequences and re-denoise for R rounds) and a **best-of-N** validity-guided rejection wrapper — no diffusion math is reimplemented. `sampling_strategies.generate(...)` dispatches `baseline, more_steps, top_p(=nucleus), top_k, low_temp, remask, best_of_n, nucleus_remask` (default `nucleus_remask`). `generate_valid.py` loads the real ckpt via `inference.load_model`, or falls back to `build_random_model` (random-init `Diffusion`, CPU dev) when `--ckpt_path` is absent. Output: valid-only CSV (`idx,sequence,n_chars`) + a printed valid-yield summary.
- **`inference.py --sampler`** (opt-in; default `baseline` = original behavior, byte-for-byte) selects a
  Function-B sampling strategy for the candidate pool, with pass-through knobs
  (`--num_steps --top_p --top_k --temperature --remask_rounds --remask_frac --best_of_n`). Non-baseline
  strategies dispatch to `sampling_strategies.generate`. Function B is also available standalone via
  `generate_valid.py`.

## Landmines

A batch of path/wiring bugs that stopped the OSS release from running was fixed on 2026-07-09.
`python inference.py --help` and `python finetune_multi_target.py --help` now import cleanly (verified
in the `tr2d2-pep` conda env); a full generation run still needs the Google-Drive artifacts + a GPU.

**Fixed — do NOT reintroduce:**
- `td3b/td3b_finetune.py`: the `from plotting import ...` (no such module) is now `try/except`-guarded,
  and `loss_wdce` is imported **lazily inside `td3b_finetune()`** to break a `finetune_utils ↔ td3b`
  circular import. Keep both — `td3b/__init__.py` eagerly imports `td3b_finetune`, and `finetune_utils`
  imports the `td3b` package at module load, so any module-level `from training.finetune_utils import …`
  in `td3b_finetune.py` re-creates the cycle (it was inference.py's first crash, before plotting).
- Stale `tr2d2-pep/` prefix stripped from the tokenizer loader (`finetune_utils.load_tokenizer`), every
  classifier loader (`scoring/functions/*.py`, `scoring/scoring_functions.py`, `binding.py`), the
  `Diffusion` fallback tokenizer, and the training results dir. Assets now resolve to the README layout
  (`tokenizer/`, `scoring/functions/classifiers/`, `results/`).
- `inference.py` reward wiring rewritten: build `MultiTargetBindingAffinity` + `DirectionalOracle` once,
  wrap each target with `TargetSpecificBindingAffinity`, call `create_td3b_reward_function`. (It used to
  call a non-existent `create_reward_function` signature whose `TypeError` was swallowed → empty CSV.)
- Demo notebook: `from models.diffusion import Diffusion` (was `from diffusion import …`), clone URL now
  points at the OSS HF repo with `GIT_LFS_SKIP_SMUDGE=1`, `total_memory` typo fixed.
- Added `__init__.py` to the 8 dirs `find_packages()` missed; fixed corrupted `configs/peptune_config.yaml`
  key (`batchinohup ng` → `batching`).

**Fixed — round 2 (runtime hardening, 2026-07-10; verified dynamically in `tr2d2-pep`):**
- `inference.py` Algorithm-2 resampling: now gates candidates by `finite-reward AND valid-peptide`
  **before** sampling and draws **without replacement** (`k=min(val_samples_per_target, n_eligible)`).
  Previously `replacement=True` + the peaked softmax produced duplicate rows (inflated counts, skewed
  means) and validity was filtered only afterward (could save 0 samples despite valid candidates).
- Checkpoint-load guards (silent-random-weights class): `inference.py::load_model` now raises if the
  backbone loaded **no** weights and warns on partial loads; `direction_oracle.py::TR2D2RoFormerEncoder`
  now unwraps `model_state_dict`/`state_dict` and raises if **zero** RoFormer keys matched (was silently
  leaving the ligand encoder random); `binding.py` tolerates raw/`state_dict`/`model_state_dict` ckpt
  containers; `_load_state_dict_flexible` loudly flags missing **non-ESM** (trained) keys. All guards
  fail only on the impossible-for-a-valid-checkpoint case, so they can't break a good load.

**Must NOT change — tied to pretrained weights:** `max_position_embeddings=1035`,
`parameterization='subs'`, `T=0`, `time_conditioning=False`. `Diffusion.forward` hard-fails if `seq_len > 1035`.

**Remaining known quirks (not inference blockers):**
- **Checkpoint generates valid SMILES only at SHORT length (verified end-to-end 2026-07-12 on real weights
  from `/data1/hanqun/TD3B/checkpoints`).** Valid-peptide yield vs generation length (SMILES tokens),
  `is_peptide` over 32–64 samples: L=40 → 28% baseline / 59% `best_of_n`; L=60 → ~5–30%; **L≥150 → ~0%
  regardless of step count (128/256/512) or best_of_n.** Real 25–30-residue binders are 150–210 tokens, i.e.
  outside the valid regime — that is why a naive `inference.py` run writes an empty CSV. Fix applied:
  `inference.py` now derives length from the binder's *token* count (not char count; reads the `SMILES`
  column, not the nonexistent `Ligand_SMILES`), adds `--seq_length`/`--max_seq_length`, and hints on empty
  output. Working run: `--seq_length 50 --sampler best_of_n --best_of_n 6` → 16 valid oracle-scored binders,
  antagonist direction-accuracy ≈ 1.0. This is a model-capability limit, not a code bug.
- **ESM2 network fetch:** `facebook/esm2_t33_650M_UR50D` is downloaded at inference twice — by the oracle
  (`transformers`) and by `BindingAffinity`/`MultiTargetBindingAffinity` (`fair-esm`, no offline flag).
  Offline runs need warm HF + torch-hub caches.
- **Training default paths** (`finetune_multi_target.py:551`, factory fallback `td3b_scoring.py:345`) still
  resolve under `{base_path}/tr2d2-pep/...` with a wrong default oracle filename
  (`best_model_tr2d2_gpcr_fixed.pt` vs. shipped `direction_oracle.pt`). `launch_multi_target.sh` overrides
  all of these with explicit `checkpoints/` args, so training-via-launch-script works; a bare
  `python finetune_multi_target.py` does not.
- `baselines/run.sh` loads `pretrained/peptune-pretrained.ckpt` (not `checkpoints/pretrained.ckpt`) and
  defaults `CSV_PATH="To Be Added"` — pass real args.
- `td3b/td3b_finetune.py::td3b_finetune()` is legacy/unused (superseded by `finetune_multi_target.py`);
  still writes to `{base}/TR2-D2/tr2d2-pep/results/...` and needs the optional `plotting` module.
- `TD3BDataset`/`load_td3b_data` (oracle-training path) require CSV columns `Action`, `Ligand_UniProt_ID`
  that the inference/finetune CSVs don't carry.
- Dead/inert flags: `--contrastive_type` (loss always uses `'margin'`), `--num_epoch_for_sampling`,
  `min_affinity_threshold=0.0` (down-weight branch never fires), `use_confidence_weighting` (stored but
  never applied in `compute_gated_reward`).
- **`PeptideAnalyzer.is_peptide` (`utils/app.py`) false-positives** on atom-only / pure-amino-acid-letter
  strings (e.g. `"CCNCCF"`, `"cccccc"` → True) because it checks AA-letter membership before RDKit. Harmless
  for a well-trained model (emits real peptide SMILES) but can inflate `valid_mask` for an untrained/early
  checkpoint. Shared by MCTS/baselines — do not change its semantics without checking all callers.
- Oracle confidence κ is actually ∈[0.5,1] (max of a 2-class softmax), not the documented [0,1]; unused at
  inference. `TD3BConfidenceWeighting.compute_importance_weights` uses raw `exp(reward/α)` which can overflow
  for large affinities — off the inference path (MCTS/training only).
- **Device coupling:** `models/roformer.py` and helpers pick their own device independently of `--device`
  (some hardcode `cuda:0`); `resolve_device` in `scoring_functions.py`/`direction_oracle.py` silently falls
  back to `cuda:0`/CPU. Mixed-device errors are easy to create.

**New entry points (2026-07-12) — usage caveats:**
- **The validity toggle is a FILTER, not a reward term.** `finetune_on_target.py --validity_reward {on,off}`
  (default `on`) never changes the reward formula (`affinity × direction`); it toggles the
  `PeptideAnalyzer.is_peptide` gate on **both** halves — the finetune-side MCTS expansion (forwarded to
  `finetune_multi_target.py --validity_reward` → `args.enforce_validity`) and the generation-side Algorithm-2
  resampling. `off` retains invalid samples (each output row still records `is_valid`).
  `--finetune_validity_hook {on,off}` decouples the finetune-side gate from the generation-side toggle.
- **Function B benchmarks were on a RANDOM-INIT model** → only *relative* trends (which strategy helps most at
  long length) are meaningful; **absolute** valid-yields need the real checkpoint. `generate_valid.py`
  **silently falls back to random init** when `--ckpt_path` is missing/omitted (it prints a NOTE, but the
  numbers are garbage) — always pass a real `--ckpt_path` for reportable yields.
- `finetune_on_target.py` runs the finetune half as a **subprocess** (default `WANDB_MODE=disabled`) and locates
  the produced checkpoint by diffing `results/<run_name>_*` dirs before/after (newest `model_final.ckpt`, else
  newest `model_epoch_*.ckpt`); a renamed/failed run dir would break that discovery. Its own oracle/path defaults
  are passed through explicitly so the subprocess never falls back to `finetune_multi_target.py`'s legacy
  `{base}/tr2d2-pep/...` defaults.
