# d2 Reproduction Design

Attempt: `0f0cfb98-1f1a-4c31-8881-32cb6e02abde`
Paper: `ldCiNVFt8O`
Owner: `codex-paper-owner-05`
Snapshot: `730efd6146ac7814c07bd5e2d3908fb59c0435dd83f013b261493ce06c6b3d08`
Phase at design time: `design-pending`

## Primary Artifacts

- Paper: `arxiv:2509.21474`
- Official project page: `https://www.guanghanwang.com/d2/`
- Official code: `github:kuleshov-group/d2@381b9f14f4afd0719297ac852e4015c74e0ed235`
- Referenced checkpoint: `hf:GuanghanWang/d2_anyorder_causal_llada_intellectsft@3c334aa4931697841a923d6caad3b12d5eaa4409`
- Referenced checkpoint: `hf:GuanghanWang/d2_anyorder_causal_llada_intellectsft_gsm8k@e93476e1f676abfaaf0bdc036aa24d3f04c213f4`

The code repository is Apache-style redistributable only where its license permits; the Hugging Face checkpoint metadata did not expose a license field, so the submission will not redistribute weights.

## Target Claims

1. d2 is a reinforcement-learning framework for masked diffusion language models built around estimating sampling-trajectory likelihoods.
2. d2-AnyOrder provides exact trajectory likelihood with a single model pass for DLMs that support any-order decoding.
3. The paper empirically shows that any-order decoding support is not universal across widely used DLMs.
4. d2-StepMerge approximates trajectory likelihood for standard masked diffusion models with a tractable compute-accuracy tradeoff.
5. d2 reports stronger performance than widely used RL baselines when applied to popular diffusion language models.
6. d2 reports new state-of-the-art diffusion-language-model results on Countdown, Sudoku, GSM8K, and MATH500 reasoning benchmarks.

## Evidence Plan

Create a CPU-only Python project under `submissions/d2-improved-techniques-for-training-reasoning-diffusion-language-models`.

Tests first:

- Unit tests for finite masked-token trajectory likelihood calculators.
- Unit tests proving the any-order toy likelihood equals exhaustive ordered trajectory enumeration for an order-invariant toy DLM while requiring one aggregate pass.
- Unit tests proving StepMerge groups trajectory steps and converges toward exact trajectory likelihood as the merge factor becomes finer.
- Unit tests for source audits that detect the official AnyOrder and StepMerge trainer implementations, benchmark scripts, and absence of raw result logs.
- CLI test that writes a deterministic `evidence/bundle.json`.

Implementation:

- Fetch or inspect the pinned GitHub source and record SHA-256 hashes for key files.
- Audit official source paths:
  - `diffu-grpo-ao/diffu_grpo_trainer_ao.py` for d2-AnyOrder single-pass logits, doubled sequence construction, and causal pair attention.
  - `diffu-grpo/diffu_grpo_trainer.py` for trajectory masks, StepMerge `N`, per-token log-probabilities, and old/reference trajectory likelihood code.
  - `diffu-grpo*/bash_scripts/*d2*.sh` for d2 trainer names and dataset coverage.
  - `eval/parse_and_get_acc.py` plus `dataset/*` for benchmark parser/data scaffolding.
- Recompute toy finite-state likelihood checks locally without running large DLM checkpoints.
- Record that no raw machine-readable benchmark result files or latency/result logs are released in the official repo.

## Expected Claim Outcomes

- Claim 1: verified or toy, depending on source audit plus toy likelihood result.
- Claim 2: verified for the finite order-invariant likelihood identity and source wiring; limited because no checkpoint inference is run.
- Claim 3: toy or inconclusive; source/paper provides different code paths, but broad empirical model support is not rerun.
- Claim 4: toy; StepMerge approximation can be recomputed on a finite toy DLM and source wiring is present.
- Claims 5 and 6: inconclusive unless raw benchmark outputs are discoverable; paper-reported values will not be treated as reproduced measurements.

## Validation

The controller validation manifest will run:

1. `uv run --project submissions/d2-improved-techniques-for-training-reasoning-diffusion-language-models python submissions/d2-improved-techniques-for-training-reasoning-diffusion-language-models/generate_evidence.py`
2. `uv run --project submissions/d2-improved-techniques-for-training-reasoning-diffusion-language-models python -m pytest submissions/d2-improved-techniques-for-training-reasoning-diffusion-language-models/tests -q`
3. `uv run pytest -q`
4. `uv run skills/icml-repro-loop/scripts/quick_validate.py skills/icml-repro-loop`
5. `uv run pre-commit run -a`

## Cost And Safety

Estimated paid API cost: USD 0.00.
No GPU training or model-weight inference is planned. The submission is deterministic, CPU-only, and limited to released source/data inspection plus local finite-state likelihood checks.
