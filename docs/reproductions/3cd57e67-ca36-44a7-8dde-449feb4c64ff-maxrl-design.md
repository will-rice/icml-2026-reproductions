# Reproduction Design: Maximum Likelihood Reinforcement Learning

Attempt: `3cd57e67-ca36-44a7-8dde-449feb4c64ff`
Paper: `EeuLO2BjFN`
Owner: `codex-paper-owner-04`
Snapshot: `f4900cfc639405f1865cf4cdc4eee4d8aa321fb566510dc852b614b43a1751df`

## Scope

Build a CPU-only evidence package for MaxRL using the released official code
and small deterministic simulations. The package will not report paper table
numbers as reproduced results. It will classify claims that require multi-node
Qwen3 or ImageNet training as unavailable unless the released artifacts contain
machine-readable logs or checkpoints sufficient for direct recomputation.

## Pinned Artifacts

- Official code: `https://github.com/tajwarfahim/maxrl`
- Code revision: `7197bbb46a2ecd866da52f6b401ff20a34fe9390`
- Project page: `https://zanette-labs.github.io/MaxRL/`
- Paper: `arxiv:2602.02710`
- Released models/weights: Hugging Face collection linked from the official
  README; specific model revisions will be pinned only if used as evidence.

## Evidence Plan

1. Static artifact audit:
   - Verify the official README and scripts expose MaxRL experiments and
     document GPU requirements.
   - Verify `qwen3_experiments/run_qwen3_training.sh` selects
     `algorithm.adv_estimator=maxrl` for the headline Qwen3 training path.
   - Verify `verl/trainer/ppo/core_algos.py` implements the MaxRL advantage
     branch and compare it with GRPO/REINFORCE branches.
2. Toy mathematical checks:
   - Implement a small Bernoulli sampling environment.
   - Enumerate all rollout success/failure outcomes for small `N` and verify
     the conditional-success estimator expectation equals the truncated
     MaxRL gradient objective used in the paper statement.
   - Verify increasing rollout count reduces the truncation gap to the exact
     maximum-likelihood gradient for a range of probabilities.
3. Claim classification:
   - Mark objective-family, estimator, and infinite-compute-limit claims as
     toy or verified depending on exact code/math checks.
   - Mark all headline Pareto-dominance and 20x scaling claims unavailable
     unless exact released evaluation outputs are discovered and recomputed
     from pinned checkpoints/logs.

## Validation

The submission will include pytest tests that fail before the evidence code is
implemented:

- code pin and metadata extraction test;
- estimator expectation identity test for small rollout counts;
- truncation convergence test;
- claim-status and no-paper-number-copying test.

The full validation command set will include the submission pytest suite,
repository root pytest, `quick_validate.py`, and `pre-commit run -a`.

## Expected Limitations

The official README states the Qwen3 experiments used 4 nodes of 8 H200 GPUs.
Those experiments are out of scope for this CPU-only worker. The reproduction
will therefore provide independently executable evidence for the core MaxRL
mechanism and will explicitly leave large-scale empirical claims unreproduced
unless primary released artifacts make them independently checkable.
