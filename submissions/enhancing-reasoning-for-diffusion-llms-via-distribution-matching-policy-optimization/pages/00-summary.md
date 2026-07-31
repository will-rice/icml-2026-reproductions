# DMPO Reproduction Evidence Summary

This submission independently reproduces and evaluates key claims from the official DMPO repository:
`yuchen-zhu-zyc/DMPO@1661fa7d75f0ccec3bbc1b6cae94e9e3fb88571a`.

## Target Claims & Findings

1. **Distribution Matching Policy Optimization Objective (Section 3)**:
   - **Status**: Verified
   - **Details**: Verified WDCE loss formulation, reward-tilted softmax weighting, and weighted denoising cross-entropy computation through deterministic unit tests.

2. **Weight Baseline Subtraction (Section 3.4)**:
   - **Status**: Verified
   - **Details**: Verified advantage-centering baseline subtraction branches in `DMPO/dmpo_trainer.py` and confirmed output behavior on test inputs.

3. **R1-Zero-like Reasoning Recipe (Section 4)**:
   - **Status**: Verified (Config & Code Audit)
   - **Details**: Audited `DMPO/dmpo_train_config.yaml`, `DMPO/reward_func.py`, and `DMPO/dmpo_train.py`. Confirmed RL-only training structure with reasoning reward functions (GSM8K, MATH, Countdown, Sudoku).

4. **Benchmark Performance & Baselines (Table 1)**:
   - **Status**: Inconclusive (CPU-only Scope)
   - **Details**: Full 8B model training and distributed benchmark evaluation require non-portable multi-GPU infrastructure and exceed the USD 10 paid-API / CPU-only execution budget.

## Execution
- Unit tests: `uv run pytest -q`
- Evidence bundle generation: `uv run python generate_evidence.py`
