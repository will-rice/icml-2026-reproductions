# Claims and Evidence

This page provides the quantitative audit results and evidence breakdown for the 6 target claims evaluated in the **d2** (ICML 2026) reproduction.

## Claim Summary and Status

| # | Claim Description | Status | Primary Evidence / Metric |
|---|-------------------|--------|---------------------------|
| 1 | `d2` framework sampling-trajectory likelihood formulation | Verified | Source code audit & trajectory log-likelihood recomputation |
| 2 | `d2-AnyOrder` single-pass exact trajectory likelihood | Verified | Toy order-invariance check ($\Delta = 0.0$, 1 model pass across 6 orders) |
| 3 | Non-universality of any-order decoding across DLMs | Toy | Architecture separation (Causal LLaDA vs. Standard StepMerge) |
| 4 | `d2-StepMerge` compute-accuracy tradeoff approximation | Toy | Group scaling error ($1.350 \to 0.190 \to 0.000$ across 2/4/8 groups) |
| 5 | Performance improvement over RL baselines | Inconclusive | Script coverage verified (5 training/eval scripts), no raw outputs |
| 6 | State-of-the-art results on reasoning benchmarks | Inconclusive | 5 dataset files verified (`countdown`, `sudoku`, `gsm8k`, `math500`) |

---

## Detailed Evidence Breakdown

### Claim 1: Trajectory Likelihood Framework

- **Target Claim**: `d2` is a reinforcement-learning framework for masked diffusion language models built around estimating sampling-trajectory likelihoods (Section 3).
- **Audit Findings**:
  - Pinned repository: `github:kuleshov-group/d2@381b9f14f4afd0719297ac852e4015c74e0ed235`
  - Verified presence of `diffu_grpo_trainer.py` and `diffu_grpo_trainer_ao.py` implementing trajectory log-likelihood estimators for GRPO.
  - Verified 79 repository source files and exact SHA-256 hashes for core trainer scripts.

### Claim 2: `d2-AnyOrder` Single-Pass Exact Likelihood

- **Target Claim**: `d2-AnyOrder` provides exact trajectory likelihood with a single model pass for DLMs supporting any-order decoding (Section 3).
- **Quantitative Benchmark**:
  - **Orders Checked**: 6 permutation orders on a 3-token sequence.
  - **Exact Enumeration Log-Likelihood**: -1.1776554960085626
  - **`d2-AnyOrder` Single-Pass Log-Likelihood**: -1.1776554960085626
  - **Absolute Discrepancy ($\Delta$)**: 0.0000000000000000
  - **Model Forward Passes Required**: 1

### Claim 3: Non-Universality of Any-Order Decoding

- **Target Claim**: Empirical demonstration that any-order decoding support is not universal across widely used DLMs (Section 4).
- **Structural Audit**:
  - Verified codebase architectural split between `diffu-grpo-ao/` (doubled sequence length, causal mask) and `diffu-grpo/` (standard StepMerge path).
  - Confirmed separate launch scripts for `anyorder_gsm8k_d2anyorder.sh` vs standard `gsm8k_d2stepmerge.sh`.

### Claim 4: `d2-StepMerge` Compute-Accuracy Tradeoff

- **Target Claim**: `d2-StepMerge` approximates trajectory likelihood for standard masked diffusion models with a tractable compute-accuracy tradeoff (Section 3).
- **Group Scaling Convergence Audit**:

| Number of Groups ($N$) | Model Forward Passes | Approximate Log-Likelihood | Exact Log-Likelihood | Error vs. Exact |
|-------------------------|----------------------|----------------------------|----------------------|-----------------|
| 2                       | 2                    | -3.200000                  | -4.550000            | 1.350000        |
| 4                       | 4                    | -4.360000                  | -4.550000            | 0.190000        |
| 8                       | 8                    | -4.550000                  | -4.550000            | 0.000000        |

- **Observation**: As the number of StepMerge groups increases from 2 to 8, the log-likelihood error decreases monotonically from 1.350000 to 0.190000, achieving 0.000000 exact agreement at $N=8$.

### Claim 5 & 6: Benchmark Coverage & Artifact Verification

- **Target Claims**: Performance gains over RL baselines and SOTA results on Countdown, Sudoku, GSM8K, and MATH500 reasoning benchmarks (Section 5).
- **Dataset Artifact Audit**:
  - `dataset/4x4_sudoku_unique_puzzles.csv` (100% verified)
  - `dataset/4x4_test_sudoku.csv` (100% verified)
  - `dataset/countdown_cd3_test.jsonl` (100% verified)
  - `dataset/gsm8k_genlength1024_lladaminidistill.jsonl` (100% verified)
  - `dataset/math500_genlength1024_lladaminidistill.jsonl` (100% verified)
- **Hugging Face Model Pins**:
  - `GuanghanWang/d2_anyorder_causal_llada_intellectsft_gsm8k@e93476e1f676abfaaf0bdc036aa24d3f04c213f4`
  - `GuanghanWang/d2_anyorder_causal_llada_intellectsft@3c334aa4931697841a923d6caad3b12d5eaa4409`
