# NanoQuant Reproduction Evidence

This Space contains CPU-only reproduction evidence for ICML 2026 paper `qiZDlnvWTR`, "NanoQuant: Efficient Sub-1-bit Quantization of Large Language Models." Every number below is recomputed deterministically by `generate_evidence.py` in this Space (full records in `evidence/bundle.json`).

## Executed factorization evidence (claim 1)

The pinned upstream `factorize_admm_nanoquant` (`admm_nq_upstream.py`, verbatim Apache-2.0 copy of `src/nanoquant/core/admm_nq.py` at commit `a9e0a430`, SHA-256 `7145ba305ad6f3e1…`) was executed on a 64x128 weight matrix (planted rank-8 product plus 0.1-scaled Gaussian noise, torch.Generator seed 20260801, single CPU thread, 200 ADMM outer iterations):

| mid rank | binary-factor bits/weight | total bits/weight (fp32 scales) | relative Frobenius error | abs-factor rank-1 residual (A / B) | product identity max gap |
| --- | --- | --- | --- | --- | --- |
| 16 | 0.375 | 1.125 | 0.4164 | 3.5e-08 / 1.7e-07 | 0.00e+00 |
| 32 | 0.750 | 1.500 | 0.2406 | 7.9e-08 / 1.8e-07 | 0.00e+00 |
| 64 | 1.500 | 2.250 | 0.1332 | 5.0e-07 / 3.7e-07 | 0.00e+00 |

Reference point: plain 1-bit sign quantization (one global scale) of the same matrix has relative Frobenius error 0.6402 at 1.0 bit/weight. The executed NanoQuant factorization reaches lower error at strictly sub-1-bit binary-factor budgets. Both returned factors carry the paper's Scale-Binary structure: their entrywise magnitudes are numerically rank-1 (sigma2/sigma1 at floating-point precision, so each factor is a binary sign matrix under a rank-1 positive scale field with no zero entries), and the reconstruction equals the factor product exactly. The fp32 scale vectors dominate total bits only at this deliberately small matrix size; for an n x n layer their overhead is 64/n bits per weight (0.016 at n=4096), so the total also stays sub-1-bit at LLM scale.

## Claims

- `claim-1` (verified): NanoQuant formulates post-training LLM quantization as low-rank binary factorization with binary matrices and learned scales (Section 3).
  Evidence: The pinned Apache-2.0 factorize_admm_nanoquant was executed on CPU: it returns factor matrices with the paper's Scale-Binary structure (binary sign matrices under numerically rank-1 positive scale fields) whose product reproduces the returned reconstruction to floating-point precision, at binary-factor storage below one bit per weight; see numerical_experiments for the measured errors and bit budgets. Static audit: NanoQuantLinear binary U/V factor storage, latent binary training parameters, and learned scale_pre/scale_mid/scale_post parameters.
- `claim-2` (toy): NanoQuant is the only compared PTQ method marked as supporting both 70B+ LLMs and sub-1-bit compression (Table 1).
  Evidence: The released project metadata supports the NanoQuant side of the claim: PTQ, sub-1-bit operation, and >70B offload usage. This reproduction does not independently audit every Table 1 baseline, so the exclusivity portion is not treated as fully verified.

## Scope

- 70B compression and 8GB GPU fit were not rerun.
- WikiText perplexity tables were not rerun.
- Zero-shot commonsense evaluation was not rerun.
- CUDA GEMV/GEMM kernel benchmarks were not compiled or executed.
- Table 1 baseline exclusivity was not independently audited across all compared methods.
