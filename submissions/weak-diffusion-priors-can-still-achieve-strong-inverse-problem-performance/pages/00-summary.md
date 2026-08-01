# Weak Diffusion Priors - CPU Reproduction Summary

Reproduction package for **Weak Diffusion Priors Can Still Achieve Strong
Inverse-Problem Performance** (ICML 2026, paper `fdkSA4F0lN`).

> **Notice:** the outcomes below are local reproduction audits in an exactly
> solvable linear-Gaussian analog of the paper's setting, computed on CPU from
> `generate_evidence.py`. They are **not** an official verdict, and they are
> **not** the paper's diffusion-model image experiments: the paper's UNet/DiT
> experiments, cross-domain studies, and figure-level results are out of scope
> for CPU-only reproduction and are **not** claimed here.

## What was executed (real numbers, deterministic)

Two claims are audited with exact linear-Gaussian computation (seeded, byte
reproducible - regenerating `evidence/evidence.json` is byte-identical):

1. **Informativeness vs prior strength** (claim
   `f92ef3142d3eb9876b4885e506e2318923f6277bd326f60f7e741fa6259e7ba9`,
   Table 1 direction): with 25% of pixels observed, a weak prior reaches only
   **0.579** of the strong-prior PSNR (20.29 vs 35.04 dB); with 90% observed
   the ratio rises to **0.816** (33.93 vs 41.60 dB). Highly informative
   measurements close most of the weak-vs-strong prior gap - the paper's
   qualitative Table-1 claim, reproduced in the analog model. See
   [01-informativeness-sweep.md](01-informativeness-sweep.md).
2. **Posterior concentration under weak priors** (claim
   `4d2832c903b2d7d6e55947d20468d734b233f664c678deed688b9c37ae5b8aac`,
   Theorem 3.1 direction): as the measurement ratio m/n grows 0.1 to 1.0
   (n=128), the weak-prior posterior trace collapses **230.0 to 9.97**, cosine
   similarity to the true signal rises **0.169 to 0.963**, and the
   reconstruction-error ratio vs the true-prior estimator approaches
   **0.988** - the posterior concentrates near the truth despite the weak
   prior, exactly as the theorem's conditions predict. See
   [02-posterior-concentration.md](02-posterior-concentration.md).

## Scope and honesty

- Local outcomes are labelled `supported (analog scale)`; nothing here is
  entered as a reproduction of the paper's image-domain PSNR tables,
  cross-domain transfer, local-correlation analysis, or failure-regime
  figures.
- Reproduce everything with:

```
uv run --project . python generate_evidence.py
uv run --project . python -m pytest tests -q
```
