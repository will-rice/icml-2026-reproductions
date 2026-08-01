# Reproduction: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for Large Language Models

Paper `HSuU4xBmAv` | Attempt `c1b6dd10-f227-4d24-89a0-17fb00ec9147` |
Pinned upstream `arxiv:2602.10346v2+github.com/arashgholami/top-w-decoding@5949bfae5e6a81bc279c65923f1adc1c9f2e2059` |
Paid API cost USD 0.00

## Pages

| Page |
| --- |
| Executive summary (this page) |
| Claim 1: Wasserstein-entropy-mass objective — numerical audit |
| Claim 2: Exact prefix-form subset update vs brute force |
| Claim 3: GSM8K baseline table — not reproduced |
| Claim 4: GPQA baseline table — not reproduced |
| Claim 5: AlpacaEval / MT-Bench judge evaluations — not reproduced |
| Methods and provenance |

## Executive summary

This is an independent CPU reproduction of the Top-W decoding
*mechanism*, audited numerically against the released official
implementation (pinned in the upstream manifest). It does **not** rerun
any language model, so the benchmark tables of the paper are explicitly
out of scope here.

| Claim | Self-assessed status | Key numbers |
| --- | --- | --- |
| 1. Objective and geometry (Sec. 3, Alg. 1) | verified | f-step surrogate max error 1.2e-07; uniform-metric reduction 20/20; 40/40 converged |
| 2. Exact subset update (Sec. 4.2, Thm. 3.4) | verified | 120/120 brute-force matches; 25/25 identical to official code |
| 3. GSM8K table (Table 1) | unreplicated | no model runs; no accuracy numbers claimed |
| 4. GPQA table (Table 2) | unreplicated | no model runs; no accuracy numbers claimed |
| 5. AlpacaEval / MT-Bench win rates (Fig. 1-2) | unreplicated | no generations or judge runs; no win rates claimed |

Every number above is recomputed by `generate_evidence.py` from fixed
seeds; the full raw values are in `evidence/bundle.json`.
