# Claim 3: GSM8K comparison against Min-p, Top-p, and Top-H

**Claim.** Top-W is evaluated against Min-p, Top-p, and Top-H on GSM8K across multiple temperatures and models (Table 1).

**Self-assessed status: unreplicated**

## What this reproduction did NOT do

Not reproduced. The Table 1 comparison requires GSM8K decoding runs across three instruction-tuned models and five temperatures. No language model was executed in this reproduction; no GSM8K accuracy numbers exist here. The official evaluation harness (run.sh, huggingface.py) is pinned in the upstream manifest for an independent GPU reproduction.

## What exists for an independent benchmark rerun

The official repository (pinned at
`github.com/arashgholami/top-w-decoding@5949bfae5e6a81bc279c65923f1adc1c9f2e2059`) ships the evaluation
harness: `run.sh` (GSM8K), `run_gpqa.sh`, `alpaca_generate_w.py`, and
`huggingface.py`. Reproducing Table 1 requires decoding GSM8K with three
instruction-tuned models at five temperatures per method, which needs
GPU inference outside this CPU-only attempt's budget.

## Synthetic distribution shaping (context only — NOT benchmark evidence)

On fixed synthetic logits (seed 42, 500-token vocabulary), the decoder
comparison at the paper's temperatures:

| T | Top-W H | Min-p H | Top-p H | Top-H H | Top-W kept | Top-p kept |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | -0.0000 | 0.7960 | 1.9557 | -0.0000 | 1 | 25 |
| 0.7 | -0.0000 | 2.3968 | 3.3608 | -0.0000 | 1 | 68 |
| 1.0 | -0.0000 | 3.5336 | 4.4128 | -0.0000 | 1 | 151 |
| 1.5 | -0.0000 | 4.7612 | 5.1723 | 1.7034 | 1 | 254 |
| 2.0 | -0.0000 | 5.4768 | 5.5051 | 2.2626 | 1 | 311 |

These numbers characterize how each truncation rule shapes a
distribution; they say nothing about GSM8K accuracy.
