---
title: Top-W Geometry-Aware Decoding Reproduction
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
tags:
- icml2026-repro
- paper-HSuU4xBmAv
---

# Reproduction for "Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for Large Language Models"

Paper ID: `HSuU4xBmAv`
Attempt ID: `c1b6dd10-f227-4d24-89a0-17fb00ec9147`

## Target Claims
1. Top-W decoding selects token subsets by optimizing a Wasserstein-entropy-mass objective using embedding-induced geometry (Section 3, Algorithm 1).
2. The method instantiates a practical alternating decoder with an exact subset-update step inside a candidate-pool loop (Section 4.2).
3. Top-W is evaluated against Min-p, Top-p, and Top-H on GSM8K across multiple temperatures and models (Table 1).

## Logbook

The claim-by-claim evidence lives in `pages/*.md` and is rendered by the
Space app. Claims 1 and 2 are numerical audits of the mechanism: the
exact prefix-form S-step (Theorem 3.4a) is checked against brute-force
enumeration of all 1023 subsets per instance and against the
byte-exact vendored official implementation
(`evidence/inputs/upstream/`, pinned in
`evidence/inputs/upstream_manifest.json`). Claim 3 (GSM8K benchmark
table) is **not reproduced** — no language model was run.

## Reproduce

```bash
uv run --project . python generate_evidence.py
uv run --project . python -m pytest tests -q
```

CPU-only, zero metered API cost, fixed seeds; `evidence/bundle.json` is
the machine-readable output backing every number on the pages.
