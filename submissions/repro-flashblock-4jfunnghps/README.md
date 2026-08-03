---
title: FlashBlock Attention Caching Reproduction
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.20.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: CPU evidence for FlashBlock attention caching
tags:
  - icml2026-repro
  - paper-4jfuNNghPS
---

# FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion

This repository provides an independent, reproducible implementation and evidence bundle for ICML 2026 Paper `4jfuNNghPS`: **FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion** (arXiv:2602.05305v3).

Attempt: `ee4b5986-ff11-4f99-9a93-cd8fc43eb04d`
Snapshot: `c68adfe585882f99e8f3dd3ed496aedc650f5b64684955045d04513816cbe106`
Upstream revision: `arxiv:2602.05305v3`

## Reproduction Highlights & Target Claims

1. **`cross-step-attention-stability-discrepancy`**: Demonstrates high stability in synthetic block-external attention outputs ($\ge 0.95$ cosine similarity) across adjacent block diffusion steps compared to synthetic block-internal attention ($\le 0.70$).
2. **`block-external-attention-caching-speedup`**: Recomputes analytic FLOP and memory traffic reductions from reusing block-external attention $(A_{\text{out}}, L_{\text{out}})$.
3. **`log-space-attention-composition-fidelity`**: Verifies numerical equivalence ($L_\infty < 10^{-5}$) between FlashBlock's log-space composition operator and full single-pass dense attention.

The included CPU evidence supports the mechanism underlying the throughput claims. It does not reproduce Trado-8B hardware throughput or downstream generation-quality benchmarks.

## Usage & Verification

Run tests:
```bash
uv run --extra dev pytest -q
```

Generate evidence bundle:
```bash
uv run python generate_evidence.py
```
