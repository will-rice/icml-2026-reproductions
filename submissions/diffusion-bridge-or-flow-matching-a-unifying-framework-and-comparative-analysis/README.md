---
title: DBFM Reproduction
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
tags:
  - icml2026-repro
  - paper-aIFgQusnPy
---

# Diffusion Bridge or Flow Matching? Reproduction

Paper: `aIFgQusnPy`
Attempt: `daf2b529-c050-4cde-9218-281e985315dd`

This submission provides CPU-only evidence for selected claims from
"Diffusion Bridge or Flow Matching? A Unifying Framework and Comparative
Analysis." It pins the paper to `arxiv:2509.24531v2` and the released code to
`zhukaizhen/diffusion_bridge_flow_matching@2def77bd3ee7a2a37cdf6ce5d5393915604619f7`.

The evidence bundle intentionally marks full image restoration, inpainting,
training-data scaling, and network-condition ablation claims as unavailable
because the released workflow requires external datasets, checkpoints, CUDA
training, and image metrics that are not recomputed here.

Run:

```bash
python generate_evidence.py
python -m pytest tests -q
```
