---
title: DFlash Reproduction Evidence
emoji: "D"
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
tags:
  - icml2026-repro
  - paper-Oz335dV48X
---

# DFlash Reproduction Evidence

This submission audits the pinned DFlash source tree for paper `Oz335dV48X`,
"DFlash: Block Diffusion for Flash Speculative Decoding".

The evidence checks implementation-level mechanisms in
`z-lab/dflash@94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756`. It does not run
GPU serving benchmarks, download model weights, or reproduce paper speedup
tables.

```bash
uv run --project . python generate_evidence.py --output evidence/bundle.json
uv run --project . python -m pytest tests -q
```
