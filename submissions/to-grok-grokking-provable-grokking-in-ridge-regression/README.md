---
title: To Grok Grokking Reproduction
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
python_version: 3.10
app_file: app.py
pinned: false
tags:
  - icml2026-repro
  - paper-5nNNVY8NW4
---

# To Grok Grokking reproduction

CPU-only reproduction package for ICML 2026 paper `5nNNVY8NW4`.

The evidence bundle records an arXiv v4 theorem-structure audit and a
deterministic toy ridge-delay sweep. It does not machine-check the proof, reuse
paper-reported values as measurements, or reproduce the nonlinear ReLU
experiments.

## Commands

```bash
uv run python generate_evidence.py
uv run python -m pytest tests -q
```
