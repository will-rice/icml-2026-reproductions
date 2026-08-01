---
title: Learning Unmasking Policies Repro
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.29.0
app_file: app.py
pinned: false
license: mit
tags:
- icml2026-repro
- paper-F9NDKf5oPy
---

# Learning Unmasking Policies Reproduction

CPU-only evidence package for `Learning Unmasking Policies for Diffusion Language Models`.

The package audits the public Apple repository and performs deterministic local checks of masked-state transitions, confidence-based token selection, block schedules, and expert left-to-right unmasking. Benchmark claims that require trained checkpoints or raw evaluation outputs are marked inconclusive unless those artifacts are present.

## Run

```bash
uv run --project submissions/learning-unmasking-policies-for-diffusion-language-models python submissions/learning-unmasking-policies-for-diffusion-language-models/generate_evidence.py
uv run --project submissions/learning-unmasking-policies-for-diffusion-language-models python -m pytest submissions/learning-unmasking-policies-for-diffusion-language-models/tests -q
```
