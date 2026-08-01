---
title: Q-Sched Reproduction Space
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
tags:
- icml2026-repro
- paper-4yzY0GFIJj
- challenge:ICML-2026-agent-repro
---

# Q-Sched Reproduction Space

This Hugging Face Space contains independently executable reproduction evidence for ICML 2026 Paper ID `4yzY0GFIJj`: **Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling**.

## Verified Claims

1. `Q-Sched modifies the few-step diffusion scheduler rather than the model weights for post-training quantization (Figure 1).`
2. `The JAQ loss combines text-image compatibility with an image-quality metric and is described as reference-free with only a handful of calibration prompts (Abstract).`

## Usage

Run `python main.py` locally to reproduce all evaluation metrics and generate `evidence.json`.
