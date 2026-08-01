---
title: SCALE Reproduction
sdk: gradio
app_file: app.py
tags:
  - icml2026-repro
  - paper-7MlfE2Da2W
  - robotics
  - reproducibility
---

# SCALE Reproduction

This Space reports CPU-only reproduction evidence for "SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models."

The bundle pins `snumprlab/scale@b4ad2a69d14f91712704711e810cf9830e2b7121` and inspects official source files for the SCALE self-uncertainty, adaptive action decoding, and adaptive visual attention paths. Robot benchmark and real-world success-rate claims are marked unavailable unless independently rerun from raw executable artifacts.

Run locally:

```bash
uv run --project submissions/scale-self-uncertainty-conditioned-adaptive-looking-and-execution-for-vision-language-action-models python submissions/scale-self-uncertainty-conditioned-adaptive-looking-and-execution-for-vision-language-action-models/generate_evidence.py --output submissions/scale-self-uncertainty-conditioned-adaptive-looking-and-execution-for-vision-language-action-models/evidence/bundle.json
```
