---
title: GUDA Evidence Bundle
emoji: "🧪"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.44.0
app_file: app.py
tags:
  - icml2026-repro
  - paper-5f0gw9YpZC
---

# GUDA Evidence Bundle

This Space reports independently generated evidence for ICML 2026 paper
`5f0gw9YpZC`, "GUDA: Counterfactual Group-wise Training Data Attribution for
Diffusion Models via Unlearning".

The bundle pins `sony/guda` at
`9fcf10cc4362199efc4f975e4a950df826fada07` and records CPU-only source,
metadata, and synthetic ranking checks. It does not claim full CIFAR-10 or
Stable Diffusion training results as reproduced measurements.

Run locally:

```bash
python generate_evidence.py --output evidence/bundle.json
pytest -q
```
