---
title: MaxRL Reproduction
emoji: M
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
tags:
- icml2026-repro
- paper-EeuLO2BjFN
---

# MaxRL Reproduction

CPU-only evidence package for ICML 2026 challenge paper `EeuLO2BjFN`.

This reproduction audits the pinned official MaxRL code and runs deterministic
toy checks for the MaxRL estimator/objective identity. It does not claim to
reproduce the paper's large-scale Qwen3, ImageNet, Pareto-dominance, or 20x
test-time scaling measurements.

Run:

```bash
python generate_evidence.py --source-root /tmp/maxrl-src-3cd57e67
pytest -q
```
