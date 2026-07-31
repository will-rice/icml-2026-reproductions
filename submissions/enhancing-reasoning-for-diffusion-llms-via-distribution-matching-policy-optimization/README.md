---
title: Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization
emoji: "🎲"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
tags:
  - icml2026-repro
  - paper-09CSjVeDug
  - dmpo
  - diffusion-llm
---

# DMPO Reproduction

CPU-only evidence for selected implementation claims from
`yuchen-zhu-zyc/DMPO@1661fa7d75f0ccec3bbc1b6cae94e9e3fb88571a`.

Run:

```bash
uv run python generate_evidence.py
uv run pytest -q
```

The bundle verifies objective, baseline-subtraction, and recipe/configuration
evidence. It does not rerun 8-GPU LLaDA training or reproduce Table 1 accuracy
claims.
