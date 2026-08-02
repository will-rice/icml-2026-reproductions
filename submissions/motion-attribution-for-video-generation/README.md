---
title: Motion Attribution for Video Generation Reproduction
emoji: 🎥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
python_version: "3.10"
app_file: app.py
pinned: false
tags:
  - icml2026-repro
  - paper-zAl9heLw4q
  - motion-attribution
  - video-generation
---

# Motion Attribution for Video Generation Reproduction

Reproduction repository for ICML 2026 Paper ID `zAl9heLw4q`: "Motion Attribution for Video Generation".

This is a CPU-only, toy-scale mechanism reproduction: the motion-mask
localization, frame-length bias fix, and dynamics-versus-magnitude claims are
exercised on deterministic synthetic videos with known ground truth, and the
measured numbers render into `pages/report.md`. The VBench fine-tuning
comparisons and the 74.1% human-preference study are reported as
**unreplicated**. Regenerate all evidence with:

```bash
uv run python generate_evidence.py
```
