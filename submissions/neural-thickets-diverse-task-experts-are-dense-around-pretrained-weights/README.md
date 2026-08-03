---
title: Neural Thickets Reproduction
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
tags:
  - icml2026-repro
  - paper-92oF5bU4cU
---

# Neural Thickets Reproduction Evidence

CPU-only evidence bundle for `Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights`.

This project audits the pinned official RandOpt repository at `536df0a308f3990b6270c991fbb96bd0b779a58e` and runs deterministic toy mechanism checks. It does not enter paper-reported benchmark metrics as reproduced measurements.

Run:

```bash
uv run python generate_evidence.py
uv run pytest -q
```

Judge-visible details are in `pages/report.md`; machine-readable results are in `evidence/bundle.json`.
