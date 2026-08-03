---
title: Rare Event Analysis Reproduction
sdk: gradio
sdk_version: "6.20.0"
app_file: app.py
pinned: false
license: mit
python_version: "3.12"
short_description: CPU rare-event estimator checks for text generation
tags:
  - icml2026-repro
  - paper-2RJN5vDHG0
---

# Rare Event Analysis of Large Language Models

Independent CPU-only evidence for ICML 2026 challenge paper `2RJN5vDHG0`.

The reproduction uses an exactly enumerable finite stochastic text process to
check rare-event sampling, MBAR-style reweighting, and interval comparisons
without reporting paper-provided values as reproduced measurements.

```bash
uv run --project submissions/rare-event-analysis-of-large-language-models \
  python submissions/rare-event-analysis-of-large-language-models/generate_evidence.py
uv run --project submissions/rare-event-analysis-of-large-language-models \
  python -m pytest submissions/rare-event-analysis-of-large-language-models/tests -q
```
