---
title: DeMix Data Mixing Model Merging
emoji: 🔀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
- icml2026-repro
- paper-uyRIOjFgOn
---

# DeMix Reproduction Space

Reproduction evidence and interactive simulator for:
**Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training** (ICML 2026 Paper ID `uyRIOjFgOn`).

## Test

From this directory, run the complete suite in an isolated environment:

```bash
uv run --isolated --no-project \
  --with-requirements requirements-test.txt \
  python -m pytest -q
```
