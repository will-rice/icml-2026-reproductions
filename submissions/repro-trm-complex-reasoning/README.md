---
title: TRM Complex Reasoning Reproduction
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.20.0"
app_file: app.py
pinned: false
tags:
  - icml2026-repro
  - paper-IMFgiWw4jd
---

# TRM Complex Reasoning Reproduction

This submission checks released artifacts for "Characterizing, Evaluating, and
Optimizing Complex Reasoning" (`IMFgiWw4jd`).

It targets three claims:

- ME2 characterizes reasoning traces along macro/micro and
  efficiency/effectiveness dimensions.
- Reasoning traces are represented as DAGs with progression, branching, and
  merging structures.
- TRM is trained from the TRM-Preference dataset with a preference-loss reward
  model path.

The evidence is CPU-only. It pins the arXiv source hash, GitHub revision, HF
dataset revision, and HF model revision; it avoids downloading the 2 GB
training JSON and 30 GB model weights.

Run:

```bash
uv run python generate_evidence.py
uv run pytest -q
```
