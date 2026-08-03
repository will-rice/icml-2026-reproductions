---
title: DHSA Reproduction Evidence
emoji: "D"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
tags:
  - icml2026-repro
  - paper-o3gN27ITWV
  - reproducibility
license: mit
---

# DHSA Reproduction Evidence

This Space contains CPU-only evidence for attempt
`020e5035-01ad-40a7-9ab0-9147289ab70c`, paper `o3gN27ITWV`.

The evidence pins the public `sxiong/DHSA` model card and
`sxiong/DHSA_Long-Data-Collections` dataset card, recomputes length-bucket
totals from the released JSON summary, and runs a deterministic sparse-routing
mechanism check. Full LongBench and 4-bit latency claims are marked
`inconclusive` because this submission does not execute those large model
benchmarks.

Run:

```bash
python generate_evidence.py
python -m pytest tests -q
```
