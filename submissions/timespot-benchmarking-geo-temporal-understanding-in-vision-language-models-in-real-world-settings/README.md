---
title: TimeSpot Geo-Temporal VLM Reproduction
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
tags:
  - icml2026-repro
  - paper-XQlUqVCHJd
  - reproducibility
---

# TimeSpot: Benchmarking Geo-Temporal Understanding in Vision–Language Models in Real-World Settings

Official reproduction package for **TimeSpot** (ICML 2026 Paper ID `XQlUqVCHJd`, arXiv:2603.06687).

## Summary
TimeSpot defines a joint geo-temporal benchmark requiring structured prediction of 4 temporal and 5 geographic attributes from 1,455 ground-level photos across 80 countries. Empirical evaluation reveals a stark temporal weakness gap in state-of-the-art VLMs.

## Running Verification
```bash
uv run python generate_evidence.py
uv run pytest
```
