---
title: SleepLM Reproduction Evidence
emoji: "💤"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
tags:
  - icml2026-repro
  - paper-9wpwfSJCp9
  - sleeplm
  - polysomnography
  - reproducibility
---

# SleepLM Reproduction Evidence

This Space contains CPU-only reproduction evidence for `9wpwfSJCp9`,
`SleepLM: Natural-Language Intelligence for Human Sleep`.

The bundle verifies released-artifact claims from pinned GitHub, Hugging Face,
and project-page sources. It does not train on NSRR cohorts, download raw
credentialed sleep-study data, or claim reproduced benchmark performance.

Run locally:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run --project submissions/sleeplm-natural-language-intelligence-for-human-sleep pytest submissions/sleeplm-natural-language-intelligence-for-human-sleep/tests -q
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run --project submissions/sleeplm-natural-language-intelligence-for-human-sleep python submissions/sleeplm-natural-language-intelligence-for-human-sleep/generate_evidence.py
```
