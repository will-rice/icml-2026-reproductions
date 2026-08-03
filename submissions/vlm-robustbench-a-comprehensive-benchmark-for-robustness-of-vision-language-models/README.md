---
title: VLM-RobustBench Reproduction
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
python_version: "3.12"
tags:
  - icml2026-repro
  - paper-HwXyyvK7ZJ
---

# VLM-RobustBench Reproduction

CPU-only evidence audit for ICML 2026 Agent Repro Challenge paper `HwXyyvK7ZJ`.

This reproduction verifies the augmentation taxonomy and corrupted-setting count
from pinned public artifacts. It does not rerun VLM inference. The glass-blur
severity claim is conservatively marked from primary project artifacts without
treating the paper-reported 8.1 percentage-point value as reproduced evidence.
