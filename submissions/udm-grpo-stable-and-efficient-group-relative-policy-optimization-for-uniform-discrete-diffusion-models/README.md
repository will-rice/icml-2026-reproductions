---
title: UDM-GRPO Reproduction Audit
emoji: U
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
tags:
  - icml-2026
  - reproducibility
  - WJcFtJriqv
  - reinforcement-learning
---

# UDM-GRPO Reproduction Audit

This Space contains a CPU-only source/config audit for ICML 2026 paper `WJcFtJriqv`.

The audit pins `Yovecent/UDM-GRPO@d1bec49f4500873606f8345d81692143de059891` and checks that the released configs implement forward-process trajectory reconstruction, final-clean-sample actions, reduced-step training, and CFG-free rollout. GPU benchmark and ablation metrics are left inconclusive because they were not recomputed.
