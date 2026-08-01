---
title: WarmServe Reproduction
sdk: gradio
app_file: app.py
tags:
  - icml2026-repro
  - paper-DVHpvumD60
  - llm-serving
  - reproducibility
---

# WarmServe Reproduction

CPU-only reproduction evidence for "WarmServe: Enabling One-for-Many GPU Prewarming for Multi-LLM Serving."

The bundle pins `LLMServe/WarmServe@a60121519e077d2f128b597cbabc947e3e618aaf`, `arxiv:2512.09472v2`, and checks released source paths for the WarmServe scheduler, prewarm manager, CUDA virtual memory manager, worker prewarming hooks, model config, and trace generator. GPU cluster TTFT, ablation, and 512-GPU simulation claims are not counted as reproduced measurements.
