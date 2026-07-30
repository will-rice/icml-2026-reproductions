---
title: Top-W Geometry-Aware Decoding Reproduction
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
tags:
- icml2026-repro
- paper-HSuU4xBmAv
---

# Reproduction for "Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for Large Language Models"

Paper ID: `HSuU4xBmAv`
Attempt ID: `572f7d7b-f6a5-4004-9389-22ac5af0d0f6`

## Target Claims
1. Top-W decoding selects token subsets by optimizing a Wasserstein-entropy-mass objective using embedding-induced geometry (Section 3, Algorithm 1).
2. The method instantiates a practical alternating decoder with an exact subset-update step inside a candidate-pool loop (Section 4.2).
3. Top-W is evaluated against Min-p, Top-p, and Top-H on GSM8K across multiple temperatures and models (Table 1).

## Independent Reproduction Protocol
- CPU-only execution path with zero metered API cost.
- Machine-readable evidence bundle generated in `evidence/bundle.json`.
- Interactive Gradio demo displaying verified claim statuses and decoding metrics.
