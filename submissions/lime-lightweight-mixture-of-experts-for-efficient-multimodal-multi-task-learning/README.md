---
title: LiME Reproduction
emoji: 🍋
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
tags:
- icml2026-repro
- paper-KRSZj8z5Lr
---

# LiME: Lightweight Mixture of Experts for Efficient Multimodal Multi-task Learning

This directory contains the reproducible evidence, evaluation scripts, unit tests, and logbook for:

**LiME: Lightweight Mixture of Experts for Efficient Multimodal Multi-task Learning** (ICML 2026 Challenge Paper ID: `KRSZj8z5Lr`)

## Contents

- `lime_peft/`: Python package implementing shared PEFT adapters, expert modulation vectors, zero-parameter routing, and parameter reduction metrics.
- `tests/`: Pytest suite for expert modulation, parameter efficiency, and routing granularity.
- `generate_evidence.py`: Evidence generation script.
- `evidence/evidence.json`: Generated evidence record.
- `pages/logbook.md`: Reproduction logbook.
- `app.py`: Gradio app serving the reproduction report.
