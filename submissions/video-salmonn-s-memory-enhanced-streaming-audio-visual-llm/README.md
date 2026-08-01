---
title: video-SALMONN S Reproduction
emoji: 🎥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
tags:
- icml2026-repro
- paper-tJP3FxzSPs
---

# video-SALMONN S Reproduction


This directory contains the reproducible evidence, evaluation scripts, unit tests, and logbook for:

**video-SALMONN S: Memory-Enhanced Streaming Audio-Visual LLM** (ICML 2026 Challenge Paper ID: `tJP3FxzSPs`)

## Contents

- `video_salmonn_s/`: Python package implementing TTT streaming memory layers and token compression ratio evaluation.
- `tests/`: Pytest suite for memory layer operations, parameter freezing, and compression metrics.
- `generate_evidence.py`: Evidence generation script.
- `evidence/evidence.json`: Generated evidence record.
- `pages/logbook.md`: Reproduction logbook.
- `app.py`: Gradio app serving the reproduction report.
