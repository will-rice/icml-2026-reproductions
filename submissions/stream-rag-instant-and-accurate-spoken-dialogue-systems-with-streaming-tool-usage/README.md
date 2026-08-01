---
title: Stream RAG Evidence
emoji: "🔎"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.44.0
app_file: app.py
tags:
  - icml2026-repro
  - paper-NMMmwSbzRx
---

# Stream RAG Evidence

CPU-only evidence for ICML 2026 paper `NMMmwSbzRx`, based on pinned arXiv
source `2510.02044v1`.

The bundle audits the released TeX source and runs deterministic toy checks of
fixed-interval streaming calls, model-triggered single-thread calls, latency
arithmetic, and negative-sampling label recovery. It does not claim AudioCRAG,
model accuracy, or latency results as reproduced measurements.

Run locally:

```bash
python generate_evidence.py --output evidence/bundle.json
pytest -q
```
