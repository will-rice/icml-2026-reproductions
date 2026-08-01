---
title: MedMosaic Reproduction Evidence
emoji: "M"
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
tags:
  - icml2026-repro
  - paper-OMdQJQwp26
---

# MedMosaic Reproduction Evidence

This submission audits the released MedMosaic Hugging Face dataset index for
paper `OMdQJQwp26`, "MedMosaic: A Challenging Large Scale Benchmark of Diverse
Medical Audio".

The evidence uses only the pinned `data/test.parquet` index from
`icml-anon-submission/medmosaic-dataset@a6ea67bd4a65b87248c6651e559656b2c31fa669`.
It does not download the full audio payload and does not run medical model
inference.

```bash
uv run --project . python generate_evidence.py --output evidence/bundle.json
uv run --project . python -m pytest tests -q
```
