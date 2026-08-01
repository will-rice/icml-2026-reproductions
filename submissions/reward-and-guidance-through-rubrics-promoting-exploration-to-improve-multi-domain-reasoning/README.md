---
title: RGR-GRPO Evidence
emoji: "R"
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.44.0
app_file: app.py
tags:
  - icml2026-repro
  - paper-AfqsNFzJcs
---

# RGR-GRPO Evidence

CPU-only evidence for `AfqsNFzJcs`, based on pinned arXiv source
`2511.12344v2`.

Run locally:

```bash
python generate_evidence.py --output evidence/bundle.json
pytest -q
```
