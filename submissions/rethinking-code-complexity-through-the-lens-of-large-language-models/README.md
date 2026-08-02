---
title: LM-CC Reproduction
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
tags:
- icml2026-repro
- paper-tI5CFbRhmV
---

# LM-CC Reproduction

Attempt `48a537d9-3320-4f51-80f5-45c226518c38` for ICML 2026 Agent Repro
Challenge paper `tI5CFbRhmV`.

Run:

```bash
uv run python generate_evidence.py
uv run python -m pytest tests -q
```

The evidence generator clones the pinned public upstream repository into
`/tmp`, records file hashes, and writes `evidence/bundle.json`. It does not
vendor upstream code or cached outputs because the probed upstream tree has no
explicit license file.
