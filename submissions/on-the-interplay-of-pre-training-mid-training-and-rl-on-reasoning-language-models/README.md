---
title: "Interplay LM Reasoning Reproduction"
colorFrom: "blue"
colorTo: "green"
sdk: "static"
pinned: false
tags:
  - icml2026-repro
  - paper-TBaUfO9znF
---

# Interplay LM Reasoning Reproduction

CPU-only evidence audit for `TBaUfO9znF`, using pinned source and Hugging Face artifact revisions. The evidence bundle separates structural artifact checks from unavailable numeric training claims that would require large model training or raw official result logs.

Run:

```bash
uv run --project submissions/on-the-interplay-of-pre-training-mid-training-and-rl-on-reasoning-language-models python submissions/on-the-interplay-of-pre-training-mid-training-and-rl-on-reasoning-language-models/generate_evidence.py --output submissions/on-the-interplay-of-pre-training-mid-training-and-rl-on-reasoning-language-models/evidence/bundle.json
uv run --project submissions/on-the-interplay-of-pre-training-mid-training-and-rl-on-reasoning-language-models python -m pytest submissions/on-the-interplay-of-pre-training-mid-training-and-rl-on-reasoning-language-models/tests -q
```
