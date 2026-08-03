---
title: Learning Randomized Reductions Evidence
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
tags:
- paper-hCAEcqig2C
- icml2026-repro
---

# Learning Randomized Reductions Artifact Reproduction

This repository contains the CPU-only reproduction audit for ICML 2026 paper `hCAEcqig2C` (**Learning Randomized Reductions**, arXiv:2412.18134v5) under attempt `eb10c79b-fc26-47c4-88c1-6f45cb592833`.

## Local Reproduction Instructions

To run the complete audit locally:

```bash
# 1. Install dependencies
uv sync --frozen

# 2. Acquire and verify upstream artifacts (network required only for acquire)
uv run lrr-repro acquire --manifest evidence/inputs/upstream_manifest.json --cache-dir .cache/upstream

# 3. Execute offline reproduction audit
uv run lrr-repro audit --project-root . --cache-dir .cache/upstream --schema schema/evidence-v1.schema.json --output evidence/results.json

# 4. Validate evidence JSON bundle against schema
uv run lrr-repro validate evidence/results.json --schema schema/evidence-v1.schema.json --validation-output evidence/validation.json

# 5. Run full pytest suite
uv run pytest -q
```

## Reviewer Interface

Launch the read-only Gradio viewer locally:

```bash
uv run python app.py
```

Note: Remote LLM inference (Claude-Opus-4.1), GPU training, paid API calls, and Gurobi solver reruns were not executed as part of this audit. All results are derived deterministically from released primary raw artifacts and formal algebraic/finite-model checks.
