---
title: "Conditional DPO/RLHF Reproduction"
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
tags:
  - paper-7UEBX1KU1y
  - icml2026-repro
---

# Conditional DPO/RLHF Reproduction

This Hugging Face Space presents CPU evidence auditing finite-response DPO, RLHF, and CPO mathematical claims from paper `arxiv:2605.20834v1`.

## Local Execution Commands

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv sync --frozen
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run conditional-dpo-repro generate --project-root . --output evidence.json
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run conditional-dpo-repro validate --project-root . --evidence evidence.json
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q
```

## Honest Limitations

- No language model was trained or evaluated.
- The benchmark SOTA claim was not reproduced.
- The advertised author repository (`visitworld123/CPO`) was unavailable during assessment.
- Only the challenge can issue official verdict labels.
