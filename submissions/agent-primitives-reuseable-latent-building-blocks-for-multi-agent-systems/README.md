---
title: "Agent Primitives Reproduction"
colorFrom: "blue"
colorTo: "green"
sdk: "static"
pinned: false
tags:
  - icml2026-repro
  - paper-CzShhpY2qU
---

# Agent Primitives Reproduction

CPU-only evidence package for `Agent Primitives: Reuseable Latent Building Blocks for Multi-Agent Systems`.

This reproduction separates source/audit evidence from paper-reported performance context. The local pipeline emits deterministic toy evidence for the three architecture/mechanism claims and marks the accuracy, token, and latency claims inconclusive unless released raw benchmark outputs are available.

## Run

```bash
uv run --project submissions/agent-primitives-reuseable-latent-building-blocks-for-multi-agent-systems python submissions/agent-primitives-reuseable-latent-building-blocks-for-multi-agent-systems/generate_evidence.py
uv run --project submissions/agent-primitives-reuseable-latent-building-blocks-for-multi-agent-systems python -m pytest submissions/agent-primitives-reuseable-latent-building-blocks-for-multi-agent-systems/tests -q
```

Generated files:

- `evidence/bundle.json`
- `pages/report.md`
