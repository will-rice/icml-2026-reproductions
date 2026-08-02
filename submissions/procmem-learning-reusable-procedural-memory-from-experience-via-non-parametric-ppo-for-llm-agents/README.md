---
title: ProcMEM Reproduction Evidence
emoji: "\U0001F9EA"
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
tags:
  - icml2026-repro
  - paper-9kJQjx2B80
---

# ProcMEM Reproduction Evidence

This submission provides a CPU-only reproduction bundle for paper
`9kJQjx2B80`, "ProcMEM: Learning Reusable Procedural Memory from Experience
via Non-Parametric PPO for LLM Agents".

The evidence checks local implementations of the Skill-MDP representation,
semantic-gradient skill proposal, PPO-style candidate gating, and online
skill-pool scoring. Benchmark table values from the paper are recorded only as
paper-reported context and are not emitted as reproduced measurements.

Run:

```bash
uv run --project . --with pytest python -m pytest tests -q
uv run --project . python generate_evidence.py --output evidence/bundle.json
```
