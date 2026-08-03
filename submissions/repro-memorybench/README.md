---
title: MemoryBench Reproduction Evidence
sdk: gradio
app_file: app.py
tags:
  - icml2026-repro
  - paper-If4X4W2HWx
---

# MemoryBench Reproduction

This submission provides CPU-only evidence for `If4X4W2HWx`, MemoryBench: A
Benchmark for Memory and Continual Learning in LLM Systems.

The evidence is generated from pinned public artifacts:

- `github:LittleDinoC/MemoryBench@5eafebca4e9ffbb2f0087ade13c498cf95fbc09a`
- `hf:THUIR/MemoryBench@3acd60a4bd35b43b408f0e6db4c5f1e88df5e96d`
- `hf:THUIR/MemoryBench-Results@742b433c48edfcb20dd102fd5fcf32f3053baada`

Run:

```bash
python generate_evidence.py
```

The generated bundle is written to `evidence/bundle.json`, and the judge-facing
summary is written to `pages/00-summary.md`.
