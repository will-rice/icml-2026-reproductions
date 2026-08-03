---
title: VenusBench-Mobile Reproduction Evidence
sdk: gradio
app_file: app.py
tags:
  - icml2026-repro
  - paper-coHiGZOFtS
---

# VenusBench-Mobile Reproduction Evidence

This submission verifies released-artifact claims for:

VenusBench-Mobile: A Challenging and User-Centric Benchmark for Mobile GUI Agents with Capability Diagnostics.

The evidence bundle is generated from the pinned upstream branch:

- Repository: `https://github.com/inclusionAI/UI-Venus.git`
- Branch: `VenusBench-Mobile`
- Commit: `5b2c618ef146ea38890ea35dca8b07ec2d0284dd`
- License: Apache-2.0

## Scope

The reproduction inspects the released benchmark repository and records counts, metadata fields, PUDAM labels, verification paths, and stability-mode definitions in `evidence/bundle.json`.

It does not run Android emulator episodes or recompute agent success rates. The stability claim is therefore marked `partial`: the released README and scripts support the five-mode stability protocol as Original, Question Variation, Chinese, Mobile Dark mode, and Pad mode, but the exact challenge wording about min/max setting variants is not present in the released scripts.

## Commands

```bash
uv run pytest tests/test_evidence_bundle.py
uv run python generate_evidence.py --source-root /tmp/icml-venusbench-inspect-fz4OWv
```
