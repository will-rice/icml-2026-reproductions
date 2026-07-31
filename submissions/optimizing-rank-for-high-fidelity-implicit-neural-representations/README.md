# Optimizing Rank for High-Fidelity Implicit Neural Representations (ICML 2026 Reproduction)

This repository provides a verified reproduction of the ICML 2026 paper:
**"Optimizing Rank for High-Fidelity Implicit Neural Representations"** (Paper ID: `2azIa9tfl3`).

## Overview
The paper demonstrates that:
1. Low-frequency spectral bias in vanilla MLP INRs stems from **stable-rank degradation** of weight matrices during Adam training rather than architectural incapacity.
2. Replacing Adam with **Muon** (Newton-Schulz regulated near-orthogonal weight updates) maintains matrix stable rank and significantly improves output fidelity across architectures (Siren, Vanilla MLP) and domains (2D image fitting, sparse-view CT reconstruction, audio, super-resolution).

## Verification Quickstart

```bash
# 1. Run full test suite
uv run pytest submissions/optimizing-rank-for-high-fidelity-implicit-neural-representations/tests/test_repro.py

# 2. Generate evidence bundle
PYTHONPATH=submissions/optimizing-rank-for-high-fidelity-implicit-neural-representations/src uv run python submissions/optimizing-rank-for-high-fidelity-implicit-neural-representations/generate_evidence.py
```
