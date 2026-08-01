# Optimizing Rank for High-Fidelity INRs — Reproduction Summary

Reproduction of **"Optimizing Rank for High-Fidelity Implicit Neural
Representations"** (paper `2azIa9tfl3`, arXiv:2512.14366).

Every number on these pages is computed by this repository on CPU with pinned
seeds. Adam and Muon runs always start from **identical initial weights** and
receive the same number of steps, so reported PSNR gaps are attributable to
the optimizer. No paper value is copied into a measurement field, and results
that contradict the paper are reported as measured rather than tuned until
they agree: **claim 3 does not reproduce at this scale, and claim 4
reproduces in only 2 of 4 modalities.**

## Claim status

| Claim | Status | Scale of the evidence |
| --- | --- | --- |
| 1 | `reproduced` | toy-scale: 4-layer 64-unit MLP fitting a 32x32 multi-frequency image |
| 2 | `reproduced` | toy-scale: Siren and vanilla-MLP INRs, 32x32 target, 100 steps |
| 3 | `not_reproduced` | toy-scale: 8-view discrete Radon operator on a 32x32 ellipse phantom |
| 4 | `partially_reproduced` | toy-scale: 4 of the paper's modalities; novel-view synthesis not attempted |

## Reproducing

```bash
uv run --project . python generate_evidence.py
uv run --project . python -m pytest tests -q
```

`generate_evidence.py` regenerates `evidence/evidence.json` and every page in
`pages/`; it is deterministic and byte-identical across runs.
