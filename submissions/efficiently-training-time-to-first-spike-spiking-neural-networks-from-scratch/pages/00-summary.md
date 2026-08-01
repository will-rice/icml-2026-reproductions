# ETTFS SNN Reproduction Summary

Reproduction of **"Efficiently Training Time-to-First-Spike Spiking Neural
Networks from Scratch"** (paper `3EcT46wsdc`, arXiv:2410.23619).

Every number on these pages is computed by this repository on CPU with pinned
seeds: integrate-and-fire dynamics are simulated step by step and the ablation
networks are trained from scratch. No value is copied from the paper into a
measurement field. Dataset-scale claims that need GPU training are reported as
`unreplicated` rather than asserted.

## Claim status

| Claim | Status | Scale of the evidence |
| --- | --- | --- |
| 1 | `reproduced` | toy-scale simulation (6 layers x 128 units, 32 time-steps) |
| 2 | `partially_reproduced` | toy-scale: four synthetic input regimes, not the paper's four datasets |
| 3 | `reproduced` | exact numerical property check on simulated post-synaptic currents |
| 4 | `unreplicated` | requires MNIST/Fashion-MNIST/CIFAR/DVS-Gesture training runs on GPU |
| 5 | `partially_reproduced` | toy-scale synthetic 3-class task trained from scratch, not Fashion-MNIST |

Status counts: {"partially_reproduced": 2, "reproduced": 2, "unreplicated": 1}

## Reproducing

```bash
uv run --project . python -m ettfs_snn.evidence
uv run --project . python -m pytest tests -q
```

The first command regenerates `evidence/bundle.json` and every page in
`pages/`; it is deterministic and byte-identical across runs.
