# GUDA: Counterfactual Group-wise Training Data Attribution

Paper: `5f0gw9YpZC`

Attempt: `3ffbc4da-8f54-4a81-b70e-8103fe8eda1d`

This page exposes the independently generated GUDA evidence bundle for the
judge-visible Space logbook. The evidence pins `sony/guda` at commit
`9fcf10cc4362199efc4f975e4a950df826fada07` and records CPU-only source,
metadata, and synthetic ranking checks. It does not present paper-reported
CIFAR-10 or Stable Diffusion benchmark values as reproduced measurements.

## Claim Results

| claim_sha256 | status | evidence summary |
| --- | --- | --- |
| `106c8d047410261b6f3b2038b498207ec9be867e354c567664d5f4cdd33c0917` | toy | Pinned source contains LOGO/GUDA scoring paths and deterministic toy ranking evidence, but no diffusion checkpoint was trained in this CPU-only run. |
| `8cfe641882a49b33f0db50a94de87d4f60cbdda050fc34364c2267d024e9254d` | inconclusive | CIFAR-10 LOGO, GUDA-U, ranking, and summary source paths are present, but full CIFAR-10 training/checkpoints/results were not distributed or recomputed. |
| `f2148792206d4cebe4304f05bcc130d9f83a77acab08a5df4e5b21d69930e619` | toy | UnlearnCanvas prompt metadata and scoring/ranking source paths are verified; Stable Diffusion fine-tuning and image attribution metrics were not recomputed. |
| `9c7f6323ad0541e5afe3f24417bbbe62c2510b3a1bd286cbae41f473122bd4ed` | toy | Weighted-style-select and uniform/weighted ablation configurations are distinct and internally consistent, without running SD training. |
| `dcd3d556206571fbe9121fac83ded756c044b9b7ff14ff48058d3c319b7d1338` | inconclusive | Timing/cost source and reproduction artifact paths exist, but the 100x and 5.9x wall-clock ratios were not recomputed from generated checkpoints. |
| `1821ab64dbe97bf7121de7694c3a3715844bb37300235ffb5b4451e878ba17ab` | inconclusive | The CPU source audit found noise/partition-related source references, but no runnable 5% noisy-partition robustness artifact was recomputed. |

## Reproduction Commands

```bash
python generate_evidence.py --output evidence/bundle.json
pytest -q
```

The machine-readable record is `evidence/bundle.json`.
