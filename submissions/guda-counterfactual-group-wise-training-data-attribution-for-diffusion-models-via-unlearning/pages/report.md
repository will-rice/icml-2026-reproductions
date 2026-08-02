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
| `8cfe641882a49b33f0db50a94de87d4f60cbdda050fc34364c2267d024e9254d` | toy | CIFAR-10 LOGO, GUDA-U, ranking, and summary source paths are present. A deterministic CPU ranking proxy gives GUDA top-1 accuracy 1.0 while semantic and gradient baselines miss the head group; full CIFAR-10 diffusion training/checkpoints were not recomputed. |
| `f2148792206d4cebe4304f05bcc130d9f83a77acab08a5df4e5b21d69930e619` | toy | UnlearnCanvas prompt metadata and scoring/ranking source paths are verified; Stable Diffusion fine-tuning and image attribution metrics were not recomputed. |
| `9c7f6323ad0541e5afe3f24417bbbe62c2510b3a1bd286cbae41f473122bd4ed` | toy | Weighted-style-select and uniform/weighted ablation configurations are distinct and internally consistent. A deterministic CPU anchor-ranking proxy ranks the weighted anchor above the uniform anchor without running SD training. |
| `dcd3d556206571fbe9121fac83ded756c044b9b7ff14ff48058d3c319b7d1338` | toy | Timing/cost source and reproduction artifact paths exist. CPU cost accounting records the relative number of training/unlearning runs implied by the LOGO-vs-GUDA setup without copying the paper's 100x or 5.9x wall-clock ratios. |
| `1821ab64dbe97bf7121de7694c3a3715844bb37300235ffb5b4451e878ba17ab` | toy | The CPU source audit found noise/partition-related source references. A deterministic 5% noisy-partition ranking proxy preserves the same head group as the clean proxy, but no full CIFAR-10 robustness experiment was rerun. |

## Computed CPU Observations

- CIFAR-10 proxy: GUDA top-1 accuracy `1.0`, MRR `1.0`, nDCG@3 `1.0`; semantic baseline top-1 `0.0`; gradient baseline top-1 `0.0`.
- UnlearnCanvas metadata: train prompts `24000`, eval prompts `1200`, styles `60`, objects `20`, paper-faithful style count `16`.
- Anchor proxy: weighted-anchor top-1 `1.0`, nDCG@3 `1.0`; uniform-anchor top-1 `0.0`, nDCG@3 `0.9464556027366066`.
- Cost accounting: CIFAR-10 group count `10`, UnlearnCanvas style count `16`, relative training runs versus LOGO proxy `0.55`, paper wall-clock values used `false`.
- Noisy partition proxy: clean top-1 `1.0`, 5% noisy top-1 `1.0`, both nDCG@3 `1.0`.

## Reproduction Commands

```bash
python generate_evidence.py --output evidence/bundle.json
pytest -q
```

The machine-readable record is `evidence/bundle.json`.
