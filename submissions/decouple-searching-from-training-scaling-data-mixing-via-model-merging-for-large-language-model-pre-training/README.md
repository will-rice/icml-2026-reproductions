---
title: DeMix Released Artifact Audit
emoji: 🔀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
- icml2026-repro
- paper-uyRIOjFgOn
---

# DeMix released-artifact audit

This submission audits released evidence for **Decouple Searching from
Training: Scaling Data Mixing via Model Merging for Large Language Model
Pre-training** (paper ID `uyRIOjFgOn`). The overall reproduction status is
`partial`.

## Claim status

- Weighted linear model merging: `partial`. The exact released mixture
  manifest and pinned upstream merge configuration support independent checks
  of domain weights and normalization. No checkpoint was merged and no model
  behavior was measured.
- Spearman proxy accuracy (Table 2): `unavailable`. The release has no
  OpenCompass result CSVs or equivalent per-mixture benchmark outputs. The
  pinned upstream `eval_merged/proxy_eval.py` loader uses path placeholders and
  `random.random()`, so it cannot provide reproduced measurements.
- Mixture optimization and benchmarking (Table 3): `unavailable`. The final
  benchmark outputs needed for recomputation are not released, and the full
  model evaluation was not run.

Any paper-reported values in `evidence/bundle.json` are context only and are
explicitly marked `reproduced: false`.

## Pinned sources

- Paper: `arXiv:2602.00747v3`; PDF SHA-256
  `85ea10da0925ee5bd284eeb3143c345129c74c320829dabd9d0ba4413acf55a3`.
- Code: `Lucius-lsr/DeMix` commit
  `d0c945ca84d5632c6ed1bfe469337cf880757422`.
- Dataset: `lucius1022/DeMix_Corpora` revision
  `82a2effc58eb79bec691280a4e4fc50be0968b1e`.
- Primary input:
  `DeMix_reproduce/reference_models/sampled_mixture.json`; SHA-256
  `2be00152f98c44a740bc2f8e2098be3740ea2f1cd31b7158ade9d54c8e852dc2`.

The released manifest contains 17 mixture entries over seven domains, while
the release inventory contains 16 reference-model roots. Eight raw mixture
vectors do not sum exactly to one and require normalization. These are
artifact observations, not benchmark results.

The fourteen released Safetensors shards for the seven component checkpoints
total 48,176,346,736 bytes before merge outputs, evaluation datasets, and
caches. Full model merging and OpenCompass evaluation require material storage
and accelerator resources and are outside this CPU-only audit.

The dataset card declares Apache-2.0. No license file was detected at the
pinned GitHub commit, so upstream code is cited as read-only provenance and is
not vendored here. The vendored dataset manifest is covered by the scoped
notice in `THIRD_PARTY_NOTICES.md`; the complete license is packaged at
`LICENSES/Apache-2.0.txt`.

## Regenerate evidence

From this directory:

```bash
env PYTHONPATH=src UV_CACHE_DIR=/tmp/demix-repro-uv-cache \
  uv run --isolated --no-project --python 3.12.11 \
  python -m demix.pipeline \
  --input evidence/inputs/sampled_mixture.json \
  --provenance evidence/provenance.json \
  --output evidence/bundle.json
```

The command validates the input SHA-256 and immutable provenance contract,
recomputes manifest observations with decimal arithmetic, and writes canonical
JSON. Repeated runs are byte-identical.

## Test

The Space runtime uses Python 3.12.11 and Gradio 6.20.0. Run the complete suite
in an isolated environment:

```bash
uv run --isolated --no-project \
  --with-requirements requirements-test.txt \
  python -m pytest -q
```

The Space is a read-only evidence and released-mixture viewer. It listens on
`0.0.0.0:7860` in the Docker runtime.
