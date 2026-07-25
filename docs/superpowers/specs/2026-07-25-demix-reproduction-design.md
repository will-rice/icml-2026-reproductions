# DeMix Released-Artifact Audit Design

**Paper ID:** `uyRIOjFgOn`
**Paper:** *Decouple Searching from Training: Scaling Data Mixing via Model
Merging for Large Language Model Pre-training*
**Submission path:**
`submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/`
**Approved approach:** conservative, CPU-only audit of released pinned artifacts
**Estimated external cost:** USD 0.00

## Objective and evidence boundary

The submission will independently recompute only facts supported by small,
released artifacts. It will not treat paper tables, handwritten tensors,
invented targets, or scores computed from synthetic examples as reproduced
measurements.

The evidence boundary is:

- paper: `arXiv:2602.00747v3`, PDF SHA-256
  `85ea10da0925ee5bd284eeb3143c345129c74c320829dabd9d0ba4413acf55a3`;
- code: `Lucius-lsr/DeMix` commit
  `d0c945ca84d5632c6ed1bfe469337cf880757422`;
- dataset: `lucius1022/DeMix_Corpora` revision
  `82a2effc58eb79bec691280a4e4fc50be0968b1e`;
- primary computation input:
  `DeMix_reproduce/reference_models/sampled_mixture.json`, SHA-256
  `2be00152f98c44a740bc2f8e2098be3740ea2f1cd31b7158ade9d54c8e852dc2`.

The exact mixture manifest is vendored under `evidence/inputs/`. A compact
provenance file records source URLs, immutable revisions, acquisition commands,
file hashes, the upstream source-file hashes used in the audit, and the remote
release inventory. The evidence generator rejects an input whose bytes do not
match the pinned hash.

## Claim dispositions

### 1. Weighted linear model merging — `partial`

The released manifest permits independent computation of:

- the exact mixture and domain counts;
- each raw weight sum;
- normalized domain weights;
- which mixtures require normalization;
- non-negativity and positive-sum checks;
- the mismatch between 17 manifest entries and the 16 reference-model
  directories described by the release.

The pinned upstream merge configuration provides supporting source evidence
that normalized ratios are mapped to a weighted linear merge. No component
checkpoint is downloaded or merged, so the resulting observation is partial,
not a verification of model behavior.

### 2. Spearman proxy accuracy — `unavailable`

The release contains no OpenCompass benchmark-result CSVs or equivalent
per-mixture evaluation outputs from which Table 2 correlations can be
recomputed. The pinned `eval_merged/proxy_eval.py` has path placeholders and
uses `random.random()` in its sample data loader. Therefore it is not valid
evidence for reported correlations.

The bundle records no reproduced Spearman value. Paper-reported values may
appear only under `paper_context`, explicitly labeled as non-reproduced.

### 3. Mixture optimization and benchmarking — `unavailable`

The release has neither the final benchmark output needed to recompute Table 3
nor a CPU-feasible execution path for the released component models. The
bundle records no reproduced benchmark comparison. Paper-reported values may
appear only under `paper_context`.

## Resource rationale

The seven released 30B-token component checkpoints at `checkpoint-7500`
contain fourteen Safetensors shards totaling 48,176,346,736 bytes before model
merge outputs, evaluation datasets, or caches. Reproducing the model merge and
OpenCompass evaluation would require material storage and accelerator
resources. Downloading those checkpoints is outside this CPU-only audit and is
not necessary to establish that the released evaluation inputs are absent.

The release inventory is captured from the Hugging Face tree API at the pinned
revision. It records 1,469 files below `DeMix_reproduce/`, 16 reference-model
roots, seven component-model roots, no CSV files, and no released OpenCompass
result path. It also records every component shard's immutable LFS SHA-256 and
byte count without downloading the shards.

## Deterministic architecture

The current synthetic modules and calculator are removed. The replacement has
three parts:

1. `src/demix/artifacts.py` validates the pinned input, parses finite numeric
   weights, and deterministically derives released-manifest observations using
   the Python standard library.
2. `src/demix/pipeline.py` combines those observations with the checked-in
   provenance document and per-claim dispositions, then serializes canonical
   JSON with sorted keys, fixed indentation, UTF-8, and a final newline.
3. `app.py` is a read-only evidence and released-mixture inspector. It displays
   the committed bundle and computes normalization only from the pinned
   released manifest. It does not accept user-authored targets or produce
   scores.

The canonical regeneration command, run from the submission directory, is:

```bash
PYTHONPATH=src python -m demix.pipeline \
  --input evidence/inputs/sampled_mixture.json \
  --provenance evidence/provenance.json \
  --output evidence/bundle.json
```

Two consecutive runs and a run to a temporary output must be byte-identical to
the committed bundle.

## Bundle contract

`evidence/bundle.json` has:

- an overall `reproduction_status` of `partial`;
- immutable paper, code, dataset, and input provenance;
- the exact regeneration command and pinned environment;
- released-artifact observations derived from the vendored manifest;
- one record per target claim with `id`, `status`, `observation`,
  `input_artifacts`, and `limitations`;
- a separate `paper_context` object for non-reproduced reported values.

The schema forbids `verified` statuses and synthetic result keys such as
`macro_spearman`, `ground_truth`, `proxy_scores`, or `multi_seed_stability`.
An unavailable claim has no reproduced numeric result. A partial claim must
cite the released input artifact that supports its observation.

## Test-first acceptance criteria

Tests are written before implementation changes and must initially fail for
the current synthetic pipeline. They cover:

- rejection of modified or alternate input bytes by SHA-256;
- exact mixture count, domain order, raw sums, normalized values, and
  normalization classifications from the pinned manifest;
- rejection of non-finite, negative, empty, or zero-sum weights;
- correct overall and per-claim statuses;
- absence of synthetic scores, fabricated targets, and `verified`;
- presence of immutable revisions, hashes, acquisition commands, environment
  pins, and resource limitations;
- byte-identical CLI regeneration;
- a read-only app with the existing runtime contract
  `server_name="0.0.0.0", server_port=7860`;
- an integration startup check that observes the real Gradio listener on
  wildcard port 7860.

The required verification command is:

```bash
uv run --isolated --no-project \
  --with-requirements requirements-test.txt \
  python -m pytest -q
```

Scoped pre-commit validation excludes the archival NAPE submission, as required
by workspace policy.

## Runtime and dependency pins

The Space base is
`python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7`.
Direct Python dependencies are pinned to `gradio==6.20.0` and
`pytest==8.4.2`. NumPy and SciPy are removed because evidence computation uses
the standard library and no longer calculates synthetic tensor or correlation
examples.

The bundle records Python `3.12.11`, Gradio `6.20.0`, and pytest `8.4.2` as the
validated regeneration/test environment. Dependency resolution itself is not
treated as reproduced scientific evidence.

## Safety, licensing, and publication

The Hugging Face dataset card declares Apache-2.0. The inspected GitHub
revision has no detected license file, so its code is used only as cited,
read-only provenance and is not vendored. The vendored primary input is from
the licensed dataset revision.

This change does not deploy, publish, claim a challenge paper, or mutate
coordinator state. A later deployment requires a separate authorization,
live-status recheck, exact Space commit verification, and updates to
`state/repro-loop.json` and `docs/HANDOFF.md`.
