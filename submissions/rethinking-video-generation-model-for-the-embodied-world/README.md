---
title: RBench Artifact Reproduction
emoji: 🤖
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
tags:
  - paper-p5QSlnwume
  - icml2026-repro
---

# RBench artifact reproduction

This CPU-only reproduction audits immutable released artifacts for ICML 2026
Agent Repro Challenge paper `p5QSlnwume`. It recomputes prompt-manifest census
facts, validates separate paper-era and later leaderboard cohorts, checks a
displayed-mean consistency rule, and searches pinned source routes for the
three named failure modes.

## Computed results

- The nine prompt manifests contain 650 records across five task categories
  and four embodiment categories.
- The paper-era leaderboard has 25 valid, unique records. The later cohort has
  28; it prepends three models while preserving the original records.
- The later cohort has one material displayed-mean discrepancy:
  `LingBot-Video` reports `0.620` versus `0.614` recomputed from its nine
  displayed fields.
- The exact phrases “structural distortion,” “floating components,” and
  “key-action omission” were not found in the pinned allowlisted artifacts, so
  the failure-mode claim is inconclusive.

These are artifact measurements, not paper-reported values. The aggregation
rule was not source-traced; it is an internal consistency rule inferred from
rounded paper-era fields. Video generation was not rerun. Human correlation
was not reproduced.

## Reproduce locally

From this directory:

```bash
env UV_CACHE_DIR=/tmp/rbench-task-uv-cache uv run rbench-repro acquire --cache-dir .cache/rbench --manifest evidence/input-manifest.json --acquired-at 2026-07-27T15:17:46Z
env UV_CACHE_DIR=/tmp/rbench-task-uv-cache uv run rbench-repro validate-inputs --manifest evidence/input-manifest.json --cache-dir .cache/rbench
env ALL_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY= UV_CACHE_DIR=/tmp/rbench-task-uv-cache uv run rbench-repro audit --manifest evidence/input-manifest.json --cache-dir .cache/rbench --schema schema/evidence-v1.schema.json --output evidence/results.json --generated-at 2026-07-27T00:00:00+00:00 --tool-revision 1e92b10f4139c34012a5698031bdfe2beea95dae
env UV_CACHE_DIR=/tmp/rbench-task-uv-cache uv run rbench-repro validate evidence/results.json --schema schema/evidence-v1.schema.json
env UV_CACHE_DIR=/tmp/rbench-task-uv-cache uv run pytest -q
```

Only `acquire` uses the network. `validate-inputs`, `audit`, `validate`, tests,
and the Space app operate on local, hash-verified or committed files.

## Evidence

- `evidence/input-manifest.json`: immutable revisions, full source trees, and
  hashes for each allowlisted input.
- `evidence/results.json`: canonical claim-scoped results.
- `evidence/commands.json`: literal commands and environment fingerprints.
- `evidence/validation.json`: schema-validation record and results hash.
- `POSTER.md`: one-screen reviewer summary.

The Space is read-only and performs no runtime network fetches.
