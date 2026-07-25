# EEG-FM-Bench released-artifact audit

This project independently audits three structural claims against
`xw1216/EEG-FM-Bench@325398d7d057ecc1216fb3510d70c16eb60337cc` and
`arxiv:2508.17742v3`. It does not reproduce the GPU leaderboard or claim access
to the gated raw EEG datasets.

## Reproduce the evidence

From this directory:

```bash
uv sync --frozen
uv run python -m eeg_fm_bench_repro.cli \
  --cache-dir /tmp/icml-eeg-controller-validation-cache \
  --output-dir evidence
uv run pytest -q
```

The first run downloads the pinned repository archive and paper PDF and
verifies both against `evidence/provenance.json`. Later runs use the verified
cache. Every repository reuse rehashes the persisted archive bytes and
re-extracts the source tree from that verified archive; mutable tree markers
are not provenance authority. The PDF is likewise rehashed before use.

For controller and clean-clone validation, always pass an explicit external
`--cache-dir` such as the `/tmp` path above. If the option is omitted, the
honest default is the ignored project-local `.cache/upstream` directory. A
successful CLI run may register its resolved cache directory under a
user-scoped shared `/tmp` root so a later process can find it across distinct
`TMPDIR` values. That pointer and the referenced cache remain untrusted hints:
their bytes are verified against provenance before reuse. Set
`EEG_FM_BENCH_CACHE_DIR` to select a reusable cache explicitly, or
`EEG_FM_BENCH_REGISTRY_ROOT` to relocate the shared registry. The command
atomically writes `evidence/results.json` and `evidence/measurements.csv`.

## Evidence boundary

The bundle distinguishes paper-reported context from computed measurements:

- the dataset/paradigm census is computed from released source and config;
- preprocessing executes exact pinned method bodies through MNE on seeded
  synthetic EEG;
- harness evidence combines released-source branch checks with three
  deterministic audit-local CPU semantic smoke steps and is marked `partial`
  because no released baseline is executed;
- GPU performance and representation-analysis claims are explicitly
  `unavailable`.
