# EEG-FM-Bench Implementation Plan

> **For agentic workers:** Execute task-by-task with test-driven development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build, validate, publish, and submit a deterministic CPU audit of three structural claims from `arxiv:2508.17742v3` against the released benchmark repository `github:xw1216/EEG-FM-Bench@325398d7d057ecc1216fb3510d70c16eb60337cc`.

**Architecture:** An isolated Python project downloads the pinned upstream snapshot (paper PDF + repo tarball, sha256-verified), statically audits the dataset/paradigm census and preprocessing standardization, and smoke-runs the three fine-tuning strategies on synthetic EEG tensors. One CLI emits a self-describing JSON evidence bundle and tidy CSV; a fixed-structure Trackio logbook presents it without conflating paper context and computed measurements.

**Tech Stack:** Python 3.11+, NumPy, PyTorch (CPU), pytest, Trackio, Hugging Face Hub.

**Selected target claims (state `target_claims`):**

1. `fourteen-dataset-ten-paradigm-curation` — the benchmark curates 14 public datasets spanning 10 canonical EEG paradigms with standardized preprocessing and evaluation (paper Figure 1).
2. `standardized-preprocessing-reproducibility` — the standardized preprocessing pipeline (resampling, channel-name/montage standardization, windowing) is deterministic and consistent across datasets.
3. `three-strategy-evaluation-harness` — the harness supports frozen-backbone single-task fine-tuning, full-parameter single-task fine-tuning, and full-parameter multi-task fine-tuning (paper Figure 1).

The paper's frozen-backbone generalization-gap and fine-tuning-improvement performance claims (Tables 1-2) and qualitative analyses (Figure 2) require GPU foundation-model runs and are explicitly out of scope: marked `unavailable` in the logbook, never targeted.

## Global Constraints

- Do not modify or validate `submissions/nape/` in place.
- Write a failing test and observe the expected failure before production code.
- CPU only and USD 0.00 paid API cost.
- Pin the paper to `arxiv:2508.17742v3` and the repo to `325398d7d057ecc1216fb3510d70c16eb60337cc`; sha256 every downloaded input in `evidence/provenance.json`.
- Paper-reported values (Figure 1 lists, Tables 1-2) are context only; only code-computed outputs support claims.
- Label all outputs as audits of the released artifact; never claim leaderboard reproduction.

---

### Task 1: Pinned upstream snapshot + census audit

**Files:**
- Create: `submissions/eeg-fm-bench/pyproject.toml`
- Create: `submissions/eeg-fm-bench/src/eeg_fm_bench_repro/upstream.py`
- Create: `submissions/eeg-fm-bench/src/eeg_fm_bench_repro/census.py`
- Test: `submissions/eeg-fm-bench/tests/test_upstream.py`, `tests/test_census.py`
- Create: `submissions/eeg-fm-bench/evidence/provenance.json`

**Interfaces (pinned; Task 2 builds against them):**
- `upstream.ensure_repo_snapshot(cache_dir: Path) -> Path`: downloads `https://codeload.github.com/xw1216/EEG-FM-Bench/tarball/325398d7d057ecc1216fb3510d70c16eb60337cc`, verifies sha256 against `evidence/provenance.json`, extracts to `cache_dir`, returns the snapshot root. Idempotent and offline-after-first-fetch (cache hit skips download).
- `upstream.ensure_paper_pdf(cache_dir: Path) -> Path`: same pattern for `https://arxiv.org/pdf/2508.17742v3`.
- `census.run_census_audit(snapshot: Path) -> dict`: JSON-serializable claim record for claim 1; parses `data/processor/wrapper.py` `DATASET_SELECTOR`, `data/dataset/*.py` builder classes, and `assets/conf/**`; computes the repo-side census of the paper's Figure 1 datasets and paradigm coverage, embedding the paper-reported 14/10 list as labeled context.

- [ ] Write tests: sha256 mismatch raises; cache hit avoids re-download (monkeypatched fetch); census record contains claim id `fourteen-dataset-ten-paradigm-curation`, computed dataset/paradigm counts, per-dataset paradigm mapping, and paper-context list.
- [ ] Run focused tests; confirm missing-module failure.
- [ ] Implement; re-run until green. First fetch computes and records the real sha256 values in `evidence/provenance.json` (schema_version 1, acquisition commands, revisions, license notes).

### Task 2: Preprocessing + harness audits

**Files:**
- Create: `submissions/eeg-fm-bench/src/eeg_fm_bench_repro/preproc_audit.py`
- Create: `submissions/eeg-fm-bench/src/eeg_fm_bench_repro/harness_audit.py`
- Test: `submissions/eeg-fm-bench/tests/test_preproc_audit.py`, `tests/test_harness_audit.py`
- Local stand-ins only (discarded at merge): `pyproject.toml`, `upstream.py`, `__init__.py` implementing the pinned Task 1 interface, marked `# LOCAL STAND-IN`.

**Interfaces:**
- `preproc_audit.run_preproc_audit(snapshot: Path) -> dict`: claim id `standardized-preprocessing-reproducibility`. Runs the snapshot's preprocessing primitives (resampling to configured `fs`, `standardize_chs_names` montage standardization, windowing) on synthetic seeded EEG arrays for at least two distinct dataset builder configs; asserts deterministic identical outputs across repeated runs and consistent output structure (channel counts, window shapes) across configs.
- `harness_audit.run_harness_audit(snapshot: Path) -> dict`: claim id `three-strategy-evaluation-harness`. Verifies `freeze_encoder` and `multitask` config flags exist and are honored (encoder `requires_grad` False when frozen; multi-dataset loader path when multitask), and smoke-runs one CPU training step per strategy (frozen-backbone single-task, full-parameter single-task, full-parameter multi-task) with the smallest baseline model on synthetic tensors.

- [ ] Write tests: deterministic records across runs, all three strategies exercised, frozen mode provably freezes encoder params, claim ids and `kind: "numerical_audit"` present.
- [ ] Run focused tests; confirm missing-function failure.
- [ ] Implement; re-run until green. Total added CPU time under ~10 minutes.

### Task 3: Evidence CLI and bundle

**Files:**
- Create: `submissions/eeg-fm-bench/src/eeg_fm_bench_repro/cli.py`
- Create: `submissions/eeg-fm-bench/tests/test_cli.py`
- Create: `submissions/eeg-fm-bench/README.md`

**Interfaces:** consumes the three audit functions; produces `evidence/results.json` and `evidence/measurements.csv` with exact claim IDs, observations, thresholds, statuses, provenance, and environment (versions only, no env dumps).

- [ ] Subprocess test: byte-identical JSON/CSV across two runs; schema-complete; invalid args exit nonzero.
- [ ] Implement atomic bundle generation and README rerun commands.

### Task 4: Logbook, poster, and local validation

**Files:**
- Create: `submissions/eeg-fm-bench/logbook/` through Trackio.
- Create: `submissions/eeg-fm-bench/poster.html`, `poster_embed.html`.

- [ ] Canonical page order (Index, Executive summary, one page per target claim, Conclusion); pin summary then poster; link upstream/Hub resources; mark GPU-only claims `unavailable` explicitly.
- [ ] Render poster HTML and navigable embed (posterly tooling is unavailable on this host; record the deviation and perform an equivalent local static check).
- [ ] Run submission pytest, root pytest, skill validation, pre-commit.

### Task 5: Deployment, submission, and judging

**Files:** Modify `state/repro-loop.json`, `docs/HANDOFF.md`.

- [ ] Refresh live paper/claim/queue/verdict status; stop if eligibility changed.
- [ ] Publish a dedicated Space (`wrice/repro-eeg-fm-bench-*` per challenge naming), tags including `paper-vGeNaFHdET` and `icml2026-repro`; verify deployed SHA equals the intended commit; exercise the live evidence path.
- [ ] Persist `validated`, `deployed`, `submitted`, `judging` with finite poll budget; record exact claim-level verdict; improve once only if eligible; complete or persist a blocker.
