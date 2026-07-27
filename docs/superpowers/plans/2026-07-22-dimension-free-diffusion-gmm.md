# Dimension-Free Diffusion/GMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, publish, and submit a deterministic CPU numerical audit of three claims from `arxiv:2504.05300v1`.

**Architecture:** An isolated NumPy project implements analytic isotropic-GMM operations and a seeded DDPM audit pipeline. One CLI emits a self-describing JSON evidence bundle and tidy CSV, which a fixed-structure Trackio logbook presents without conflating paper context and computed measurements.

**Tech Stack:** Python 3.11+, NumPy, pytest, Trackio, Hugging Face Hub.

## Global Constraints

- Do not modify or validate `submissions/nape/` in place.
- Write a failing test and observe the expected failure before production code.
- Use CPU only and USD 0.00 paid API cost.
- Pin the paper to `arxiv:2504.05300v1` and hash every downloaded input.
- Label all outputs numerical audits, not proof replacements.
- Preserve the canonical logbook page order from the live challenge guide.

---

### Task 1: Analytic GMM primitives

**Files:**
- Create: `submissions/dimension-free-diffusion-gmm/pyproject.toml`
- Create: `submissions/dimension-free-diffusion-gmm/src/diffusion_gmm_repro/model.py`
- Test: `submissions/dimension-free-diffusion-gmm/tests/test_model.py`

**Interfaces:**
- Produces: `IsotropicGMM`, `log_density(x)`, `score(x)`, and `score_jacobian_trace(x)` using NumPy arrays.

- [ ] Write tests for single-Gaussian identities, mixture shape validation, and finite-difference Jacobian agreement.
- [ ] Run `uv run --project submissions/dimension-free-diffusion-gmm pytest tests/test_model.py -q` and confirm missing-module failure.
- [ ] Implement normalized weights, stable responsibilities, analytic scores, and trace.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Deterministic DDPM audits

**Files:**
- Create: `submissions/dimension-free-diffusion-gmm/src/diffusion_gmm_repro/audit.py`
- Test: `submissions/dimension-free-diffusion-gmm/tests/test_audit.py`

**Interfaces:**
- Consumes: `IsotropicGMM`.
- Produces: `run_dimension_audit`, `run_score_error_audit`, and `run_jacobian_audit`, each returning JSON-serializable records.

- [ ] Write tests for deterministic seeds, dimension-invariant exact Gaussian behavior, monotone score-error degradation, and assumption-breaking controls.
- [ ] Run the focused tests and confirm missing-function failures.
- [ ] Implement the minimum seeded samplers and diagnostics needed by the tests.
- [ ] Re-run focused and full submission tests and confirm they pass.

### Task 3: Evidence CLI and bundle

**Files:**
- Create: `submissions/dimension-free-diffusion-gmm/src/diffusion_gmm_repro/cli.py`
- Create: `submissions/dimension-free-diffusion-gmm/tests/test_cli.py`
- Create: `submissions/dimension-free-diffusion-gmm/README.md`
- Create: `submissions/dimension-free-diffusion-gmm/evidence/provenance.json`

**Interfaces:**
- Consumes: the three audit functions.
- Produces: `evidence/results.json` and `evidence/measurements.csv` with exact claim IDs, observations, thresholds, statuses, provenance, and environment.

- [ ] Write a subprocess test that requires deterministic, schema-complete JSON/CSV and rejects invalid arguments.
- [ ] Run it and confirm the CLI is absent.
- [ ] Implement argument validation, atomic bundle generation, and README rerun commands.
- [ ] Run the CLI twice and assert byte-identical results, then run all submission tests.

### Task 4: Logbook, poster, and local validation

**Files:**
- Create: `submissions/dimension-free-diffusion-gmm/logbook/` through Trackio.
- Create: `submissions/dimension-free-diffusion-gmm/poster.html`
- Create: `submissions/dimension-free-diffusion-gmm/poster_embed.html`

**Interfaces:**
- Consumes: validated evidence JSON/CSV.
- Produces: the canonical Trackio pages, attached trace, pinned summary/poster, and conclusion.

- [ ] Attach the current scrubbed Codex session and verify the trace view.
- [ ] Populate the fixed page order and link all upstream/Hub resources.
- [ ] Render posterly output, pass `--strict-polish`, and create the navigable embed.
- [ ] Run submission pytest, root pytest, skill validation, pre-commit, and the authoritative logbook validator.

### Task 5: Deployment, submission, and judging

**Files:**
- Modify: `state/repro-loop.json`
- Modify: `docs/HANDOFF.md`

**Interfaces:**
- Consumes: the validated local logbook and committed evidence.
- Produces: a dedicated published Space, verified exact SHA, challenge submission, and bounded verdict polling.

- [ ] Refresh live paper/claim/queue/verdict status and stop if eligibility changed.
- [ ] Publish the dedicated Space and verify its deployed SHA equals the intended commit.
- [ ] Persist `validated`, `deployed`, `submitted`, and `judging` transitions with all external IDs.
- [ ] Poll within a finite round; record the exact claim-level verdict, improve once only if eligible, and complete or persist a blocker.
