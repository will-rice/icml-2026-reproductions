# Mechanistic Data Attribution Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing toy proposal into honest CPU evidence for influence ranking, causal intervention, and structural-pattern concentration.

**Architecture:** A deterministic tiny attention model consumes a pre-labelled synthetic corpus. Attribution never sees category labels; intervention evidence comes from actual matched retraining; pattern evidence comes from independent enrichment statistics.

**Tech Stack:** Python 3.12, PyTorch CPU, NumPy, SciPy, pytest, JSON, CSV.

## Global Constraints

- Attempt `3a44d506-d7a0-4bb8-abf7-d51a55c0018c`, paper `PQaxfoEcRc`,
  admitted snapshot
  `09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199`.
- Write only `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/`.
- Pin arXiv `2601.21996v2` and Git commit `faa0890bc2d7961a0f177a422849b4e0801943c0`.
- CPU only, paid API cost USD 0.00, deterministic algorithms and seeds.
- Preserve pre-existing user/worker changes; review and commit only claim-relevant content.
- Produce direct `pages/reproduction.md`, integrate it into `app.py` and
  `poster.html`, and never call Pythia-scale claims reproduced.
- Pin Gradio exactly to `6.20.0` in `pyproject.toml`, `uv.lock`, and README
  Space metadata.
- Preserve the exact five claim strings, hashes, and order from the approved
  design. Claims 4–5 must remain present as `unreplicated`.
- Local statuses are `supported`, `not-supported`, `limited`, or
  `unreplicated`; official `verified`/`falsified`/`toy` labels are forbidden.

### Task 0: Immutable bindings and upstream manifest

**Files:**
- Create: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/evidence/inputs/upstream_manifest.json`
- Create: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/src/mechanistic_data_attribution_repro/provenance.py`
- Create: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_provenance.py`

- [ ] **Step 1: Write failing immutable-binding tests**

Hard-code the five exact `(claim_text, sha256)` constants and their order from
the approved design. Recompute every digest and assert the attempt, paper,
snapshot, arXiv revision, and Git commit. Do not test a supplied string against
a supplied hash.

- [ ] **Step 2: Write failing tamper tests**

The tests alter one byte, remove one manifest item, add one unmanifested input,
and change one Git blob. Each case must raise `IntegrityError` before evidence
generation.

- [ ] **Step 3: Implement fail-closed acquisition and verification**

Acquire the exact arXiv v2 bytes and only the used repository files from commit
`faa0890...`; record URL, byte count, SHA-256, and Git blob where applicable in
the committed manifest. Acquisition verifies temporary bytes before atomic
installation. All later commands are offline and call the verifier first.

- [ ] **Step 4: Run focused tests and commit**

---

### Task 1: Independent Corpus Labels and Tiny Model

**Files:**
- Create: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/src/mechanistic_data_attribution_repro/corpus.py`
- Create: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/src/mechanistic_data_attribution_repro/tiny_model.py`
- Create: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_corpus_model.py`

**Interfaces:**
- Produces: `make_corpus(seed: int, samples_per_category: int) -> Corpus`, with immutable tokens, category, and sample ID.
- Produces: `train_model(corpus: Corpus, seed: int, steps: int) -> TrainedModel`.
- Produces: `probe_scores(model: TrainedModel) -> ProbeScores`.

- [ ] **Step 1: Write failing tests**

```python
def test_labels_are_determined_only_by_token_constructor():
    corpus = make_corpus(seed=7, samples_per_category=16)
    assert corpus.labels == labels_from_tokens(corpus.tokens)
    assert "influence" not in inspect.getsource(labels_from_tokens)

def test_training_is_byte_deterministic():
    a = train_model(make_corpus(7, 64), seed=11, steps=160)
    b = train_model(make_corpus(7, 64), seed=11, steps=160)
    assert state_dict_sha256(a) == state_dict_sha256(b)
```

- [ ] **Step 2: Run and confirm missing-module failures**

Run: `uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_corpus_model.py`

- [ ] **Step 3: Implement four predeclared categories and a minimal one-layer attention model**

Categories are repeated spans, LaTeX-like delimiters, HTML-like tags, and
matched random tokens, with 64 samples per category, vocabulary 64, and
sequence length 16. The model has one attention layer, two heads, width 32,
and feed-forward width 64. Training uses one CPU thread, deterministic
algorithms, AdamW at `1e-3`, batch size 16, 160 fixed batches, and seeds
`[11, 23, 37, 53, 71]`. Each probe has 256 held-out sequences generated only
from seed `seed + 10_000`. Previous-token positives are 128 iid-uniform
vocabulary-64 sequences modified at four uniformly selected nonadjacent
positions to copy their predecessor; its 128 negatives are iid uniform
conditioned on no adjacent equality. Induction positives are 128 sequences
whose iid-uniform distinct-token first eight positions are copied into
positions 8–15; its 128 negatives are iid uniform conditioned on unequal
halves. Reject every held-out sequence whose token hash occurs in training.
Previous-token accuracy is the fraction of
positions 1–15 where head 0's attention argmax equals `t-1`. Induction-copy
accuracy uses period-eight sequences with independently sampled distinct
first-half tokens and is the fraction of positions 8–15 where head 1's
attention argmax equals `t-8`. These exact alignments define toy-scale head
emergence; the held-out generator rejects overlap with training sample IDs.

- [ ] **Step 4: Run the focused tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units
git commit -m "feat(mda): add deterministic labelled corpus and tiny model"
```

### Task 2: Influence Ranking Without Label Leakage

**Files:**
- Modify: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/src/mechanistic_data_attribution_repro/attribution.py`
- Modify: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_attribution.py`

**Interfaces:**
- Consumes: trained model and token/loss tensors, never category labels.
- Produces: `influence_scores(model: TrainedModel, samples: Tensor, probe: ProbeSpec, damping: float) -> list[InfluenceRow]`.

- [ ] **Step 1: Add failing exactness and leakage tests**

```python
def test_influence_matches_finite_difference_fixture():
    rows = influence_scores(scalar_fixture_model(), scalar_samples(), induction_probe(), damping=1e-3)
    assert np.allclose([r.score for r in rows], finite_difference_scores(), rtol=1e-4)

def test_influence_api_has_no_labels_parameter():
    assert "labels" not in inspect.signature(influence_scores).parameters
```

- [ ] **Step 2: Run and verify the current proposal fails at least one new assertion**

Run: `uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_attribution.py`

- [ ] **Step 3: Implement damped Hessian-vector influence and stable sample IDs**

Freeze the transformer and fit two float64 32-dimensional logistic readouts.
For head 0, project its value slice through the corresponding `W_O` slice and
average the resulting 32-vectors over positions 1–15; do the same for head 1
over positions 8–15. Binary targets are the two probe constructors' exact
positive/negative labels, never semantic category labels. Concatenate the
readout parameters as the 64-vector `theta`. For mean binary cross-entropy
`ell_i`, define
`H = grad²_theta mean_i ell_i` and
`g = grad_theta L_probe`, where `L_probe` is the equally weighted mean BCE
over the two disjoint 256-example held-out sets. Preserve
`I_i = -gᵀ(H + 1e-3 I)⁻¹ grad_theta ell_i / n` before joining labels. Build
the exact dense 64-by-64 Hessian and solve with `torch.linalg.solve`; reject
non-float64 operands, non-finite values, nonpositive damping, or duplicate
sample IDs.

- [ ] **Step 4: Run Tasks 1–2 tests**

Run: `uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_corpus_model.py submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_attribution.py`

- [ ] **Step 5: Commit**

```bash
git add submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units
git commit -m "fix(mda): compute label-independent influence rankings"
```

### Task 3: Actual Targeted-versus-Random Retraining

**Files:**
- Modify: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/src/mechanistic_data_attribution_repro/intervention.py`
- Modify: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_intervention.py`

**Interfaces:**
- Consumes: corpus, influence rows, intervention fraction, and seed list.
- Produces: `run_interventions(...) -> list[InterventionRow]`, including baseline, targeted removal, targeted augmentation, and matched random controls.

- [ ] **Step 1: Write failing tests proving retraining and matched controls**

```python
def test_every_intervention_retrains_from_same_initialization(rows):
    assert {r.initial_state_sha256 for r in rows} == {rows[0].initial_state_sha256}
    assert all(r.final_state_sha256 != r.initial_state_sha256 for r in rows)

def test_random_controls_match_targeted_sample_count(rows):
    targeted = next(r for r in rows if r.kind == "targeted_remove")
    controls = [r for r in rows if r.kind == "random_remove"]
    assert all(r.sample_count == targeted.sample_count for r in controls)
    assert all(r.training_fingerprint == targeted.training_fingerprint for r in controls)

def test_augmentation_controls_match_every_training_condition(rows):
    targeted = next(r for r in rows if r.kind == "targeted_augment")
    controls = [r for r in rows if r.kind == "random_augment"]
    assert all(r.sample_count == targeted.sample_count for r in controls)
    assert all(r.training_fingerprint == targeted.training_fingerprint for r in controls)
```

- [ ] **Step 2: Run and confirm the existing score-subtraction shortcut fails**

Run: `uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_intervention.py`

- [ ] **Step 3: Implement full retraining per intervention and seed**

Remove or duplicate exactly the top 32 samples (12.5%). Matched random
removal/augmentation uses the same count and the identical initialization,
ordered batch-index schedule, AdamW configuration, and 160-step count for each
of
`[11, 23, 37, 53, 71]`. Serialize model hashes, selected sample IDs, probe
scores before/after, and targeted-minus-random effect sizes. A canonical
`training_fingerprint` hashes the initialization, complete ordered batches,
optimizer hyperparameters, step count, and exact ordered training sample IDs
with multiplicity. Thus intervention count is part of the fingerprint rather
than merely a separate field.

For each probe define paired effects as
`random_remove_after - targeted_remove_after` and
`targeted_augment_after - random_augment_after`. Use exactly 10,000 paired
bootstrap resamples with seed 20260728. Claim 2 is `supported` only when all
four probe/direction comparisons have point effect at least 0.02 and a 95%
bootstrap lower endpoint above zero. Valid null/reversed results are
`not-supported`; convergence/integrity loss is `limited`. Serialize raw pairs,
bootstrap seed/count, point effects, and intervals.

- [ ] **Step 4: Run the focused and cumulative tests**

Run: `uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests`

- [ ] **Step 5: Commit**

```bash
git add submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units
git commit -m "feat(mda): measure causal sample interventions"
```

### Task 4: Non-Circular Pattern Enrichment

**Files:**
- Modify: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/src/mechanistic_data_attribution_repro/patterns.py`
- Modify: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_patterns.py`

**Interfaces:**
- Consumes: independent labels and influence rows joined only by sample ID.
- Produces: `pattern_enrichment(labels, influences, top_fraction, permutations, seed) -> PatternAudit`.

- [ ] **Step 1: Write failing permutation and shuffled-null tests**

```python
def test_enrichment_reports_raw_counts_and_interval(audit):
    assert sum(audit.top_counts.values()) == audit.top_n
    assert audit.permutation_interval[0] <= audit.null_mean <= audit.permutation_interval[1]

def test_shuffled_scores_do_not_report_structural_enrichment():
    audit = pattern_enrichment(labels(), shuffled_scores(), .125, 2000, 5)
    assert audit.status != "supported"
```

- [ ] **Step 2: Run and verify the current category diagnostic fails**

Run: `uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_patterns.py`

- [ ] **Step 3: Implement exact enrichment and deterministic permutation statistics**

Use the fixed top fraction `.125`, 2,000 seeded permutations, and odds ratio.
Keep raw counts and permutation summaries. Report `supported` only when the
odds ratio is greater than 1 and its 95% permutation interval excludes 1;
otherwise report `not-supported` or `limited`. The threshold is immutable and
does not depend on category names or observed results.

- [ ] **Step 4: Run the full project tests**

Run: `uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests`

- [ ] **Step 5: Commit**

```bash
git add submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units
git commit -m "feat(mda): audit structural influence enrichment"
```

### Task 5: Evidence, Provenance, and Root Page

**Files:**
- Modify: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/src/mechanistic_data_attribution_repro/cli.py`
- Modify: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_cli.py`
- Create: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/pages/reproduction.md`
- Modify: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/README.md`
- Modify: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/pyproject.toml`
- Modify: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/uv.lock`
- Modify: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/app.py`
- Modify: `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/poster.html`

**Interfaces:**
- Consumes: Tasks 1–4 raw rows.
- Produces: `evidence/results.json`, `measurements.csv`, `provenance.json`, `repro-bundle.tar.gz`, and `pages/reproduction.md`.

- [ ] **Step 1: Write failing schema and honesty tests**

```python
def test_results_bind_three_exact_live_claims(results):
    assert [(c["challenge_claim"], c["challenge_claim_sha256"]) for c in results["claims"]] == EXPECTED_FIVE_CLAIMS
    assert [c["status"] for c in results["claims"][3:]] == ["unreplicated", "unreplicated"]

def test_page_states_scale_limit(page_text):
    assert len(page_text.strip()) >= 200
    assert "tiny" in page_text.lower()
    assert "Pythia-scale" in page_text

def test_offline_root_judge_surfaces(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(socket, "socket", forbidden_socket)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)
    monkeypatch.setattr(httpx, "Client", ForbiddenClient)
    app = import_fresh_app()
    assert app_reads("pages/reproduction.md")
    assert judge_text(app, "poster.html") == expected_committed_summary()
    metadata = read_space_metadata()
    assert metadata["sdk"] == "gradio"
    assert metadata["sdk_version"] == "6.20.0"
    assert metadata["app_file"] == "app.py"
    assert {"paper-PQaxfoEcRc", "icml2026-repro"} <= set(metadata["tags"])
    assert project_requirement("gradio") == "==6.20.0"
    assert locked_version("gradio") == "6.20.0"
```

- [ ] **Step 2: Run and verify failure before bundle correction**

Run: `uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_cli.py`

- [ ] **Step 3: Generate sorted deterministic evidence and the direct judge page**

Record exact source pins, blob hashes, commands, package versions, seeds,
tolerances, raw rows, and both unreplicated claims. README metadata must set
`sdk: gradio`, pin its SDK version, name `app.py`, and include
`paper-PQaxfoEcRc` and `icml2026-repro`. `app.py` and `poster.html` render the
same committed result/page without recomputation.

- [ ] **Step 4: Verify two byte-identical runs and all tests**

```bash
uv run --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units python -m mechanistic_data_attribution_repro.cli --output /tmp/mda-run-a
uv run --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units python -m mechanistic_data_attribution_repro.cli --output /tmp/mda-run-b
diff -r /tmp/mda-run-a /tmp/mda-run-b
cmp /tmp/mda-run-a/repro-bundle.tar.gz /tmp/mda-run-b/repro-bundle.tar.gz
uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests
```

The archive writer sorts member paths and fixes mode, uid, gid, uname, gname,
and mtime. Expected: both directories and tarballs are byte-identical and all
tests pass.

- [ ] **Step 5: Commit**

```bash
git add submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units
git commit -m "feat(mda): publish causal attribution evidence"
```
