# Mechanistic Data Attribution Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing toy proposal into honest CPU evidence for influence ranking, causal intervention, and structural-pattern concentration.

**Architecture:** A deterministic tiny attention model consumes a pre-labelled synthetic corpus. Attribution never sees category labels; intervention evidence comes from actual matched retraining; pattern evidence comes from independent enrichment statistics.

**Tech Stack:** Python 3.12, PyTorch CPU, NumPy, SciPy, pytest, JSON, CSV.

## Global Constraints

- Write only `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/`.
- Pin arXiv `2601.21996v2` and Git commit `faa0890bc2d7961a0f177a422849b4e0801943c0`.
- CPU only, paid API cost USD 0.00, deterministic algorithms and seeds.
- Preserve pre-existing user/worker changes; review and commit only claim-relevant content.
- Produce direct `pages/reproduction.md`; never call Pythia-scale claims reproduced.

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
    a = train_model(make_corpus(7, 16), seed=11, steps=80)
    b = train_model(make_corpus(7, 16), seed=11, steps=80)
    assert state_dict_sha256(a) == state_dict_sha256(b)
```

- [ ] **Step 2: Run and confirm missing-module failures**

Run: `uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_corpus_model.py`

- [ ] **Step 3: Implement four predeclared categories and a minimal one-layer attention model**

Categories are repeated spans, LaTeX-like delimiters, HTML-like tags, and
matched random tokens. Training uses one CPU thread, deterministic algorithms,
fixed batches, and explicit steps.

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

Preserve raw scores before joining labels downstream. Validate positive finite
damping and reject duplicate sample IDs.

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
    assert all(r.sample_count == targeted.sample_count for r in rows if r.kind == "random_remove")
```

- [ ] **Step 2: Run and confirm the existing score-subtraction shortcut fails**

Run: `uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_intervention.py`

- [ ] **Step 3: Implement full retraining per intervention and seed**

Serialize model hashes, selected sample IDs, probe scores before/after, and
targeted-minus-random effect sizes.

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
    audit = pattern_enrichment(labels(), shuffled_scores(), .1, 1000, 5)
    assert audit.status != "verified"
```

- [ ] **Step 2: Run and verify the current category diagnostic fails**

Run: `uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_patterns.py`

- [ ] **Step 3: Implement exact enrichment and deterministic permutation statistics**

Keep all raw counts and null draws; status derives from a predeclared effect
threshold and interval, not category names.

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

**Interfaces:**
- Consumes: Tasks 1–4 raw rows.
- Produces: `evidence/results.json`, `measurements.csv`, `provenance.json`, `repro-bundle.tar.gz`, and `pages/reproduction.md`.

- [ ] **Step 1: Write failing schema and honesty tests**

```python
def test_results_bind_three_exact_live_claims(results):
    assert len(results["claims"]) == 3
    assert all(c["challenge_claim_sha256"] for c in results["claims"])

def test_page_states_scale_limit(page_text):
    assert len(page_text.strip()) >= 200
    assert "tiny" in page_text.lower()
    assert "Pythia-scale" in page_text
```

- [ ] **Step 2: Run and verify failure before bundle correction**

Run: `uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests/test_cli.py`

- [ ] **Step 3: Generate sorted deterministic evidence and the direct judge page**

Record exact source pins, blob hashes, commands, package versions, seeds,
tolerances, raw rows, and unreplicated claims.

- [ ] **Step 4: Verify two byte-identical runs and all tests**

```bash
uv run --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units python -m mechanistic_data_attribution_repro.cli
sha256sum submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/evidence/*
uv run --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units python -m mechanistic_data_attribution_repro.cli
sha256sum submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/evidence/*
uv run --with pytest --project submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units pytest -q submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/tests
```

Expected: hashes match and all tests pass.

- [ ] **Step 5: Commit**

```bash
git add submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units
git commit -m "feat(mda): publish causal attribution evidence"
```
