# RACO Score Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the toy RACO implementation with deterministic evidence for four CPU-feasible algorithm and theorem claims.

**Architecture:** Focused pure-Python/PyTorch modules implement pairwise losses, exact two-objective CAGrad-Clip, and finite smooth-objective audits. A single generator serializes measurements, provenance, and the direct root judge page.

**Tech Stack:** Python 3.12, PyTorch CPU, NumPy, pytest, JSON, CSV.

## Global Constraints

- Write only `submissions/reward-free-alignment-for-conflicting-objectives/`.
- Pin arXiv `2602.02495v3` and Git commit `84a943c34f38520c7e0c9dd3066517c111b3c8fa`.
- Use CPU only, paid API cost USD 0.00, and deterministic seeds.
- Never label paper-reported empirical values as reproduced.
- Produce direct `pages/reproduction.md` plus machine-readable evidence.

---

### Task 1: Objective-Specific Pairwise Losses

**Files:**
- Create: `submissions/reward-free-alignment-for-conflicting-objectives/src/reward_free_alignment/pairwise.py`
- Create: `submissions/reward-free-alignment-for-conflicting-objectives/tests/test_pairwise.py`
- Modify: `submissions/reward-free-alignment-for-conflicting-objectives/pyproject.toml`

**Interfaces:**
- Produces: `pairwise_logistic_loss(chosen_logp: Tensor, rejected_logp: Tensor, ref_chosen_logp: Tensor, ref_rejected_logp: Tensor, beta: float) -> Tensor`.
- Produces: `objective_losses(batches: Sequence[PairwiseBatch], beta: float) -> Tensor`.

- [ ] **Step 1: Write the failing tests**

```python
def test_pairwise_loss_matches_closed_form():
    loss = pairwise_logistic_loss(tensor([-.2]), tensor([-.8]), tensor([-.4]), tensor([-.6]), 0.5)
    expected = -logsigmoid(tensor([0.2]))
    assert torch.allclose(loss, expected)

def test_objectives_remain_separate():
    losses = objective_losses([fixture_a(), fixture_b()], beta=0.5)
    assert losses.shape == (2,)
    assert losses[0] != losses[1]
```

- [ ] **Step 2: Run the tests and verify the import/function failures**

Run: `uv run --with pytest --project submissions/reward-free-alignment-for-conflicting-objectives pytest -q submissions/reward-free-alignment-for-conflicting-objectives/tests/test_pairwise.py`

Expected: FAIL because `reward_free_alignment.pairwise` does not exist.

- [ ] **Step 3: Implement stable log-sigmoid losses without scalarizing objectives**

Use `-torch.nn.functional.logsigmoid(beta * ((chosen_logp - rejected_logp) - (ref_chosen_logp - ref_rejected_logp)))`, validate equal shapes and positive finite `beta`, then stack one mean loss per objective.

- [ ] **Step 4: Run the focused tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add submissions/reward-free-alignment-for-conflicting-objectives
git commit -m "feat(raco): implement objective-specific pairwise losses"
```

### Task 2: Exact CAGrad-Clip Update

**Files:**
- Create: `submissions/reward-free-alignment-for-conflicting-objectives/src/reward_free_alignment/cagrad_clip.py`
- Create: `submissions/reward-free-alignment-for-conflicting-objectives/tests/test_cagrad_clip.py`
- Modify: `submissions/reward-free-alignment-for-conflicting-objectives/src/reward_free_alignment/raco.py`

**Interfaces:**
- Consumes: a sequence of flattened objective gradients from Task 1.
- Produces: `solve_two_objective_alpha(g1: Tensor, g2: Tensor, c: float) -> float`.
- Produces: `cagrad_clip(gradients: Sequence[Tensor], weights: Tensor, c: float) -> CAGradResult`, where `CAGradResult` contains `gradient`, `coefficients`, `unclipped_coefficients`, `clip_bound`, and `stationarity_residual`.

- [ ] **Step 1: Write failing source-parity and edge-case tests**

```python
def test_two_objective_solver_matches_upstream_fixture():
    result = cagrad_clip([tensor([1., 0.]), tensor([0., 1.])], tensor([.7, .3]), c=.4)
    assert torch.allclose(result.gradient, expected_upstream_gradient(), atol=1e-10)
    assert result.coefficients.abs().max() <= result.clip_bound + 1e-12

def test_collinear_gradients_are_finite():
    result = cagrad_clip([tensor([1., 2.]), tensor([2., 4.])], tensor([.5, .5]), c=.4)
    assert torch.isfinite(result.gradient).all()
```

- [ ] **Step 2: Run focused tests and confirm they fail against the fixed-alpha toy code**

Run: `uv run --with pytest --project submissions/reward-free-alignment-for-conflicting-objectives pytest -q submissions/reward-free-alignment-for-conflicting-objectives/tests/test_cagrad_clip.py`

- [ ] **Step 3: Port the exact pinned solver and independent clipping equations**

Record upstream blob paths and SHA-256 in module constants. Reject unsupported objective counts instead of silently averaging them.

- [ ] **Step 4: Run Tasks 1–2 tests**

Run: `uv run --with pytest --project submissions/reward-free-alignment-for-conflicting-objectives pytest -q submissions/reward-free-alignment-for-conflicting-objectives/tests/test_pairwise.py submissions/reward-free-alignment-for-conflicting-objectives/tests/test_cagrad_clip.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add submissions/reward-free-alignment-for-conflicting-objectives
git commit -m "fix(raco): reproduce exact CAGrad-Clip update"
```

### Task 3: Theorem 3.1 and 3.2 Audits

**Files:**
- Create: `submissions/reward-free-alignment-for-conflicting-objectives/src/reward_free_alignment/theorem_audit.py`
- Create: `submissions/reward-free-alignment-for-conflicting-objectives/tests/test_theorem_audit.py`

**Interfaces:**
- Consumes: `cagrad_clip` from Task 2.
- Produces: `audit_pareto_criticality(grid: AuditGrid) -> list[AuditRow]`.
- Produces: `compare_two_objective_rates(grid: AuditGrid) -> list[RateRow]`.

- [ ] **Step 1: Write failing tests for preconditions, stationarity, and rate comparison**

```python
def test_pareto_audit_records_every_precondition():
    rows = audit_pareto_criticality(small_convex_grid())
    assert rows
    assert all(r.smoothness_bound > 0 and r.step_size > 0 for r in rows)
    assert max(r.stationarity_residual for r in rows) < 1e-8

def test_clipped_rate_is_strictly_better_on_witness_family():
    rows = compare_two_objective_rates(theorem_3_2_witness_grid())
    assert all(r.clipped_iterations < r.unclipped_iterations for r in rows)
```

- [ ] **Step 2: Run and verify missing-interface failures**

Run: `uv run --with pytest --project submissions/reward-free-alignment-for-conflicting-objectives pytest -q submissions/reward-free-alignment-for-conflicting-objectives/tests/test_theorem_audit.py`

- [ ] **Step 3: Implement deterministic analytic-gradient sweeps**

Use seeded quadratic/nonconvex smooth objectives, record exact stopping rules, and serialize counterexamples rather than filtering them.

- [ ] **Step 4: Run the full project tests**

Run: `uv run --with pytest --project submissions/reward-free-alignment-for-conflicting-objectives pytest -q submissions/reward-free-alignment-for-conflicting-objectives/tests`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add submissions/reward-free-alignment-for-conflicting-objectives
git commit -m "feat(raco): audit Pareto convergence claims"
```

### Task 4: Evidence Bundle and Judge Page

**Files:**
- Modify: `submissions/reward-free-alignment-for-conflicting-objectives/src/reward_free_alignment/generate_evidence.py`
- Create: `submissions/reward-free-alignment-for-conflicting-objectives/tests/test_evidence.py`
- Create: `submissions/reward-free-alignment-for-conflicting-objectives/pages/reproduction.md`
- Modify: `submissions/reward-free-alignment-for-conflicting-objectives/README.md`

**Interfaces:**
- Consumes: Tasks 1–3 audit rows.
- Produces: `evidence/results.json`, `evidence/measurements.csv`, `evidence/provenance.json`, and `pages/reproduction.md`.

- [ ] **Step 1: Write failing schema, determinism, and honesty tests**

```python
def test_bundle_binds_all_four_live_claims(generated_bundle):
    assert len(generated_bundle["claims"]) == 4
    assert all(c["challenge_claim_sha256"] for c in generated_bundle["claims"])

def test_page_does_not_claim_gpu_results(root_page):
    assert len(root_page.strip()) >= 200
    assert "unreplicated" in root_page.lower()
    assert "BeaverTails" in root_page
```

- [ ] **Step 2: Run and verify failures before generator changes**

Run: `uv run --with pytest --project submissions/reward-free-alignment-for-conflicting-objectives pytest -q submissions/reward-free-alignment-for-conflicting-objectives/tests/test_evidence.py`

- [ ] **Step 3: Generate sorted, deterministic outputs and a direct root page**

Hash every pinned input; include commands, Python/package versions, seeds, tolerances, and measured results. Never put paper-reported table values in measurement fields.

- [ ] **Step 4: Verify reproducibility and the complete suite**

Run:

```bash
uv run --project submissions/reward-free-alignment-for-conflicting-objectives python -m reward_free_alignment.generate_evidence
sha256sum submissions/reward-free-alignment-for-conflicting-objectives/evidence/*
uv run --project submissions/reward-free-alignment-for-conflicting-objectives python -m reward_free_alignment.generate_evidence
sha256sum submissions/reward-free-alignment-for-conflicting-objectives/evidence/*
uv run --with pytest --project submissions/reward-free-alignment-for-conflicting-objectives pytest -q submissions/reward-free-alignment-for-conflicting-objectives/tests
```

Expected: both hash lists are identical and all tests pass.

- [ ] **Step 5: Commit**

```bash
git add submissions/reward-free-alignment-for-conflicting-objectives
git commit -m "feat(raco): publish deterministic evidence bundle"
```
