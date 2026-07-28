# Success Conditioning Formal-Evidence Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic CPU reproduction that independently checks all four admitted Success Conditioning claims with universal symbolic certificates, exact finite MDPs, and analytic Beta-bandit instances, while preserving source-locator discrepancies.

**Architecture:** A paper-scoped Python project pins and verifies arXiv v2, verifies a universal symbolic proof certificate for claims 1–3, computes finite-MDP truth with exact rational arithmetic, cross-checks success conditioning through an independent constrained optimizer and exhaustive controls, then emits schema-validated evidence and generated root pages. The symbolic certificate is independent of all bounded enumeration and SLSQP results.

**Tech Stack:** Python 3.11+, `fractions.Fraction`, SymPy, NumPy, SciPy, JSON Schema, pytest, Gradio, and uv.

## Global Constraints

- Attempt ID is `b2b5899c-43a1-4c91-8b1f-9122d746f4c6`; paper ID is `FEmXFeqYNZ`; assessed snapshot is `09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199`.
- The project path is exactly `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success`.
- Preserve the four ordered target claims and hashes from the design verbatim.
- Do not repair the fourth claim string: it says Section 5, while pinned v2
  places return thresholding in Section 7, Proposition 7.1, and Section 7.3.
  Record both locators, expose the mismatch on every root page, and combine it
  with the derived substantive status: `supported + mismatch -> mixed`,
  `not_supported + mismatch -> not_supported`, and
  `inconclusive + mismatch -> inconclusive`.
- Pin `arxiv:2601.18175v2` PDF bytes: byte count `652921`, SHA-256 `f9194e48cadf5c13307eb6a523ed20e4fb787856b1fc8b19e4f628a8ac3ad672`, CC BY 4.0.
- Never use paper prose, printed values, Figure 2 pixels, or another contributor's outputs as reproduced measurements.
- Use exact `Fraction` truth decisions. Floating point is restricted to the independent optimizer, Beta special functions, tolerances, and rendering.
- CPU only; paid API cost USD `0.00`; no model training; evidence recomputation has no network access.
- Write and run a failing focused test before each production change, and append the red result to `evidence/tdd-log.jsonl`.
- Do not edit state, skills, controller documents, another submission, or Hub resources. Only the controller performs lifecycle actions.
- Root `README.md`, `report.md`, `poster.html`, `poster_embed.html`, and `app.py` must be generated from validated evidence or contain only static explanatory structure.

---

Set these variables for every command:

```bash
SLUG=success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success
PROJECT="submissions/$SLUG"
PINNED_INPUT_DIR=/tmp/success-conditioning-inputs
```

### Task 1: Project, immutable claims, provenance, and transcriptions

**Files:**
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/pyproject.toml`
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/uv.lock`
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/LICENSE`
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/LICENSES/CC-BY-4.0.txt`
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/NOTICE.md`
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/paper_transcriptions/manifest.json`
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/paper_transcriptions/excerpts/*.txt`
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/src/success_conditioning_repro/{__init__,claims,provenance}.py`
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/evidence/tdd-log.jsonl`
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/tests/conftest.py`
- Test: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/tests/test_provenance.py`

**Interfaces:**
- Produces: `PAPER`, `TARGET_CLAIMS`, `CLAIM_HASHES`, `verify_pdf(path: Path) -> None`, and `load_transcriptions(root: Path) -> tuple[dict[str, object], ...]`.
- Consumes: no production interfaces.

- [ ] **Step 1: Write the failing provenance and exact-claim tests**

```python
def test_pinned_pdf_rejects_wrong_bytes(tmp_path):
    bad = tmp_path / "paper.pdf"
    bad.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="pinned PDF byte count"):
        verify_pdf(bad)

def test_claim_order_and_hashes_are_immutable():
    assert tuple(hashlib.sha256(c.encode()).hexdigest() for c in TARGET_CLAIMS) == CLAIM_HASHES
    assert CLAIM_HASHES == (
        "aec6dd70e68aa5af00d320cdfb51bc69d5490b4f0de2e79d2796366dc71d0b5a",
        "7b9f91421c4d8d5bb2d744a1346fd80e3d1820ab7edac2b0a40b261b4316594b",
        "44992bcb0dd536f837f284ff9ed37465729f87e27efe1f4679b390ee957b90e3",
        "2d4441bf79a66be5f94ce57efae3bcafc796367a3caed5496a72588773e7243a",
    )
```

- [ ] **Step 2: Run the red test and record it**

Run:

```bash
uv run --project "$PROJECT" python -m pytest tests/test_provenance.py -q
```

Expected: collection fails because `success_conditioning_repro.provenance`
does not exist. Record the command, UTC timestamp, node IDs, exit code, and
`expected_missing_behavior: immutable provenance module absent`.

- [ ] **Step 3: Add minimal project metadata and provenance**

Use:

```toml
[project]
name = "success-conditioning-formal-evidence"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["gradio>=5.0", "jsonschema>=4.25", "numpy>=2.0", "scipy>=1.14", "sympy>=1.13"]

[project.scripts]
success-conditioning-repro = "success_conditioning_repro.cli:main"

[dependency-groups]
dev = ["pytest>=8.4", "pre-commit>=4.2"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Implement `verify_pdf` to check byte count before digest and raise distinct
`ValueError` messages. Implement transcription validation with exact record
keys:

```python
TRANSCRIPTION_KEYS = {
    "record_id", "section", "pdf_page", "statement",
    "source_excerpt_path", "source_excerpt_byte_count",
    "source_excerpt_sha256", "reviewed_by",
}
```

Required records cover Definition 3.1, Proposition 4.1, Definition 4.2,
Propositions 4.3 and 4.4, Corollary 4.5, Proposition 7.1, and the Section 7.3
Beta construction. The manifest also contains a closed locator-reconciliation
record with the immutable challenge locator `Section 5`, pinned-v2 content
locator `Section 7`, identity locator `Proposition 7.1`, example locator
`Section 7.3`, and `locator_status: mismatch`.

Define shared file-backed fixtures in `tests/conftest.py`:

```python
@pytest.fixture
def project_root() -> Path:
    return Path(__file__).parents[1]

@pytest.fixture
def pinned_pdf() -> Path:
    path = Path(os.environ["PINNED_PDF"])
    verify_pdf(path)
    return path

@pytest.fixture
def schema_path(project_root: Path) -> Path:
    return project_root / "evidence/schema.json"
```

- [ ] **Step 4: Lock dependencies and run focused tests**

Run:

```bash
uv lock --project "$PROJECT"
uv run --project "$PROJECT" python -m pytest tests/test_provenance.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the independently reviewable provenance lane**

```bash
git add "$PROJECT"
git commit -m "feat(success-conditioning): pin claims and paper provenance"
```

### Task 2: Exact finite-MDP semantics and independent conditioning

**Files:**
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/src/success_conditioning_repro/{types,linear,mdp,fixtures}.py`
- Test: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/tests/{test_linear,test_mdp,test_fixtures}.py`

**Interfaces:**
- Produces: `FiniteMDP`, `Evaluation`, `validate_mdp`, `evaluate_policy`, `condition_on_success`, `enumerate_conditioned_policy`, `success_conditioned_occupancy`, `layered_fixture`, and `transient_fixture`.
- Consumes: immutable claim/provenance constants from Task 1.

- [ ] **Step 1: Write failing exact-evaluation tests**

```python
def test_bayes_conditioning_matches_independent_trajectory_enumeration():
    mdp, pi0 = layered_fixture(seed=7, states=5, actions=3)
    ev = evaluate_policy(mdp, pi0)
    assert condition_on_success(mdp, pi0, ev) == enumerate_conditioned_policy(mdp, pi0)

def test_transient_policy_evaluation_satisfies_bellman_equations():
    mdp, pi0 = transient_fixture(seed=11, states=4, actions=2)
    ev = evaluate_policy(mdp, pi0)
    for state in mdp.nonterminal_states:
        assert ev.value[state] == sum(
            pi0[state][a] * ev.q_value[state, a] for a in mdp.actions[state]
        )
```

- [ ] **Step 2: Run the red tests and record both missing behaviors**

Run:

```bash
uv run --project "$PROJECT" python -m pytest tests/test_linear.py tests/test_mdp.py tests/test_fixtures.py -q
```

Expected: import failure for `success_conditioning_repro.types`.

- [ ] **Step 3: Implement frozen types and exact Gaussian elimination**

Implement the design signatures. `solve_fraction_system` must pivot
deterministically, reject inconsistent dimensions and singular systems, and
return `tuple[Fraction, ...]`. Probability rows must sum exactly to one.

```python
def solve_fraction_system(
    matrix: tuple[tuple[Fraction, ...], ...],
    rhs: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    n = len(rhs)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("square matrix and RHS dimensions must match")
    augmented = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = next(
            (row for row in range(column, n) if augmented[row][column] != 0),
            None,
        )
        if pivot is None:
            raise ValueError("singular linear system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return tuple(augmented[row][-1] for row in range(n))
```

- [ ] **Step 4: Implement evaluation and the two conditioning routes**

`evaluate_policy` solves transient Bellman and occupancy systems independently.
`condition_on_success` uses evaluated `Q/V`; `enumerate_conditioned_policy`
walks complete acyclic trajectories and accumulates successful action mass.
Neither function calls the other. Reject states with `V0 == 0`.

- [ ] **Step 5: Add deterministic full-support fixture generation**

Generate integer weights from `random.Random(seed)`, normalize them to
`Fraction`, serialize the seed, and validate absorption. Include repeated
states, rare actions, deterministic transitions, and stochastic transitions.

- [ ] **Step 6: Run focused tests and exact mutation controls**

Run:

```bash
uv run --project "$PROJECT" python -m pytest tests/test_linear.py tests/test_mdp.py tests/test_fixtures.py -q
```

Expected: PASS, including a test that changing one trajectory's terminal label
causes the independent conditioning comparison to fail.

- [ ] **Step 7: Commit the MDP truth engine**

```bash
git add "$PROJECT"
git commit -m "feat(success-conditioning): add exact finite MDP semantics"
```

### Task 3: Universal symbolic certificate and independent trust-region optimization

**Files:**
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/src/success_conditioning_repro/proofs.py`
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/src/success_conditioning_repro/trust_region.py`
- Test: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/tests/test_proofs.py`
- Test: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/tests/test_trust_region.py`

**Interfaces:**
- Produces: `UniversalCertificate`, `build_universal_certificate`,
  `verify_universal_certificate`, `TrustRegionProblem`, `SolverResult`,
  `build_trust_region_problem`, `solve_trust_region`,
  `relative_objective_gap`, `kkt_residuals`, and
  `exhaustive_bandit_bound`.
- Consumes: `FiniteMDP`, `Evaluation`, `Policy`, and exact occupancy functions from Task 2.

- [ ] **Step 1: Write failing universal-certificate tests**

```python
def test_claims_one_through_three_have_universal_certificates():
    certificate = build_universal_certificate()
    verify_universal_certificate(certificate)
    assert certificate.claim_ids == (1, 2, 3)
    assert certificate.fixture_values == ()
    assert certificate.numerical_tolerances == ()
    assert certificate.rules_used >= {
        "weighted_cauchy_schwarz",
        "kkt",
        "sum_of_squares",
        "policy_improvement",
    }

def test_symbolic_certificate_rejects_changed_radius():
    certificate = dataclasses.replace(
        build_universal_certificate(), trust_region_radius="wrong_radius"
    )
    with pytest.raises(ValueError, match="radius"):
        verify_universal_certificate(certificate)
```

Run:

```bash
uv run --project "$PROJECT" python -m pytest tests/test_proofs.py -q
```

Expected: import failure for `success_conditioning_repro.proofs`.

- [ ] **Step 2: Implement and verify the universal proof certificate**

Use formal indexed finite sums over arbitrary finite states and each state's
active behavior support, with premises `d_s >= 0`, `p_{s,i} > 0`,
`sum_i p_{s,i} = 1`, `0 <= q_{s,i} <= 1`, and
`v_s = sum_i p_{s,i} q_{s,i} > 0`. The certificate verifier must independently
reduce and check:

1. `pi_plus_{s,i} = p_{s,i} q_{s,i} / v_s`, normalization, and support
   inclusion.
2. The aggregate state/action weighted Cauchy–Schwarz trust-region upper bound
   and equality case, plus generic-coordinate KKT primal feasibility, dual
   feasibility, stationarity, and complementary slackness at `pi_plus`.
3. Independent normal forms for relative improvement,
   `chi2(pi_plus || p)`, and action influence, all equal to
   `sum_i p_{s,i} (q_{s,i}-v_s)^2 / v_s^2` at each state.
4. Nonnegative advantage as an explicit weighted sum of squares and the
   finite-MDP policy-improvement implication used by claim 3.

SymPy may canonicalize exact symbolic expressions, but the project-owned
certificate verifier must check the premises and every inference rule. It must
reject missing premises, reversed divergence arguments, changed exponents,
changed radius, or an unsigned inequality. No fixture, seed, SLSQP result,
paper conclusion, or tolerance may enter this certificate.

- [ ] **Step 3: Write the failing blind-optimizer test**

```python
def test_blind_solver_recovers_success_conditioned_optimum():
    mdp, pi0 = layered_fixture(seed=19, states=6, actions=3)
    ev0 = evaluate_policy(mdp, pi0)
    expected = condition_on_success(mdp, pi0, ev0)
    problem = build_trust_region_problem(mdp, pi0, ev0)
    result = solve_trust_region(problem, starts=8)
    assert result.success
    assert result.constraint_violation <= 1e-9
    assert relative_objective_gap(problem, result.policy, expected) <= 1e-8
```

- [ ] **Step 4: Run the red optimizer test and record it**

Run:

```bash
uv run --project "$PROJECT" python -m pytest tests/test_trust_region.py -q
```

Expected: import failure for `trust_region`.

- [ ] **Step 5: Implement problem construction without importing conditioning or proofs**

`trust_region.py` may import evaluation and occupancy interfaces, but must not
import or call `condition_on_success`, `proofs`, or the symbolic certificate.
Flatten state-action variables in
canonical order, add one simplex equality per state, nonnegative bounds, and
the aggregate chi-squared inequality.

- [ ] **Step 6: Implement eight-start SLSQP and KKT diagnostics**

Use deterministic starts: behavior policy, uniform-on-support, and six seeded
Dirichlet vectors. Record raw status, iterations, objective, constraint value,
simplex residual, projected stationarity residual, and policy. Any failed
start remains visible. At least one feasible converged solution within the
expected optimum tolerance supplies positive numerical corroboration; complete
solver failure yields `inconclusive`, while a feasible converged solution that
exceeds the expected optimum supplies a potential `not_supported`
counterexample. Neither scientific outcome aborts serialization.

- [ ] **Step 7: Add exhaustive rational-grid bandit controls**

For two- and three-arm bandits, enumerate probability vectors on denominator
`200`, retain feasible points, and assert none beats the analytical objective
beyond the grid upper-bound allowance. Add mutations for half radius, wrong
occupancy, and reverse chi-squared arguments; each must change or invalidate
the claimed optimizer.

- [ ] **Step 8: Run the independent proof and numerical suites**

```bash
uv run --project "$PROJECT" python -m pytest tests/test_proofs.py tests/test_trust_region.py -q
```

Expected: the universal certificate passes without fixture inputs or
tolerances, independently of 64 SLSQP fixtures and 128 exhaustive bandits.

- [ ] **Step 9: Commit the symbolic and numerical trust-region lanes**

```bash
git add "$PROJECT"
git commit -m "feat(success-conditioning): certify trust-region optimum"
```

### Task 4: Triple identity and conservative-improvement certificates

**Files:**
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/src/success_conditioning_repro/{identities,conservative}.py`
- Test: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/tests/{test_identities,test_conservative}.py`

**Interfaces:**
- Produces: `IdentityRow`, `statewise_identity`, `aggregate_identity`,
  `ConservativeResult`, and `conservative_checks`.
- Consumes: Task 2 exact MDP interfaces only; it does not consume Task 3 solver outputs.

- [ ] **Step 1: Write failing exact-identity and monotonicity tests**

```python
def test_three_quantities_are_exactly_equal_statewise():
    mdp, pi0 = transient_fixture(seed=23, states=5, actions=3)
    ev0 = evaluate_policy(mdp, pi0)
    pi_plus = condition_on_success(mdp, pi0, ev0)
    for row in statewise_identity(mdp, pi0, pi_plus):
        assert row.relative_advantage == row.chi_squared == row.action_influence

def test_success_conditioning_is_conservative():
    mdp, pi0 = transient_fixture(seed=29, states=5, actions=3)
    ev0 = evaluate_policy(mdp, pi0)
    result = conservative_checks(mdp, pi0, condition_on_success(mdp, pi0, ev0))
    assert result.rho_gain >= 0
    assert result.minimum_value_gain >= 0
    assert result.minimum_lower_bound_slack >= 0
    assert result.new_support_mass == 0
```

- [ ] **Step 2: Run red tests and record missing interfaces**

```bash
uv run --project "$PROJECT" python -m pytest tests/test_identities.py tests/test_conservative.py -q
```

Expected: import failures for `identities` and `conservative`.

- [ ] **Step 3: Implement three independent statewise calculations**

Compute relative advantage from fresh Q and policy weighting, chi-squared from
policy ratios, and influence from Q variance. Do not derive any one field by
copying another. Aggregate with success-conditioned occupancy and compare to
`L(pi_plus)/rho(pi0)` exactly.

- [ ] **Step 4: Implement fresh-policy conservative checks**

Re-evaluate `pi_plus` independently, compare every positive-value state,
verify the Corollary 4.5 lower bound and initial success-rate monotonicity,
check aggregate movement equals the data radius, and check support inclusion.
The output field is `bounded_chi_squared_shift`, not `safe`.

- [ ] **Step 5: Add adversarial negative controls**

Tests must prove that reversed Q weighting can degrade value and that injected
out-of-support mass is rejected. A near-zero-influence case must remain close
to `pi0` without division instability.

- [ ] **Step 6: Run focused tests**

```bash
uv run --project "$PROJECT" python -m pytest tests/test_identities.py tests/test_conservative.py -q
```

Expected: PASS across every reachable state in 256 layered and 64 transient
fixtures, with negative controls detected.

- [ ] **Step 7: Commit identity and conservative certificates**

```bash
git add "$PROJECT"
git commit -m "feat(success-conditioning): certify identities and improvement"
```

### Task 5: Return-threshold amplification and misalignment

**Files:**
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/src/success_conditioning_repro/thresholding.py`
- Test: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/tests/test_thresholding.py`

**Interfaces:**
- Produces: `ThresholdRow`, `ThresholdResult`,
  `beta_threshold_sweep(seed: int, thresholds: Sequence[float])`, and
  `proxy_identity_residual`.
- Consumes: NumPy RNG only for fixed parameter fixtures and
  `scipy.special.betainc` for analytic Beta survival probabilities.

- [ ] **Step 1: Write the failing analytic-threshold test**

```python
def test_threshold_sweep_contains_amplification_and_harm():
    result = beta_threshold_sweep(31, tuple(i / 1000 for i in range(1, 1000)))
    assert result.maximum_proxy_gain > result.faithful_gain
    assert result.minimum_proxy_gain < 0
    assert result.maximum_proxy_identity_residual <= 1e-10

def test_pinned_v2_locator_mismatch_is_preserved():
    result = beta_threshold_sweep(31, tuple(i / 1000 for i in range(1, 1000)))
    assert result.challenge_locator == "Section 5"
    assert result.source_locator == "Section 7"
    assert result.identity_locator == "Proposition 7.1"
    assert result.example_locator == "Section 7.3"
    assert result.locator_status == "mismatch"
```

- [ ] **Step 2: Run the red test and record it**

```bash
uv run --project "$PROJECT" python -m pytest tests/test_thresholding.py -q
```

Expected: import failure for `thresholding`.

- [ ] **Step 3: Implement faithful and proxy conditioning analytically**

For 99 arms draw and persist `a_i=b_i` from `Uniform(0.3, 0.7)` using
`numpy.random.default_rng(seed)`; arm 100 is `Beta(18, 2)`. The behavior
policy is uniform. Faithful weights use exact arm means. Proxy weights use
`1 - betainc(a_i, b_i, theta)`. Compute true return under both conditioned
policies and the action-influence/alignment terms independently.

- [ ] **Step 4: Verify Proposition 7.1 and population robustness**

Run seeds `0..63` and thresholds `0.001..0.999`. Require finite values for a
well-formed record and persist every arm parameter and threshold row. Derive
substantive `supported` only when the identity residual is at most `1e-10` and
both regimes occur in at least 60 seeds; a complete null or counterexample is
`not_supported`, while an incomplete valid domain is `inconclusive`. Do not use
Figure 2 coordinates.

- [ ] **Step 5: Add negative controls and run focused tests**

Swap the Beta tail for its CDF and assert that the resulting valid negative
control is serialized with a derived non-supporting status rather than
discarded or treated as an integrity failure.
Run:

```bash
uv run --project "$PROJECT" python -m pytest tests/test_thresholding.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the thresholding lane**

```bash
git add "$PROJECT"
git commit -m "feat(success-conditioning): reproduce threshold tradeoff"
```

### Task 6: Canonical evidence, schema, and deterministic bundle

**Files:**
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/src/success_conditioning_repro/{evidence,cli}.py`
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/evidence/schema.json`
- Generate: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/evidence/{evidence.json,cases.jsonl,provenance.json,manifest.json,repro-bundle.tar.gz}`
- Test: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/tests/test_evidence.py`

**Interfaces:**
- Produces: `derive_claim_status`, `combine_threshold_status`,
  `assemble_evidence`, `build_evidence`, `validate_evidence`, and CLI commands
  `recompute`, `validate`, and `render`.
- Consumes: all six lanes and file-backed provenance.

- [ ] **Step 1: Write failing schema and tamper tests**

```python
def test_complete_evidence_is_file_backed_and_semantically_valid(
    tmp_path, pinned_pdf, schema_path
):
    evidence = build_evidence(tmp_path, pinned_pdf)
    validate_evidence(tmp_path / "evidence.json", schema_path, tmp_path)
    assert all(
        claim["local_status"]
        in {"supported", "mixed", "not_supported", "inconclusive"}
        for claim in evidence["claims"]
    )
    threshold_claim = evidence["claims"][3]
    assert threshold_claim["locator_status"] == "mismatch"
    assert threshold_claim["local_status"] == combine_threshold_status(
        threshold_claim["substantive_status"],
        threshold_claim["locator_status"],
    )
    assert "Section 5" in threshold_claim["limitations"][0]
    assert "Section 7" in threshold_claim["limitations"][0]

def test_exact_counterexample_is_serialized_not_supported(
    tmp_path, schema_path, valid_provenance
):
    lanes = complete_lane_fixture()
    lanes["claim_2"]["exact_violations"] = [statewise_counterexample()]
    evidence = assemble_evidence(lanes, valid_provenance, tmp_path)
    validate_evidence(tmp_path / "evidence.json", schema_path, tmp_path)
    assert evidence["claims"][1]["local_status"] == "not_supported"
    assert evidence["claims"][1]["counterexamples"]

def test_solver_failure_is_serialized_inconclusive(
    tmp_path, schema_path, valid_provenance
):
    lanes = complete_lane_fixture()
    lanes["claim_1"]["solver_starts"] = all_failed_solver_starts()
    lanes["claim_1"]["exact_counterexamples"] = []
    evidence = assemble_evidence(lanes, valid_provenance, tmp_path)
    validate_evidence(tmp_path / "evidence.json", schema_path, tmp_path)
    assert evidence["claims"][0]["local_status"] == "inconclusive"
    assert evidence["claims"][0]["solver_starts"]

def test_complete_threshold_null_is_not_supported_not_forced_mixed(
    tmp_path, schema_path, valid_provenance
):
    lanes = complete_lane_fixture()
    lanes["claim_4"].update(
        complete=True,
        amplification_seed_count=0,
        harmful_seed_count=0,
        substantive_status="not_supported",
    )
    evidence = assemble_evidence(lanes, valid_provenance, tmp_path)
    validate_evidence(tmp_path / "evidence.json", schema_path, tmp_path)
    claim = evidence["claims"][3]
    assert claim["locator_status"] == "mismatch"
    assert claim["local_status"] == "not_supported"

def test_case_tampering_is_rejected(tmp_path, pinned_pdf, schema_path):
    build_evidence(tmp_path, pinned_pdf)
    (tmp_path / "cases.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="cases SHA-256"):
        validate_evidence(tmp_path / "evidence.json", schema_path, tmp_path)
```

- [ ] **Step 2: Run the red test and record it**

```bash
PINNED_PDF="$PINNED_INPUT_DIR/2601.18175v2.pdf" \
  uv run --project "$PROJECT" python -m pytest tests/test_evidence.py -q
```

Expected: import failure for `evidence`.

- [ ] **Step 3: Implement canonical schema and orchestration**

Top-level evidence keys are exactly:

```python
{
    "schema_version", "paper", "attempt_id", "claims", "domains",
    "acceptance", "artifacts", "limitations", "actual_api_cost_usd",
}
```

Each claim binds exact text and hash, names its independent computation,
records `local_status`, expected observation, actual summary, case pointers,
and limitations. Claims 1–3 additionally point to the validated universal
certificate and its inference records. Claim 4 records separate substantive
and locator statuses and combines them with the fixed status table; the
renderer never chooses a status. Accept and serialize complete scientific
counterexamples, null effects, and solver failures as `not_supported` or
`inconclusive`. Reject only malformed/integrity-invalid evidence: reduced
counts, reordered claims, missing required records or negative controls,
nonfinite values, invalid schemas, bad hashes, or malformed/tampered proof
certificates.

- [ ] **Step 4: Implement stable serialization and deterministic archive**

Sort JSON keys, use UTF-8 plus one terminal newline, order JSONL by canonical
case ID, and create gzip/tar entries with fixed mode, uid/gid, names, and
mtime. Manifest hashes every evidence member but not itself.

- [ ] **Step 5: Run two clean recomputations and compare hashes**

```bash
OUT1=$(mktemp -d)
OUT2=$(mktemp -d)
uv run --project "$PROJECT" success-conditioning-repro recompute --pinned-pdf "$PINNED_INPUT_DIR/2601.18175v2.pdf" --output-dir "$OUT1"
uv run --project "$PROJECT" success-conditioning-repro recompute --pinned-pdf "$PINNED_INPUT_DIR/2601.18175v2.pdf" --output-dir "$OUT2"
diff -ru "$OUT1" "$OUT2"
```

Expected: no diff.

- [ ] **Step 6: Run evidence tests and commit**

```bash
PINNED_PDF="$PINNED_INPUT_DIR/2601.18175v2.pdf" \
  uv run --project "$PROJECT" python -m pytest tests/test_evidence.py -q
git add "$PROJECT"
git commit -m "feat(success-conditioning): build canonical evidence"
```

### Task 7: Generated root pages, CPU Space proposal, and full validation

**Files:**
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/src/success_conditioning_repro/render.py`
- Create: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/{README.md,report.md,poster.html,poster_embed.html,app.py}`
- Test: `submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/tests/{test_render,test_app,test_acceptance}.py`

**Interfaces:**
- Produces: `render_all(evidence_path: Path, project_root: Path) -> None`,
  `build_app()`, and a controller-reviewable CPU Space source tree.
- Consumes: validated canonical evidence only.

- [ ] **Step 1: Write failing provenance-pointer and app tests**

```python
def test_every_rendered_measurement_has_evidence_pointer(project_root):
    render_all(project_root / "evidence/evidence.json", project_root)
    for path in ("report.md", "poster.html", "poster_embed.html"):
        text = (project_root / path).read_text()
        assert "data-evidence-pointer=" in text or "Evidence pointer:" in text

def test_space_builds_without_network(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert build_app() is not None
```

- [ ] **Step 2: Run red tests and record missing rendering**

```bash
uv run --project "$PROJECT" python -m pytest tests/test_render.py tests/test_app.py tests/test_acceptance.py -q
```

Expected: import failure for `render`.

- [ ] **Step 3: Implement evidence-only rendering**

Validate evidence before rendering. Root pages show source pin, exact commands,
four claim cards, domain counts, tolerances, negative controls, limitations,
and per-value JSON pointers. The safety wording must say the computation
checks the finite-MDP chi-squared movement bound and does not establish general
deployment safety. The fourth card must show the immutable challenge wording
unchanged, render the serialized substantive, locator, and aggregate statuses
without overriding them, and state prominently that “Section 5” is the
challenge locator while pinned v2 places the content in Section 7,
Proposition 7.1, and Section 7.3. Rendering tests cover `mixed`,
`not_supported`, and `inconclusive` claim-4 fixtures.

- [ ] **Step 4: Implement the bounded CPU app**

`app.py` loads committed evidence and exposes summary, case lookup, artifact
downloads, and a bounded recomputation action. It performs no network call,
accepts no arbitrary path, and writes recomputation only to a temporary
directory. Add Space frontmatter with `paper-FEmXFeqYNZ`,
`icml2026-repro`, CPU SDK, and exact Python/runtime metadata.

- [ ] **Step 5: Run project and root validation**

```bash
uv sync --project "$PROJECT" --frozen
PINNED_PDF="$PINNED_INPUT_DIR/2601.18175v2.pdf" \
  uv run --project "$PROJECT" python -m pytest -q
uv run --project "$PROJECT" success-conditioning-repro validate --evidence "$PROJECT/evidence/evidence.json" --schema "$PROJECT/evidence/schema.json"
uv run --project "$PROJECT" success-conditioning-repro render --evidence "$PROJECT/evidence/evidence.json" --project-root "$PROJECT"
git diff --check
uv run pre-commit run -a
```

Expected: every command succeeds; rendering leaves no diff after regeneration.

- [ ] **Step 6: Commit the complete worker proposal**

```bash
git add "$PROJECT"
git commit -m "feat(success-conditioning): add evidence-backed CPU Space"
```

- [ ] **Step 7: Return the proposal for controller review**

Report attempt ID, assigned worktree, final commit, all commands, evidence
paths and hashes, limitations, and concerns. Do not deploy, submit, mutate
state, or claim an official verdict. The controller must independently review,
run the authoritative validation manifest, and create any lifecycle
attestations.
