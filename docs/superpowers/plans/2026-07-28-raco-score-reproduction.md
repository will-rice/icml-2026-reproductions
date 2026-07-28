# RACO Score Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the toy RACO implementation with deterministic CPU evidence
for objective-specific preference losses, weighted CAGrad-Clip, Theorem 3.1,
and Theorem 3.2's exact per-step descent certificate.

**Architecture:** A fail-closed provenance layer binds the admitted snapshot,
all ten live claims, arXiv v3, and the pinned repository. Focused modules
implement stable pairwise losses, the exact weighted two-objective simplex
solver, coordinate-wise clipping, and analytic theorem audits. A canonical
evidence builder feeds committed root `pages/*.md` and a read-only Space.

**Tech Stack:** Python 3.12, PyTorch CPU, Python standard library
(`dataclasses`, `hashlib`, `json`, `math`, `pathlib`, `tempfile`), pytest,
canonical JSON, Markdown, Gradio.

## Controller Correction Gate — 2026-07-28, round 7

Use `icml-repro-loop`, `superpowers:test-driven-development`,
`superpowers:systematic-debugging`, and
`superpowers:verification-before-completion`. Guarded round 6 exited at the
unchanged base `0dc6d34a787283c65fd119f6872eb1d70c8be906` and left only two
uncommitted RED test files; the controller preserved that test-only proposal
as `40cfc73`. No production correction, regenerated evidence, or worker commit
was produced. Start by running the preserved RED tests and then implement all
items below.

1. Implement the round-6 `steps` evidence, not merely its tests. Emit exactly
   ten closed-schema records for `t=0,...,9`, each containing `step_index`,
   `current_iterate`, `weighted_anchor`, `cagrad_direction`, `next_iterate`,
   `loss_before`, `loss_after`, `m_value`, `grad_norm`, `descent_holds`, and
   `m_bound_holds`. Strengthen the behavioral test to recompute every numeric
   field independently from the two quadratic objectives and a fresh
   `cagrad_clip` call, not just the iterate-update identity. Exclude terminal
   `t=T` diagnostics from the per-update array.
2. Expose the two finite-horizon squared checks separately as
   `grad_finite_horizon_bound_holds` and
   `m_finite_horizon_bound_holds`, while retaining their conjunction if useful.
   Derive Claim 8 support only when every one of the ten step descent/M
   booleans and both separately persisted finite-horizon booleans are true.
   Mutating any one of those four conditions to false must make Claim 8
   `not-supported`.
3. Bind every artifact ID to its exact raw URL rather than accepting any URL
   on the raw host:
   `LICENSE`, `README.md`, `m=3-RACO-CAGrad-Algo.md`, and `train_raco.py` must
   map to the corresponding
   `https://raw.githubusercontent.com/PeterLauLukChen/RACO/84a943c34f38520c7e0c9dd3066517c111b3c8fa/<path>`
   URL. Reject swapped paths, query/fragment suffixes, GitHub `/blob/` HTML
   pages, and other raw-host URLs. Regenerate canonical evidence and the
   provenance page from these exact identities.
4. Update the closed evidence schema for every new field. Generate twice and
   compare bytes; run the full submission suite, frozen-lock verification,
   root pytest, skill validation, pre-commit, and `git diff --check`. Commit
   the production correction and all generated outputs before returning.

## Controller Correction Gate — 2026-07-28, round 6

Use `icml-repro-loop`, `superpowers:test-driven-development`,
`superpowers:systematic-debugging`, and
`superpowers:verification-before-completion`. Controller review rejected
proposal `618f724e8aff048dce03239aea3c56ef35736556`. Preserve that commit as
the rejected proposal. For every item below, first add a behavioral regression
and record its expected RED failure against `618f724e`.

1. `execute_raco_trajectory` now computes the requested CAGrad-Clip quantities,
   but `_run_theorem_31_audit` drops the weighted anchors, CAGrad directions,
   next iterates, loss-before/loss-after pairs, and per-step descent booleans.
   Consequently canonical `evidence/results.json` cannot substantiate that its
   advertised trajectory used RACO rather than the old pure-`g0` update.
   Persist a closed-schema `steps` array of exactly ten records for
   `t=0,...,9`. Each record must contain the step index, current iterate,
   weighted anchor, actual CAGrad-Clip direction, next iterate, loss before,
   loss after, `M(theta_t)`, weighted-gradient norm, descent-bound boolean, and
   `M(theta_t) <= ||grad L_w(theta_t)||` boolean. Assert each numeric identity
   independently in the evidence test, including
   `theta_next = theta - eta * cagrad_direction`. Do not mix the terminal
   `t=T` diagnostics into these per-update records. Claim 8 may be `supported`
   only if all ten step records pass both booleans and both finite-horizon
   squared bounds pass.
2. The artifact manifest uses GitHub HTML `/blob/<commit>/...` pages, not the
   immutable raw artifact URLs required by the round-5 gate. Bind each
   artifact ID to its exact
   `https://raw.githubusercontent.com/PeterLauLukChen/RACO/84a943c34f38520c7e0c9dd3066517c111b3c8fa/<path>`
   URL (with correct path encoding) and reject a `/blob/` URL even when it
   contains the pinned commit. Regenerate canonical evidence and the
   provenance page so they expose those exact raw URLs.
3. Regenerate canonical evidence and every affected page. Compare two
   independent evidence generations byte-for-byte; run the full submission
   suite, frozen-lock verification, root pytest, skill validation, pre-commit,
   and `git diff --check`. Commit only the assigned submission after this
   durable controller-plan commit.

## Controller Correction Gate — 2026-07-28, round 5

Use `icml-repro-loop`, `superpowers:test-driven-development`,
`superpowers:systematic-debugging`, and
`superpowers:verification-before-completion`. Controller review rejected
proposal `fe6b6b63c80aca9ac802624bdce6c165ca16570a`. Preserve that commit as
the rejected proposal. For every item below, add a behavioral regression,
run it against `fe6b6b6`, and record the expected RED failure before changing
production code.

1. `load_live_claims` still accepts a caller-edited claim when its SHA is
   recomputed to match the edited text. Bind the loader itself to the exact ten
   admitted ordinals, texts, hashes, and target flags. A temporary copy with
   one changed text and matching new SHA must raise `IntegrityError`.
2. The four packaged upstream files are verified but their lineage disappears
   from `evidence/results.json`: it has no artifacts/provenance field.
   Extend `VerifiedArtifact` and the closed evidence schema so canonical
   evidence and the provenance page expose every artifact's ID, path, SHA-256,
   Git blob, byte size, immutable commit-qualified source URL, reproducible
   acquisition command including checkout of
   `84a943c34f38520c7e0c9dd3066517c111b3c8fa`, and Apache-2.0 license.
   Reject generic repository URLs and clone-only acquisition commands.
3. `execute_raco_trajectory` claims to execute RACO but advances with
   `x = x - eta * g0`, never calling CAGrad-Clip and never using `c`.
   Execute the actual audited CAGrad-Clip update at every step. Persist each
   weighted anchor, CAGrad direction, next iterate, loss before/after,
   `M(theta_t)`, weighted-gradient norm, and separate per-step descent and
   `M <= ||grad L_w||` booleans for exactly `t=0,...,T-1`. Claim 8 may be
   `supported` only when every step and both finite-horizon squared bounds
   pass. A mutation replacing the CAGrad direction with `g0` must fail a
   literal, hand-derived test.
4. Make `compute_m_simplex` scale-aware: the opposing
   `g1=[1e-8], g2=[-1e-8]` fixture must return zero, not treat the gradients as
   identical because of an absolute squared-norm threshold.
5. For a zero weighted anchor, persist the effective correction radius as
   zero in `CAGradResult`; for identical gradients, report the exact original
   subproblem objective including the `c*||g0||*||g||` term. Add literal
   regressions for both diagnostics.
6. Regenerate canonical evidence and every page. Compare two independent
   evidence generations byte-for-byte; run the full submission suite,
   frozen-lock verification, root pytest, skill validation, pre-commit, and
   `git diff --check`. Commit only the assigned submission after this durable
   controller-plan commit.

## Global Constraints

- Work only in
  `submissions/reward-free-alignment-for-conflicting-objectives/`.
- Attempt ID is `97e213a5-7ca3-4a1b-a500-1ec52d94d87a`; paper ID is
  `vSzRJyg6k0`; admitted snapshot is
  `09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199`.
- Pin `arxiv:2602.02495v3` and repository commit
  `84a943c34f38520c7e0c9dd3066517c111b3c8fa`.
- Preserve all ten exact live claim texts, hashes, and order from the approved
  design. Claims 6–9 are the four selected targets.
- Use CPU only, paid API cost USD 0.00, deterministic algorithms, and no LLM
  training or inference.
- Never put paper-reported empirical values in reproduced measurement fields.
- Form `g0=w1*g1+w2*g2`; user weights are required solver inputs.
- Apply `p_tilde_i=min(p_i,w_i)` coordinate-wise and never renormalize after
  clipping.
- Audit Theorem 3.2 through
  `Gamma(rho_tilde)-Gamma(rho)`, not iteration counts.
- Local outcomes are only `supported`, `not-supported`, and `limited`.
  `verified`, `falsified`, and `toy` are reserved for official verdicts.
- Produce committed root `pages/*.md`; the app and tests must load them
  directly without network access.
- Use a failing test before each production behavior. The full evidence run
  must be deterministic, offline after acquisition, and finish within 30 CPU
  minutes.
- Do not mutate coordinator state, `docs/HANDOFF.md`, skill source, another
  submission, NAPE, Hub resources, submission records, verdicts, or controller
  attestations.

---

### Task 1: Immutable provenance and exact live-claim registry

**Files:**

- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/src/reward_free_alignment/provenance.py`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/evidence/inputs/upstream_manifest.json`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/evidence/inputs/live_claims.json`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/tests/test_provenance.py`
- Modify:
  `submissions/reward-free-alignment-for-conflicting-objectives/pyproject.toml`

**Interfaces:**

```python
class IntegrityError(ValueError):
    """Pinned identity, artifact, or live-claim bytes failed verification."""

@dataclass(frozen=True)
class LiveClaim:
    ordinal: int
    text: str
    sha256: str
    targeted: bool

@dataclass(frozen=True)
class VerifiedArtifact:
    artifact_id: str
    relative_path: str
    sha256: str
    git_blob: str | None
    size_bytes: int

load_live_claims(path: Path) -> tuple[LiveClaim, ...]
load_verified_artifacts(project_root: Path) -> tuple[VerifiedArtifact, ...]
```

`live_claims.json` must bind these exact texts to `EXPECTED_HASHES` below,
position by position:

1. `RACO is an offline, reward-free preference-alignment method that accepts user-specified objective weights and explicitly handles conflicting objectives (Table 1).`
2. `The method uses CAGrad-Clip to limit correction gradients so updates better respect preferred objective trade-offs (Figure 1, Algorithm 1).`
3. `On TL;DR summarization, RACO achieves better Pareto frontiers for conciseness-quality and faithfulness-quality trade-offs than AMoPO and weighted-loss DPO baselines (Figure 2, Figure 3).`
4. `On BeaverTails safety alignment, RACO improves harmlessness-helpfulness Pareto trade-offs across Qwen3 and Gemma3 setups (Figure 4).`
5. `Ablations show clipping and the correction-radius constant affect validation margins and Pareto frontiers (Figure 5, Figure 6).`
6. `RACO directly applies conflict-averse gradient descent to objective-specific pairwise preference losses instead of relying on explicit reward models (Section 3).`
7. `The clipped CAGrad update is introduced to stabilize multi-objective LLM alignment while respecting user-specified objective weights (Section 3.2).`
8. `The paper proves convergence of clipped CAGrad to Pareto-critical points that respect user-specified weights in nonconvex smooth settings (Theorem 3.1).`
9. `For two objectives, the analysis shows clipping can strictly improve the convergence rate (Theorem 3.2).`
10. `Experiments on multi-objective summarization and safety alignment across Qwen 3, Llama 3, and Gemma 3 report better Pareto trade-offs than reward-free baselines (Section 4).`

- [ ] **Step 1: Write the failing identity and exact-order tests**

```python
EXPECTED_HASHES = (
    "e9a35e34b57a7273bf84d3d5981ab19f8ff1088adef8363f4640dcf436183944",
    "7c0aa54e034d03f2d0905417a024af4db41338cd9a803a0b42e441945c307cf9",
    "85abbc8a21d5c4537409f6e9f2af6bffc7e4c15e2311dfa078bf816ea0cffc9e",
    "dac93f364ac0469302894920781b034bfcd205816fbe16537c2f8e7c10d8995d",
    "269d8a5053e224206036399bccb2435455565149086de6439a9046d89682772b",
    "0d457572ea8a502fa8a489fef3e15da21b13cc39dd3a3730843d1cbe833059b0",
    "50719d645042a500f9c4d53fbdfedf719ee56429ddb73a231912f1eaeadb1b31",
    "5ec835ce150ff60d1e2bbd4fbdf7d1ebacf91bb2b6b2d65f72c44c3b3ed65229",
    "b74a0ea75967144b210934fd40fd23449d3ef985df878d5a9e14c4b04025ba4b",
    "58b31f527bb5e1bccb05c0dab775a74c2f2bdcd8e92ef2c0dd578733b5fb058e",
)


def test_live_claims_match_admitted_snapshot(project_root):
    claims = load_live_claims(project_root / "evidence/inputs/live_claims.json")
    assert [claim.ordinal for claim in claims] == list(range(1, 11))
    assert tuple(claim.sha256 for claim in claims) == EXPECTED_HASHES
    assert [claim.targeted for claim in claims] == [
        False, False, False, False, False, True, True, True, True, False
    ]
    for claim in claims:
        assert hashlib.sha256(claim.text.encode("utf-8")).hexdigest() == claim.sha256


def test_manifest_binds_attempt_snapshot_and_upstream(project_root):
    manifest = load_manifest(project_root)
    assert manifest["attempt_id"] == "97e213a5-7ca3-4a1b-a500-1ec52d94d87a"
    assert manifest["paper_id"] == "vSzRJyg6k0"
    assert manifest["snapshot_id"] == (
        "09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199"
    )
    assert manifest["upstream_revision"] == (
        "arxiv:2602.02495v3+"
        "github:PeterLauLukChen/RACO@84a943c34f38520c7e0c9dd3066517c111b3c8fa"
    )
```

- [ ] **Step 2: Run RED**

Run:

```bash
cd submissions/reward-free-alignment-for-conflicting-objectives
env UV_CACHE_DIR=/tmp/raco-uv-cache uv run pytest -q tests/test_provenance.py
```

Expected: collection fails because the provenance module and input registries
do not exist.

- [ ] **Step 3: Add the exact ten claim records and fail-closed verifier**

`live_claims.json` contains the ten exact text/hash pairs from the design,
without normalization. Set `targeted` only for ordinals 6–9.

The upstream manifest records the admitted identity, arXiv v3 URL/license,
repository URL/commit/license, and only the repository files required for the
solver/source audit with exact size, Git blob, and SHA-256. The verifier:

```python
def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise IntegrityError(f"unsafe relative path: {value}")
    return path


def verify_bytes(payload: bytes, expected_sha256: str, expected_size: int) -> None:
    if len(payload) != expected_size:
        raise IntegrityError("artifact size mismatch")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise IntegrityError("artifact SHA-256 mismatch")
```

Reject duplicate JSON keys, duplicate claim ordinals/hashes, wrong source
identity, unsafe paths, missing files, hash drift, and Git-blob drift.

- [ ] **Step 4: Run GREEN and commit**

```bash
env UV_CACHE_DIR=/tmp/raco-uv-cache uv run pytest -q tests/test_provenance.py
git diff --check
git add pyproject.toml src/reward_free_alignment/provenance.py evidence/inputs tests/test_provenance.py
git commit -m "evidence: bind RACO snapshot and live claims"
```

---

### Task 2: Objective-specific reward-free pairwise losses

**Files:**

- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/src/reward_free_alignment/pairwise.py`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/tests/test_pairwise.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PairwiseBatch:
    chosen_logp: Tensor
    rejected_logp: Tensor
    reference_chosen_logp: Tensor
    reference_rejected_logp: Tensor

pairwise_logistic_loss(batch: PairwiseBatch, beta: float) -> Tensor
objective_losses(batches: Sequence[PairwiseBatch], beta: float) -> Tensor
objective_gradients(
    losses: Tensor, parameters: Sequence[Tensor]
) -> tuple[Tensor, ...]
```

- [ ] **Step 1: Write failing closed-form and separation tests**

```python
def test_pairwise_loss_matches_closed_form():
    batch = PairwiseBatch(
        tensor([-0.2]), tensor([-0.8]), tensor([-0.4]), tensor([-0.6])
    )
    loss = pairwise_logistic_loss(batch, beta=0.5)
    assert torch.allclose(loss, -logsigmoid(tensor([0.2])).mean())


def test_objective_losses_are_not_scalarized():
    losses = objective_losses((fixture_a(), fixture_b()), beta=0.5)
    assert losses.shape == (2,)
    assert not torch.equal(losses[0], losses[1])
```

- [ ] **Step 2: Run RED**

```bash
env UV_CACHE_DIR=/tmp/raco-uv-cache uv run pytest -q tests/test_pairwise.py
```

Expected: import failure because `pairwise.py` does not exist.

- [ ] **Step 3: Implement the exact stable loss**

```python
def pairwise_logistic_loss(batch: PairwiseBatch, beta: float) -> Tensor:
    beta = validate_positive_finite("beta", beta)
    validate_equal_shapes(batch)
    policy_gap = batch.chosen_logp - batch.rejected_logp
    reference_gap = batch.reference_chosen_logp - batch.reference_rejected_logp
    return -torch.nn.functional.logsigmoid(beta * (policy_gap-reference_gap)).mean()
```

`objective_losses` applies the function independently to every objective.
`objective_gradients` flattens each objective's gradient separately and
rejects missing or non-finite gradients.

- [ ] **Step 4: Run GREEN and commit**

```bash
env UV_CACHE_DIR=/tmp/raco-uv-cache uv run pytest -q tests/test_pairwise.py
git diff --check
git add src/reward_free_alignment/pairwise.py tests/test_pairwise.py
git commit -m "feat: implement objective-specific RACO losses"
```

---

### Task 3: Weighted two-objective alpha solver and coordinate clipping

**Files:**

- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/src/reward_free_alignment/cagrad_clip.py`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/tests/test_cagrad_clip.py`
- Modify:
  `submissions/reward-free-alignment-for-conflicting-objectives/src/reward_free_alignment/raco.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class AlphaSolution:
    alpha: float
    coefficients: Tensor
    weighted_anchor: Tensor
    objective_value: float
    candidate_count: int
    singular_case: str | None

@dataclass(frozen=True)
class CAGradResult:
    gradient: Tensor
    weighted_anchor: Tensor
    coefficients: Tensor
    clipped_coefficients: Tensor
    mixture: Tensor
    clipped_mixture: Tensor
    clipped_coordinates: tuple[int, ...]
    singular_case: str | None

solve_two_objective_alpha(
    g1: Tensor,
    g2: Tensor,
    weights: Tensor,
    c: float,
    atol: float = 1e-12,
) -> AlphaSolution

cagrad_clip(
    gradients: Sequence[Tensor],
    weights: Tensor,
    c: float,
    atol: float = 1e-12,
) -> CAGradResult
```

- [ ] **Step 1: Write failing weighted-anchor and clipping tests**

```python
def test_solver_uses_asymmetric_user_weight_anchor():
    g1, g2 = tensor([1.0, 0.0]), tensor([0.0, 2.0])
    weights = tensor([0.8, 0.2])
    solution = solve_two_objective_alpha(g1, g2, weights, c=0.4)
    assert torch.allclose(solution.weighted_anchor, tensor([0.8, 0.4]))
    assert 0.0 <= solution.alpha <= 1.0
    assert torch.allclose(
        solution.coefficients, tensor([solution.alpha, 1.0-solution.alpha])
    )


def test_clipping_is_coordinatewise_and_not_renormalized():
    result = cagrad_clip(
        (tensor([1.0, -1.0]), tensor([-0.2, 1.0])),
        weights=tensor([0.1, 0.9]),
        c=0.5,
    )
    expected = torch.minimum(result.coefficients, tensor([0.1, 0.9]))
    assert torch.allclose(result.clipped_coefficients, expected)
    assert result.clipped_coefficients.sum() <= 1.0
    assert torch.allclose(
        result.clipped_mixture,
        expected[0]*tensor([1.0, -1.0])+expected[1]*tensor([-0.2, 1.0]),
    )
```

- [ ] **Step 2: Add failing singular-case tests**

```python
@pytest.mark.parametrize(
    ("g1", "g2", "weights", "c", "case"),
    [
        ([1.0, 2.0], [1.0, 2.0], [0.7, 0.3], 0.4, "identical_gradients"),
        ([1.0, 0.0], [-1.0, 0.0], [0.5, 0.5], 0.4, "zero_anchor"),
        ([0.0, 0.0], [0.0, 0.0], [0.2, 0.8], 0.4, "zero_anchor"),
        ([1.0, 2.0], [2.0, 4.0], [0.5, 0.5], 0.4, "colinear_gradients"),
        ([1.0, 0.0], [0.0, 1.0], [0.6, 0.4], 0.0, "zero_radius"),
    ],
)
def test_singular_cases_are_finite_and_deterministic(g1, g2, weights, c, case):
    first = cagrad_clip((tensor(g1), tensor(g2)), tensor(weights), c)
    second = cagrad_clip((tensor(g1), tensor(g2)), tensor(weights), c)
    assert torch.isfinite(first.gradient).all()
    assert torch.equal(first.gradient, second.gradient)
    assert first.singular_case == case


def test_zero_weight_coordinate_cannot_be_reintroduced():
    result = cagrad_clip(
        (tensor([1.0, 0.0]), tensor([0.0, 1.0])),
        weights=tensor([1.0, 0.0]),
        c=0.4,
    )
    assert result.clipped_coefficients[1].item() == 0.0
```

- [ ] **Step 3: Run RED**

```bash
env UV_CACHE_DIR=/tmp/raco-uv-cache uv run pytest -q tests/test_cagrad_clip.py
```

Expected: tests fail against the fixed-alpha toy implementation.

- [ ] **Step 4: Implement the paper's weighted alpha subproblem**

Compute:

```python
g0 = weights[0] * g1 + weights[1] * g2
b1, b2 = torch.dot(g1, g0), torch.dot(g2, g0)
delta = b1 - b2
q2 = torch.dot(g1-g2, g1-g2)
q1 = 2.0 * (torch.dot(g1, g2)-torch.dot(g2, g2))
q0 = torch.dot(g2, g2)
s = c * torch.linalg.vector_norm(g0)
```

Enumerate endpoints, roots of `Q(alpha)=0`, and valid roots of the paper's
stationary quadratic. Handle quadratic, linear, and constant polynomials
without division by zero. Clamp only values within `atol` of `[0,1]`; reject
non-finite candidates. Evaluate every candidate in:

```python
def h(alpha):
    mixture = alpha*g1 + (1.0-alpha)*g2
    return torch.dot(mixture, g0) + s*torch.linalg.vector_norm(mixture)
```

Choose minimum `h`, then closest to `weights[0]`, then smallest alpha.
Implement the singular decisions from the design before general root solving.

Apply:

```python
clipped = torch.minimum(coefficients, weights)
clipped_mixture = clipped[0]*g1 + clipped[1]*g2
if torch.linalg.vector_norm(clipped_mixture) <= atol:
    gradient = g0
else:
    gradient = (
        g0
        + c*torch.linalg.vector_norm(g0)
        * clipped_mixture/torch.linalg.vector_norm(clipped_mixture)
    )
```

Do not divide by `clipped.sum()` or otherwise renormalize.

- [ ] **Step 5: Run GREEN and commit**

```bash
env UV_CACHE_DIR=/tmp/raco-uv-cache uv run pytest -q tests/test_pairwise.py tests/test_cagrad_clip.py
git diff --check
git add src/reward_free_alignment/cagrad_clip.py src/reward_free_alignment/raco.py tests/test_cagrad_clip.py
git commit -m "fix: reproduce weighted CAGrad-Clip"
```

---

### Task 4: Theorem 3.1 and per-step Theorem 3.2 audits

**Files:**

- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/src/reward_free_alignment/theorem_audit.py`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/tests/test_theorem_audit.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ConvergenceAudit:
    weights_in_simplex: bool
    smoothness_constants: tuple[float, ...]
    weighted_smoothness: float
    step_size: float
    correction_radius: float
    nonnegative_losses: bool
    descent_bound_holds: bool
    pareto_bound_holds: bool
    local_outcome: str

@dataclass(frozen=True)
class DescentCertificateAudit:
    rho: float | None
    rho_tilde: float | None
    gamma: float | None
    gamma_tilde: float | None
    observed_difference: float | None
    identity_rhs: float | None
    identity_residual: float | None
    strict_conditions: dict[str, bool]
    strict_expected: bool
    applicable: bool
    local_outcome: str

gamma(rho: float, c: float, weighted_smoothness: float, step_size: float) -> float
audit_theorem_31(case: SmoothObjectiveCase) -> ConvergenceAudit
audit_theorem_32(
    result: CAGradResult,
    weights: Tensor,
    c: float,
    weighted_smoothness: float,
    step_size: float,
    atol: float = 1e-10,
) -> DescentCertificateAudit
```

- [ ] **Step 1: Write failing Theorem 3.1 precondition tests**

```python
def test_theorem_31_records_and_checks_every_precondition():
    audit = audit_theorem_31(smooth_nonnegative_quadratic_case())
    assert audit.weights_in_simplex is True
    assert all(value > 0.0 for value in audit.smoothness_constants)
    assert 0.0 < audit.step_size <= 1.0/audit.weighted_smoothness
    assert 0.0 <= audit.correction_radius < 1.0
    assert audit.nonnegative_losses is True
    assert audit.descent_bound_holds is True
    assert audit.pareto_bound_holds is True
    assert audit.local_outcome == "supported"
```

- [ ] **Step 2: Write failing Theorem 3.2 identity tests**

```python
def test_theorem_32_reproduces_per_step_certificate_identity():
    weights = tensor([0.05, 0.95])
    result = cagrad_clip(
        (tensor([1.0, -1.76]), tensor([-1.0, 0.24])),
        weights,
        c=0.5,
    )
    audit = audit_theorem_32(
        result, weights, c=0.5, weighted_smoothness=4.0, step_size=0.05
    )
    expected = 0.5*(1.0-4.0*0.05)*(audit.rho_tilde-audit.rho)
    assert abs(audit.observed_difference-expected) <= 1e-10
    assert audit.identity_residual <= 1e-10
    assert audit.applicable is True


def test_strictness_requires_every_paper_condition():
    audit = strict_witness_audit()
    assert audit.strict_conditions == {
        "two_objectives": True,
        "positive_weights": True,
        "positive_c": True,
        "strict_step_size": True,
        "nonzero_anchor": True,
        "noncolinear_gradients": True,
        "interior_coefficients": True,
        "coefficients_differ_from_weights": True,
    }
    assert audit.strict_expected is True
    assert audit.observed_difference > 0.0


def test_zero_anchor_is_not_applicable_not_divided():
    audit = zero_anchor_audit()
    assert audit.applicable is False
    assert audit.rho is None and audit.rho_tilde is None
    assert audit.local_outcome == "limited"
```

- [ ] **Step 3: Run RED**

```bash
env UV_CACHE_DIR=/tmp/raco-uv-cache uv run pytest -q tests/test_theorem_audit.py
```

Expected: collection fails because the corrected audit interfaces do not
exist.

- [ ] **Step 4: Implement the exact certificate, not an iteration proxy**

```python
def gamma(rho, c, weighted_smoothness, step_size):
    return (
        1.0+c*rho
        - (weighted_smoothness*step_size/2.0)
        * (1.0+c*c+2.0*c*rho)
    )


observed = gamma(rho_tilde, c, ell_w, eta) - gamma(rho, c, ell_w, eta)
identity_rhs = c*(1.0-ell_w*eta)*(rho_tilde-rho)
```

The audit computes `rho` from the normalized *unclipped* mixture and
`rho_tilde` from the normalized clipped mixture. It records all eight
strictness Booleans shown in the test. Strict improvement is expected only
when all are true. Do not count iterations, choose a stopping threshold, or
describe a lower iteration count as Theorem 3.2 evidence.

For Theorem 3.1, use exact-gradient nonnegative quadratic and DPO fixtures,
record every precondition, and test the one-step and finite-horizon
inequalities from the design. Counterexamples outside the preconditions are
retained with `limited`; in-domain residual failures are `not-supported`.

- [ ] **Step 5: Run GREEN and commit**

```bash
env UV_CACHE_DIR=/tmp/raco-uv-cache uv run pytest -q tests/test_theorem_audit.py tests/test_cagrad_clip.py
git diff --check
git add src/reward_free_alignment/theorem_audit.py tests/test_theorem_audit.py
git commit -m "feat: audit RACO descent certificates"
```

---

### Task 5: Canonical evidence, direct root pages, and final proposal

**Files:**

- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/schema/evidence-v1.schema.json`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/src/reward_free_alignment/evidence.py`
- Modify:
  `submissions/reward-free-alignment-for-conflicting-objectives/src/reward_free_alignment/generate_evidence.py`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/tests/test_evidence.py`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/tests/test_pages.py`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/pages/00-summary.md`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/pages/01-objective-losses.md`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/pages/02-cagrad-clip.md`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/pages/03-theorem-31.md`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/pages/04-theorem-32.md`
- Create:
  `submissions/reward-free-alignment-for-conflicting-objectives/pages/05-limitations-and-provenance.md`
- Modify:
  `submissions/reward-free-alignment-for-conflicting-objectives/README.md`
- Modify:
  `submissions/reward-free-alignment-for-conflicting-objectives/app.py`

**Interfaces:**

```python
build_evidence(project_root: Path) -> dict[str, object]
validate_evidence(value: object, schema_path: Path) -> None
canonical_json(value: object) -> bytes
write_evidence_atomic(path: Path, value: object) -> None
```

- [ ] **Step 1: Write failing binding, outcome, and determinism tests**

```python
def test_bundle_contains_all_ten_live_claims_in_exact_order(project_root):
    evidence = build_evidence(project_root)
    assert evidence["snapshot_id"] == (
        "09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199"
    )
    assert tuple(c["sha256"] for c in evidence["claims"]) == EXPECTED_HASHES
    assert len(evidence["claims"]) == 10
    assert [c["targeted"] for c in evidence["claims"]] == [
        False, False, False, False, False, True, True, True, True, False
    ]


def test_local_outcomes_never_impersonate_official_verdicts(project_root):
    evidence = build_evidence(project_root)
    assert {c["local_outcome"] for c in evidence["claims"]} <= {
        "supported", "not-supported", "limited"
    }
    assert not {"verified", "falsified", "toy"} & {
        c["local_outcome"] for c in evidence["claims"]
    }


def test_evidence_is_canonical_and_byte_deterministic(project_root):
    first = canonical_json(build_evidence(project_root))
    second = canonical_json(build_evidence(project_root))
    assert first == second
    assert first.endswith(b"\n")
    assert b"NaN" not in first and b"Infinity" not in first
```

- [ ] **Step 2: Write failing direct-page and offline-app tests**

```python
EXPECTED_PAGES = (
    "00-summary.md",
    "01-objective-losses.md",
    "02-cagrad-clip.md",
    "03-theorem-31.md",
    "04-theorem-32.md",
    "05-limitations-and-provenance.md",
)


def test_root_pages_are_direct_complete_and_judge_readable(project_root):
    pages = tuple(sorted(path.name for path in (project_root / "pages").glob("*.md")))
    assert pages == EXPECTED_PAGES
    summary = (project_root / "pages/00-summary.md").read_text("utf-8")
    for digest in EXPECTED_HASHES:
        assert digest in summary
    assert "not an official verdict" in summary.lower()


def test_app_loads_only_committed_pages_and_evidence(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", fail_network)
    app = load_app()
    assert app.EVIDENCE["paper_id"] == "vSzRJyg6k0"
    assert tuple(path.name for path in app.PAGE_PATHS) == EXPECTED_PAGES
```

- [ ] **Step 3: Run RED**

```bash
env UV_CACHE_DIR=/tmp/raco-uv-cache uv run pytest -q tests/test_evidence.py tests/test_pages.py
```

Expected: failures because the closed evidence builder and complete direct
pages do not exist.

- [ ] **Step 4: Implement canonical outputs and reviewer surfaces**

The evidence schema sets `additionalProperties: false` on every owned object,
requires all ten claims, and restricts `local_outcome` to the three local
labels. Claims 3–5 and 10 are `limited`; no paper table value enters
`measurements`.

```python
def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
```

Each root page includes exact claim text/hash, local outcome, formula,
recomputed observation, source pin, command, and limitation. `app.py` loads
`evidence/results.json` and sorted pages at import without running audits or
network calls. README Space metadata includes `app_file: app.py`,
`paper-vSzRJyg6k0`, and `icml2026-repro`.

- [ ] **Step 5: Generate twice, compare bytes, and validate**

```bash
env UV_CACHE_DIR=/tmp/raco-uv-cache uv sync --frozen
env UV_CACHE_DIR=/tmp/raco-uv-cache uv run python -m reward_free_alignment.generate_evidence --output /tmp/raco-evidence-a.json
env UV_CACHE_DIR=/tmp/raco-uv-cache uv run python -m reward_free_alignment.generate_evidence --output /tmp/raco-evidence-b.json
cmp /tmp/raco-evidence-a.json /tmp/raco-evidence-b.json
cp /tmp/raco-evidence-a.json evidence/results.json
env UV_CACHE_DIR=/tmp/raco-uv-cache uv run pytest -q
git diff --check
```

From the repository root:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q --ignore=submissions/nape
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache PRE_COMMIT_HOME=/tmp/icml-repro-pre-commit uv run pre-commit run -a
```

Expected: evidence bytes compare equal, every project test passes, strict JSON
validation passes, root checks pass without running archival NAPE, and
pre-commit passes. If unrelated concurrent changes break a root check, record
the exact output as a concern; do not edit those files.

- [ ] **Step 6: Commit and hand off as a worker proposal**

```bash
git add schema src tests evidence README.md app.py pages
git commit -m "evidence: reproduce weighted RACO certificates"
git status --short
```

Report the commit, source tree, commands, test counts, result SHA-256, evidence
paths, and concerns. Request controller validation. Do not deploy, submit,
poll, mutate state, or claim an official verdict.
