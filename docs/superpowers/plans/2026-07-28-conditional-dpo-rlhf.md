# Conditional DPO/RLHF Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic CPU evidence that independently audits five
finite-response DPO, RLHF, and CPO claims from `arxiv:2605.20834v1`, with a
machine-readable bundle and static root pages.

**Architecture:** A standard-library numerical core evaluates analytic
identities and declared finite grids without training a model. Five lane
functions translate those calculations into typed observations; one canonical
evidence builder binds them to the exact challenge claims, and a thin CLI
generates and validates the bundle atomically. Root `index.html` and
`poster.html` render only the committed `evidence.json`.

**Tech Stack:** Python 3.11+, Python standard library (`argparse`,
`dataclasses`, `hashlib`, `json`, `math`, `os`, `pathlib`, `platform`,
`tempfile`), `pytest`, `uv`, JSON Schema implemented by the project validator,
HTML/CSS/JavaScript with no external assets.

## Global Constraints

- Work only in
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/`
  inside the controller-assigned worktree.
- Paper ID is `7UEBX1KU1y`; attempt ID is
  `933665ed-b7ed-4d73-9b07-35704660a184`; admitted snapshot is
  `09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199`.
- Pin only `arxiv:2605.20834v1`. Do not invent a revision for the advertised
  `visitworld123/CPO` repository, which returned HTTP 404 during assessment.
- Execute exactly five selected mathematical lanes. Keep the sixth SOTA
  benchmark claim in evidence as `not_reproduced`.
- Treat paper equations as attributed context. Only values computed by this
  project are reproduced observations.
- Keep the finite RLHF optimum, one-sided Equation 8 DPO loss, and full
  Bradley–Terry population cross-entropy distinct.
- Use the exact finite grids and `1e-12` probability-sum tolerance specified
  in the approved design.
- Reject non-finite values, invalid probabilities, `beta <= 0`, and
  `gamma < 0`.
- Runtime code uses only the Python standard library. `pytest` is the only
  development dependency.
- All tests and evidence generation are CPU-only, offline, deterministic, and
  require no paid API.
- Use red-green-refactor for every production behavior.
- Local evidence outcomes are `consistent`, `contradiction`, `mixed`, and
  `not_reproduced`; never present them as official challenge verdicts.
- Root `index.html` and `poster.html` fetch only `./evidence.json` and use no
  external assets.
- Workers do not mutate coordinator state, the reproduction-loop skill,
  another submission, `docs/HANDOFF.md`, Hub resources, submissions, or
  verdicts. Every implementation commit is a proposal for controller review.

---

### Task 1: Project identity, source pin, and closed claim bindings

**Files:**

- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/pyproject.toml`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/src/conditional_dpo_repro/__init__.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/src/conditional_dpo_repro/claims.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/sources/paper.json`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/tests/conftest.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/tests/test_identity.py`
- Generate:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/uv.lock`

**Interfaces:**

- Consumes: exact attempt identity and six live claims from the approved
  design.
- Produces:
  `ClaimBinding`, `load_source_record(path: Path) -> dict[str, object]`,
  `load_claim_bindings(path: Path) -> tuple[ClaimBinding, ...]`, and constants
  `PAPER_ID`, `ATTEMPT_ID`, `SNAPSHOT_ID`, `UPSTREAM_REVISION`.

Use these admitted `(challenge_claim, SHA-256)` pairs in exactly this order:

1. `The paper proves DPO-RLHF equivalence is conditional on the RLHF-optimal policy preferring human-preferred responses (Section 3).`
   — `588c9334124771dc2ff7fc51494f4328329ab13dc21d4522a0e91b6f6417240a`
2. `When the equivalence assumption fails, DPO optimizes relative advantage over the reference policy rather than absolute human-preference alignment (Section 3).`
   — `4820743d0eac6cc30b4a75d2be41f49193b0ea4ad4168bea2200a9f16cc77a86`
3. `The paper characterizes undesirable solution spaces in which policies reduce DPO loss while preferring dispreferred responses (Section 3).`
   — `6c26fe711e2f10b44cb933b89b12982fef3cf3bcc760668a0b0fa9d15e1965dc`
4. `Constrained Preference Optimization augments RLHF with constraints and derives a stationary DPO-like loss with an adaptive reference-based margin (Section 4.3).`
   — `a80267886061211c131041549df22264e0c713a9759a76f0ab37bac69a436af1`
5. `The paper gives a soft-margin ranking interpretation showing DPO can implement margin ranking with potentially negative targets (Section 5).`
   — `7d797875f18478f305a8dc08d860a29ba4f15c3b97fb4c9d41e55363975553be`
6. `Experiments on standard benchmarks report state-of-the-art performance for CPO (Section 6).`
   — `8df1fece656f02adbdf85fb78bc8993591f1abc9ee78c957388ab4b4eac37dcd`

- [ ] **Step 1: Write failing identity tests**

```python
from hashlib import sha256

from conditional_dpo_repro.claims import (
    ATTEMPT_ID,
    LIVE_CLAIM_HASHES,
    LIVE_CLAIMS,
    PAPER_ID,
    SNAPSHOT_ID,
    UPSTREAM_REVISION,
    load_claim_bindings,
)

EXPECTED_LIVE_CLAIMS = (
    "The paper proves DPO-RLHF equivalence is conditional on the RLHF-optimal "
    "policy preferring human-preferred responses (Section 3).",
    "When the equivalence assumption fails, DPO optimizes relative advantage "
    "over the reference policy rather than absolute human-preference alignment "
    "(Section 3).",
    "The paper characterizes undesirable solution spaces in which policies "
    "reduce DPO loss while preferring dispreferred responses (Section 3).",
    "Constrained Preference Optimization augments RLHF with constraints and "
    "derives a stationary DPO-like loss with an adaptive reference-based margin "
    "(Section 4.3).",
    "The paper gives a soft-margin ranking interpretation showing DPO can "
    "implement margin ranking with potentially negative targets (Section 5).",
    "Experiments on standard benchmarks report state-of-the-art performance "
    "for CPO (Section 6).",
)
EXPECTED_LIVE_CLAIM_HASHES = (
    "588c9334124771dc2ff7fc51494f4328329ab13dc21d4522a0e91b6f6417240a",
    "4820743d0eac6cc30b4a75d2be41f49193b0ea4ad4168bea2200a9f16cc77a86",
    "6c26fe711e2f10b44cb933b89b12982fef3cf3bcc760668a0b0fa9d15e1965dc",
    "a80267886061211c131041549df22264e0c713a9759a76f0ab37bac69a436af1",
    "7d797875f18478f305a8dc08d860a29ba4f15c3b97fb4c9d41e55363975553be",
    "8df1fece656f02adbdf85fb78bc8993591f1abc9ee78c957388ab4b4eac37dcd",
)


def test_identity_is_bound_to_admitted_attempt():
    assert PAPER_ID == "7UEBX1KU1y"
    assert ATTEMPT_ID == "933665ed-b7ed-4d73-9b07-35704660a184"
    assert SNAPSHOT_ID == (
        "09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199"
    )
    assert UPSTREAM_REVISION == "arxiv:2605.20834v1"


def test_all_six_live_claims_equal_admitted_constants(project_root):
    claims = load_claim_bindings(project_root / "sources/paper.json")
    assert LIVE_CLAIMS == EXPECTED_LIVE_CLAIMS
    assert LIVE_CLAIM_HASHES == EXPECTED_LIVE_CLAIM_HASHES
    assert tuple(item.challenge_claim for item in claims) == EXPECTED_LIVE_CLAIMS
    assert tuple(
        item.challenge_claim_sha256 for item in claims
    ) == EXPECTED_LIVE_CLAIM_HASHES
    assert sum(item.targeted for item in claims) == 5
    assert tuple(
        sha256(text.encode("utf-8")).hexdigest() for text in EXPECTED_LIVE_CLAIMS
    ) == EXPECTED_LIVE_CLAIM_HASHES
    assert claims[-1].targeted is False
```

The expected tuples above are deliberately duplicated in the test. Do not
derive either tuple from `sources/paper.json`, `ClaimBinding` objects, or the
other tuple under test.

- [ ] **Step 2: Run the identity tests and verify RED**

Run:

```bash
cd submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_identity.py
```

Expected: collection fails because `conditional_dpo_repro.claims` does not
exist.

- [ ] **Step 3: Create the package and exact source record**

Use this project metadata:

```toml
[project]
name = "conditional-dpo-repro"
version = "0.1.0"
description = "Finite-response reproduction of conditional DPO/RLHF claims"
requires-python = ">=3.11"
dependencies = []

[project.scripts]
conditional-dpo-repro = "conditional_dpo_repro.cli:main"

[dependency-groups]
dev = ["pytest>=8.4,<9"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/conditional_dpo_repro"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

Define the immutable identity and strict record loader:

```python
PAPER_ID = "7UEBX1KU1y"
ATTEMPT_ID = "933665ed-b7ed-4d73-9b07-35704660a184"
SNAPSHOT_ID = (
    "09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199"
)
UPSTREAM_REVISION = "arxiv:2605.20834v1"
LIVE_CLAIMS = (
    "The paper proves DPO-RLHF equivalence is conditional on the RLHF-optimal "
    "policy preferring human-preferred responses (Section 3).",
    "When the equivalence assumption fails, DPO optimizes relative advantage "
    "over the reference policy rather than absolute human-preference alignment "
    "(Section 3).",
    "The paper characterizes undesirable solution spaces in which policies "
    "reduce DPO loss while preferring dispreferred responses (Section 3).",
    "Constrained Preference Optimization augments RLHF with constraints and "
    "derives a stationary DPO-like loss with an adaptive reference-based margin "
    "(Section 4.3).",
    "The paper gives a soft-margin ranking interpretation showing DPO can "
    "implement margin ranking with potentially negative targets (Section 5).",
    "Experiments on standard benchmarks report state-of-the-art performance "
    "for CPO (Section 6).",
)
LIVE_CLAIM_HASHES = (
    "588c9334124771dc2ff7fc51494f4328329ab13dc21d4522a0e91b6f6417240a",
    "4820743d0eac6cc30b4a75d2be41f49193b0ea4ad4168bea2200a9f16cc77a86",
    "6c26fe711e2f10b44cb933b89b12982fef3cf3bcc760668a0b0fa9d15e1965dc",
    "a80267886061211c131041549df22264e0c713a9759a76f0ab37bac69a436af1",
    "7d797875f18478f305a8dc08d860a29ba4f15c3b97fb4c9d41e55363975553be",
    "8df1fece656f02adbdf85fb78bc8993591f1abc9ee78c957388ab4b4eac37dcd",
)


@dataclass(frozen=True)
class ClaimBinding:
    challenge_claim: str
    challenge_claim_sha256: str
    target_claim: str | None
    targeted: bool
    equations: tuple[str, ...]


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_source_record(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if value["paper_id"] != PAPER_ID:
        raise ValueError("paper identity mismatch")
    if value["upstream_revision"] != UPSTREAM_REVISION:
        raise ValueError("paper revision mismatch")
    return value
```

Populate `sources/paper.json` with the six exact challenge strings and hashes
from the design, in order. Set `targeted: true` for the first five and
`targeted: false`, `target_claim: null`, and `equations: []` for the SOTA
claim. Record the versioned arXiv abstract, HTML, and e-print URLs, the arXiv
license label, the unavailable repository URL, and these equation bindings:

```text
lane 1: [3, 5, 8, 33-43]
lane 2: [5, 8, 35]
lane 3: [8, 35]
lane 4: [10-19, 21, 59-74]
lane 5: [27-28, 120-131]
```

- [ ] **Step 4: Lock and run GREEN**

Run:

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv lock
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_identity.py
```

Expected: both identity tests pass and `uv.lock` contains no runtime numerical
or ML dependency.

- [ ] **Step 5: Commit identity and provenance**

```bash
git add pyproject.toml uv.lock sources/paper.json src/conditional_dpo_repro/__init__.py src/conditional_dpo_repro/claims.py tests/conftest.py tests/test_identity.py
git commit -m "evidence: bind conditional DPO paper identity"
```

---

### Task 2: Stable two-response mathematical core

**Files:**

- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/src/conditional_dpo_repro/math.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/tests/test_math.py`

**Interfaces:**

- Consumes: finite floats.
- Produces: `TwoResponsePolicy`, `policy_from_delta`, `softplus`,
  `sigmoid`, `logit`, `rlhf_optimal_delta`, `dpo_loss`,
  `dpo_loss_derivative`, `bt_population_loss`,
  `constrained_rlhf_objective`, `solve_exact_constrained_rlhf`,
  `cpo_optimal_policy_margin`, `cpo_reference_margin`, `cpo_loss`,
  `cpo_loss_derivative`, and `scaled_dpo_soft_margin`.

- [ ] **Step 1: Write failing primitive tests**

```python
@pytest.mark.parametrize("delta", [-8.0, -1.0, 0.0, 1.0, 8.0])
def test_policy_delta_round_trip(delta):
    policy = policy_from_delta(delta)
    assert abs(policy.delta - delta) <= 1e-12
    assert abs(policy.preferred + policy.dispreferred - 1.0) <= 1e-12


def test_dpo_derivative_is_strictly_negative():
    assert dpo_loss_derivative(delta=-2.0, delta_ref=-3.0, beta=1.0) < 0.0
    assert dpo_loss(-2.0, -3.0, 1.0) < dpo_loss(-3.0, -3.0, 1.0)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((float("nan"), 0.0, 1.0), "finite"),
        ((0.0, 0.0, 0.0), "beta"),
    ],
)
def test_invalid_dpo_inputs_fail_closed(arguments, message):
    with pytest.raises(ValueError, match=message):
        dpo_loss(*arguments)


def test_exact_constrained_rlhf_solver_recovers_unconstrained_case():
    reference = TwoResponsePolicy(0.25, 0.75)
    exact = solve_exact_constrained_rlhf(reference, 0.5, 1.0, gamma=0.0)
    assert exact["status"] == "finite_optimum"
    assert abs(
        exact["policy"].delta
        - rlhf_optimal_delta(reference.delta, 0.5, 1.0)
    ) <= 1e-12


def test_exact_constrained_rlhf_solver_certifies_positive_gamma_boundary():
    reference = TwoResponsePolicy(0.25, 0.75)
    exact = solve_exact_constrained_rlhf(reference, 0.5, 1.0, gamma=0.10)
    assert exact["status"] == "unbounded"
    assert exact["approached_boundary"] == "preferred"
```

- [ ] **Step 2: Run mathematical tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_math.py
```

Expected: collection fails because `conditional_dpo_repro.math` does not
exist.

- [ ] **Step 3: Implement stable primitives**

```python
PROBABILITY_TOLERANCE = 1e-12


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def softplus(value: float) -> float:
    value = _finite("softplus input", value)
    if value > 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def sigmoid(value: float) -> float:
    value = _finite("sigmoid input", value)
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class TwoResponsePolicy:
    preferred: float
    dispreferred: float

    def __post_init__(self):
        preferred = _finite("preferred", self.preferred)
        dispreferred = _finite("dispreferred", self.dispreferred)
        if not 0.0 < preferred < 1.0 or not 0.0 < dispreferred < 1.0:
            raise ValueError("policy probabilities must lie in (0, 1)")
        if abs(preferred + dispreferred - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError("policy probabilities must sum to one")

    @property
    def delta(self) -> float:
        return math.log(self.preferred) - math.log(self.dispreferred)
```

Implement the equations exactly:

```python
def dpo_loss(delta: float, delta_ref: float, beta: float) -> float:
    delta, delta_ref, beta = _loss_inputs(delta, delta_ref, beta)
    return softplus(-beta * (delta - delta_ref))


def dpo_loss_derivative(delta: float, delta_ref: float, beta: float) -> float:
    delta, delta_ref, beta = _loss_inputs(delta, delta_ref, beta)
    return -beta * sigmoid(-beta * (delta - delta_ref))


def bt_population_loss(
    delta: float, delta_ref: float, reward_gap: float, beta: float
) -> float:
    delta, delta_ref, beta = _loss_inputs(delta, delta_ref, beta)
    reward_gap = _finite("reward_gap", reward_gap)
    q = sigmoid(reward_gap)
    model_logit = beta * (delta - delta_ref)
    return q * softplus(-model_logit) + (1.0 - q) * softplus(model_logit)


def cpo_reference_margin(reference: TwoResponsePolicy, gamma: float) -> float:
    gamma = _finite("gamma", gamma)
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    return gamma * (1.0 / reference.preferred + 1.0 / reference.dispreferred)
```

Implement `constrained_rlhf_objective` directly from the exact constrained
RLHF objective before implementing either margin. In the two-response,
single-preference-pair reduction, use the preferred reward gap with a zero
dispreferred baseline, exact Bernoulli KL to the reference, and
`gamma * policy.delta`. `solve_exact_constrained_rlhf` must return a closed
status enum and certificates:

```text
finite_optimum: policy, objective, first_order_residual, curvature
boundary_supremum: approached_boundary, monotone_tail_values
unbounded: approached_boundary, increasing_tail_values, analytic_reason
```

For `gamma == 0`, independently recover the vanilla finite RLHF optimum. For
`gamma > 0`, check the objective on a declared increasing delta sequence and
apply the analytic boundary limit; never relabel an interior stationary point
as a global optimum. `cpo_optimal_policy_margin(policy, gamma)` evaluates the
Equations 13–16 expression with the certified optimal policy. Only
`cpo_reference_margin(reference, gamma)` performs the Equation 17
reference-policy substitution.

- [ ] **Step 4: Run GREEN and commit**

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_math.py
git diff --check
git add src/conditional_dpo_repro/math.py tests/test_math.py
git commit -m "feat: add finite preference math core"
```

Expected: all primitive, stability, and validation tests pass.

---

### Task 3: Equivalence, relative-advantage, and undesirable-space lanes

**Files:**

- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/src/conditional_dpo_repro/grids.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/src/conditional_dpo_repro/equivalence.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/src/conditional_dpo_repro/failure_modes.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/tests/test_equivalence.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/tests/test_relative_advantage.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/tests/test_undesirable_space.py`

**Interfaces:**

- Consumes: Task 2 mathematical functions and immutable grid tuples.
- Produces: `centered_derivative`, `run_equivalence_lane`,
  `run_relative_advantage_lane`, and `run_undesirable_space_lane`.

- [ ] **Step 1: Write failing lane tests**

```python
def test_equivalence_lane_distinguishes_two_dpo_objectives():
    result = run_equivalence_lane()
    assert result["case_count"] == 112
    assert result["population_stationary_max_abs_error"] <= 1e-8
    assert result["positive_loss_derivative_max"] < 0.0
    assert result["one_sided_finite_optimum"] is False
    assert result["population_identity_requires_positive_delta"] is False
    assert result["outcome"] in {"mixed", "contradiction"}


def test_relative_lane_separates_relative_and_absolute_preference():
    result = run_relative_advantage_lane()
    assert result["case_count"] == 75
    assert result["relative_improvement_count"] == 75
    assert result["absolute_preference_count"] < 75
    assert result["relative_but_not_absolute_count"] > 0


def test_undesirable_lane_emits_concrete_witnesses():
    result = run_undesirable_space_lane()
    assert result["witness_count"] > 0
    for witness in result["witnesses"]:
        assert witness["delta_ref"] < witness["delta"] < 0.0
        assert witness["candidate_loss"] < witness["reference_loss"]
        assert witness["preferred_probability"] < 0.5
```

- [ ] **Step 2: Run all three tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_equivalence.py tests/test_relative_advantage.py tests/test_undesirable_space.py
```

Expected: collection fails because the three lane modules do not exist.

- [ ] **Step 3: Implement immutable grids and equivalence audit**

```python
EQUIVALENCE_P_REF_W = (0.05, 0.10, 0.25, 0.40, 0.50, 0.75, 0.90)
EQUIVALENCE_REWARD_GAPS = (0.10, 0.50, 1.00, 2.00)
EQUIVALENCE_BETAS = (0.25, 0.50, 1.00, 2.00)
RELATIVE_DELTAS_REF = (-4.0, -2.0, -1.0, -0.25, 0.25)
RELATIVE_OFFSETS = (0.10, 0.25, 0.50, 1.00, 2.00)
RELATIVE_BETAS = (0.50, 1.00, 2.00)


def centered_derivative(function, point: float, step: float = 1e-6) -> float:
    return (function(point + step) - function(point - step)) / (2.0 * step)


def run_equivalence_lane() -> dict[str, object]:
    rows = []
    for p_ref, reward_gap, beta in itertools.product(
        EQUIVALENCE_P_REF_W, EQUIVALENCE_REWARD_GAPS, EQUIVALENCE_BETAS
    ):
        delta_ref = TwoResponsePolicy(p_ref, 1.0 - p_ref).delta
        optimum = rlhf_optimal_delta(delta_ref, reward_gap, beta)
        derivative = centered_derivative(
            lambda delta: bt_population_loss(delta, delta_ref, reward_gap, beta),
            optimum,
        )
        rows.append(
            {
                "p_ref_w": p_ref,
                "reward_gap": reward_gap,
                "beta": beta,
                "delta_ref": delta_ref,
                "delta_rlhf": optimum,
                "condition_holds": optimum > 0.0,
                "population_stationary_abs_error": abs(derivative),
                "positive_loss_derivative": dpo_loss_derivative(
                    optimum, delta_ref, beta
                ),
            }
        )
    return _summarize_equivalence(rows)
```

`_summarize_equivalence` derives its outcome from these observations. It must
not assign a finite minimizer to the one-sided loss.

- [ ] **Step 4: Implement relative and undesirable audits**

```python
def _relative_rows():
    for delta_ref, offset, beta in itertools.product(
        RELATIVE_DELTAS_REF, RELATIVE_OFFSETS, RELATIVE_BETAS
    ):
        delta = delta_ref + offset
        yield {
            "delta_ref": delta_ref,
            "offset": offset,
            "delta": delta,
            "beta": beta,
            "relative_improvement": delta > delta_ref,
            "absolute_preference": delta > 0.0,
            "reference_loss": dpo_loss(delta_ref, delta_ref, beta),
            "candidate_loss": dpo_loss(delta, delta_ref, beta),
            "preferred_probability": policy_from_delta(delta).preferred,
        }


def run_relative_advantage_lane() -> dict[str, object]:
    rows = tuple(_relative_rows())
    return {
        "case_count": len(rows),
        "relative_improvement_count": sum(r["relative_improvement"] for r in rows),
        "absolute_preference_count": sum(r["absolute_preference"] for r in rows),
        "relative_but_not_absolute_count": sum(
            r["relative_improvement"] and not r["absolute_preference"] for r in rows
        ),
        "outcome": "consistent",
    }


def run_undesirable_space_lane() -> dict[str, object]:
    witnesses = tuple(
        row for row in _relative_rows() if row["delta_ref"] < row["delta"] < 0.0
    )
    return {
        "witness_count": len(witnesses),
        "witnesses": witnesses,
        "outcome": "consistent" if witnesses else "contradiction",
    }
```

- [ ] **Step 5: Run GREEN and commit**

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_equivalence.py tests/test_relative_advantage.py tests/test_undesirable_space.py
git diff --check
git add src/conditional_dpo_repro/grids.py src/conditional_dpo_repro/equivalence.py src/conditional_dpo_repro/failure_modes.py tests/test_equivalence.py tests/test_relative_advantage.py tests/test_undesirable_space.py
git commit -m "feat: audit DPO equivalence and failure modes"
```

Expected: 112 equivalence and 75 relative cases are evaluated, and every
undesirable witness has a decreasing DPO loss while preferring `l`.

---

### Task 4: Exact constrained-RLHF and approximate CPO-margin lane

**Files:**

- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/src/conditional_dpo_repro/cpo.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/tests/test_cpo_margin.py`

**Interfaces:**

- Consumes: `TwoResponsePolicy`, the exact constrained-RLHF objective and
  solver, both margin functions, and the design's 180-case CPO grid.
- Produces: `run_cpo_margin_lane() -> dict[str, object]`, with separate
  `exact_constrained_rlhf`, `equations_13_16_optimal_policy_margin`, and
  `equation_17_reference_policy_approximation` objects.

- [ ] **Step 1: Write failing exact-versus-approximate CPO tests**

```python
from itertools import pairwise


def test_exact_constrained_rlhf_gamma_zero_recovers_rlhf_optimum():
    reference = TwoResponsePolicy(0.25, 0.75)
    exact = solve_exact_constrained_rlhf(reference, 0.5, 1.0, gamma=0.0)
    assert exact["status"] == "finite_optimum"
    assert abs(
        exact["policy"].delta
        - rlhf_optimal_delta(reference.delta, 0.5, 1.0)
    ) <= 1e-12
    assert exact["first_order_residual"] <= 1e-12


def test_positive_gamma_exact_objective_is_not_replaced_by_equation_17():
    reference = TwoResponsePolicy(0.25, 0.75)
    exact = solve_exact_constrained_rlhf(reference, 0.5, 1.0, gamma=0.10)
    assert exact["status"] == "unbounded"
    assert exact["approached_boundary"] == "preferred"
    assert all(
        right > left
        for left, right in pairwise(exact["increasing_tail_values"])
    )
    approximation = cpo_reference_margin(reference, gamma=0.10)
    assert abs(approximation - (0.10 * (4.0 + 4.0 / 3.0))) <= 1e-12


def test_equations_13_16_margin_uses_certified_optimal_policy():
    optimal_policy = TwoResponsePolicy(0.80, 0.20)
    reference = TwoResponsePolicy(0.25, 0.75)
    exact_margin = cpo_optimal_policy_margin(optimal_policy, gamma=0.10)
    approximate_margin = cpo_reference_margin(reference, gamma=0.10)
    assert abs(exact_margin - 0.10 * (1.25 + 5.00)) <= 1e-12
    assert exact_margin != approximate_margin


def test_cpo_lane_keeps_exact_result_and_approximation_separate():
    result = run_cpo_margin_lane()
    assert result["case_count"] == 180
    assert result["exact_constrained_rlhf"]["finite_optimum_count"] == 45
    assert result["exact_constrained_rlhf"]["unbounded_count"] == 135
    assert result[
        "equations_13_16_optimal_policy_margin"
    ]["evaluated_only_for_certified_optima"] is True
    approximation = result["equation_17_reference_policy_approximation"]
    assert approximation["case_count"] == 180
    assert approximation["shift_identity_max_abs_error"] <= 1e-12
    assert approximation["stationary_derivative_max_abs_error"] <= 1e-8
    assert approximation["margin_parameter_derivative"] == 0.0
    assert approximation["labeled_approximation"] is True
```

- [ ] **Step 2: Run the CPO tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_cpo_margin.py
```

Expected: collection fails because `conditional_dpo_repro.cpo` does not exist.

- [ ] **Step 3: Implement the ordered three-stage CPO audit**

```python
CPO_P_REF_W = (0.10, 0.25, 0.50, 0.75, 0.90)
CPO_REWARD_GAPS = (0.10, 0.50, 1.00)
CPO_BETAS = (0.50, 1.00, 2.00)
CPO_GAMMAS = (0.00, 0.01, 0.05, 0.10)


def run_cpo_margin_lane() -> dict[str, object]:
    rows = []
    for p_ref, reward_gap, beta, gamma in itertools.product(
        CPO_P_REF_W, CPO_REWARD_GAPS, CPO_BETAS, CPO_GAMMAS
    ):
        reference = TwoResponsePolicy(p_ref, 1.0 - p_ref)

        # Stage 1: solve or classify the exact constrained-RLHF objective.
        exact = solve_exact_constrained_rlhf(
            reference, reward_gap, beta, gamma
        )

        # Stage 2: audit Equations 13–16 only for a certified global optimum.
        exact_margin = None
        exact_margin_residual = None
        if exact["status"] == "finite_optimum":
            exact_margin = cpo_optimal_policy_margin(exact["policy"], gamma)
            exact_margin_residual = equations_13_16_residual(
                exact["policy"],
                reference,
                reward_gap,
                beta,
                exact_margin,
            )

        # Stage 3: audit Equation 17 as a reference-policy approximation.
        reference_margin = cpo_reference_margin(reference, gamma)
        dpo_delta = rlhf_optimal_delta(reference.delta, reward_gap, beta)
        approximate_cpo_delta = dpo_delta + reference_margin / beta
        derivative = centered_derivative(
            lambda delta: (
                sigmoid(reward_gap)
                * cpo_loss(delta, reference.delta, beta, reference_margin)
                + (1.0 - sigmoid(reward_gap))
                * softplus(
                    beta * (delta - reference.delta) - reference_margin
                )
            ),
            approximate_cpo_delta,
        )
        rows.append(
            {
                "p_ref_w": p_ref,
                "reward_gap": reward_gap,
                "beta": beta,
                "gamma": gamma,
                "exact_constrained_rlhf": serialize_exact_result(exact),
                "equations_13_16_optimal_policy_margin": {
                    "defined": exact_margin is not None,
                    "margin": exact_margin,
                    "implicit_equation_abs_residual": exact_margin_residual,
                },
                "equation_17_reference_policy_approximation": {
                    "labeled_approximation": True,
                    "reference_margin": reference_margin,
                    "approximate_cpo_delta": approximate_cpo_delta,
                    "shift_abs_error": abs(
                        (approximate_cpo_delta - dpo_delta)
                        - reference_margin / beta
                    ),
                    "stationary_derivative_abs_error": abs(derivative),
                    "margin_abs_error_vs_exact": (
                        None
                        if exact_margin is None
                        else abs(reference_margin - exact_margin)
                    ),
                },
            }
        )
    return _summarize_cpo(rows)
```

`solve_exact_constrained_rlhf` evaluates the exact objective, not a CPO loss
with the reference margin substituted. For this declared single-pair
two-response reduction it must independently establish the `gamma == 0`
finite optimum and the `gamma > 0` boundary behavior from both analytic limits
and the fixed, representable delta tail sequence `(8.0, 16.0, 24.0, 32.0)`.
Interior first-order roots, if any, are recorded as stationary points and
never called global optima without the global certificate.

`_summarize_cpo` keeps the three stage names above and never fills undefined
exact fields with approximation values. It reports Equation 17 as an
approximation, the substituted reference margin as parameter-independent, and
approximation error only for rows with a certified exact optimal-policy
margin. The lane outcome is derived from all three stages; a passing
Equation 17 stationary-loss identity cannot override a failed or undefined
exact-optimum claim.

- [ ] **Step 4: Run GREEN and commit**

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_cpo_margin.py
git diff --check
git add src/conditional_dpo_repro/cpo.py tests/test_cpo_margin.py
git commit -m "feat: audit stationary CPO margin identities"
```

Expected: all 180 cases have explicit exact-objective classifications; the 45
`gamma == 0` rows recover the finite RLHF optimum, the 135 positive-`gamma`
rows retain their boundary certificates, Equations 13–16 are evaluated only
where an exact optimum exists, and every Equation 17 approximation row
satisfies its separately declared stationary-loss checks.

---

### Task 5: Soft-margin limit and negative-target lane

**Files:**

- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/src/conditional_dpo_repro/soft_margin.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/tests/test_soft_margin.py`

**Interfaces:**

- Consumes: stable DPO loss and the 150-case soft-margin grid.
- Produces: `run_soft_margin_lane() -> dict[str, object]`.

- [ ] **Step 1: Write failing soft-margin tests**

```python
def test_scaled_softplus_converges_to_hinge():
    result = run_soft_margin_lane()
    errors = result["max_abs_error_by_beta"]
    assert list(errors) == ["1", "4", "16", "64", "256"]
    assert errors["256"] < errors["64"] < errors["16"] < errors["4"]
    assert errors["256"] <= math.log(2.0) / 256.0 + 1e-12


def test_negative_margin_examples_include_wrong_preference():
    examples = run_soft_margin_lane()["negative_target_examples"]
    assert examples
    assert any(
        item["delta_ref"] < item["delta"] < 0.0
        and item["target_margin"] == item["delta_ref"]
        for item in examples
    )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_soft_margin.py
```

Expected: collection fails because `conditional_dpo_repro.soft_margin` does
not exist.

- [ ] **Step 3: Implement the finite-beta audit**

```python
SOFT_MARGIN_DELTAS_REF = (-3.0, -1.0, -0.25, 0.0, 0.50)
SOFT_MARGIN_DELTAS = (-3.5, -2.0, -0.50, 0.0, 0.75, 2.0)
SOFT_MARGIN_BETAS = (1.0, 4.0, 16.0, 64.0, 256.0)


def run_soft_margin_lane() -> dict[str, object]:
    rows = []
    for delta_ref, delta, beta in itertools.product(
        SOFT_MARGIN_DELTAS_REF, SOFT_MARGIN_DELTAS, SOFT_MARGIN_BETAS
    ):
        scaled = scaled_dpo_soft_margin(delta, delta_ref, beta)
        hinge = max(0.0, delta_ref - delta)
        rows.append(
            {
                "delta_ref": delta_ref,
                "delta": delta,
                "beta": beta,
                "scaled_dpo_loss": scaled,
                "hinge": hinge,
                "abs_error": abs(scaled - hinge),
            }
        )
    errors = {
        format(beta, "g"): max(r["abs_error"] for r in rows if r["beta"] == beta)
        for beta in SOFT_MARGIN_BETAS
    }
    examples = tuple(
        {
            "delta_ref": r["delta_ref"],
            "delta": r["delta"],
            "target_margin": r["delta_ref"],
        }
        for r in rows
        if r["beta"] == 256.0 and r["delta_ref"] < r["delta"] < 0.0
    )
    return {
        "case_count": len(rows),
        "max_abs_error_by_beta": errors,
        "negative_target_examples": examples,
        "finite_beta_loss_is_literal_hinge": False,
        "outcome": "consistent",
    }
```

- [ ] **Step 4: Run GREEN and commit**

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_soft_margin.py
git diff --check
git add src/conditional_dpo_repro/soft_margin.py tests/test_soft_margin.py
git commit -m "feat: reproduce DPO soft-margin limit"
```

Expected: all 150 cases remain finite through beta 256, errors contract toward
the hinge bound, and negative target examples are explicit.

---

### Task 6: Closed evidence schema, canonical builder, and atomic CLI

**Files:**

- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/schema/evidence-v1.schema.json`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/src/conditional_dpo_repro/evidence.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/src/conditional_dpo_repro/cli.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/tests/test_evidence.py`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/tests/test_cli.py`

**Interfaces:**

- Consumes: exact source record and all five lane functions.
- Produces: `build_evidence`, `validate_evidence`,
  `canonical_json_bytes`, `atomic_write`, and console script
  `conditional-dpo-repro`.

- [ ] **Step 1: Write failing evidence and CLI tests**

Repeat `EXPECTED_LIVE_CLAIMS` and `EXPECTED_LIVE_CLAIM_HASHES` from Task 1
verbatim at module scope in `test_evidence.py`; do not import them from
production code or construct one from the other.

```python
def test_bundle_has_all_live_claims_in_order(project_root):
    value = build_evidence(project_root)
    validate_evidence(value, project_root / "schema/evidence-v1.schema.json")
    assert value["paper_id"] == "7UEBX1KU1y"
    assert value["attempt_id"] == "933665ed-b7ed-4d73-9b07-35704660a184"
    assert len(value["claims"]) == 6
    assert [claim["targeted"] for claim in value["claims"]] == [
        True, True, True, True, True, False
    ]
    assert tuple(
        claim["challenge_claim"] for claim in value["claims"]
    ) == EXPECTED_LIVE_CLAIMS
    assert tuple(
        claim["challenge_claim_sha256"] for claim in value["claims"]
    ) == EXPECTED_LIVE_CLAIM_HASHES
    assert value["claims"][-1]["outcome"] == "not_reproduced"


def test_bundle_is_byte_deterministic(project_root):
    first = canonical_json_bytes(build_evidence(project_root))
    second = canonical_json_bytes(build_evidence(project_root))
    assert first == second
    assert first.endswith(b"\n")
    assert b"NaN" not in first and b"Infinity" not in first


def test_generate_failure_preserves_existing_output(project_root, tmp_path):
    output = tmp_path / "evidence.json"
    output.write_bytes(b"preserve\n")
    result = subprocess.run(
        [
            sys.executable, "-m", "conditional_dpo_repro.cli", "generate",
            "--project-root", str(project_root / "missing"), "--output", str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert output.read_bytes() == b"preserve\n"
    assert "Traceback" not in result.stderr
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_evidence.py tests/test_cli.py
```

Expected: collection fails because the evidence and CLI modules do not exist.

- [ ] **Step 3: Implement canonical evidence assembly**

```python
LANES = (
    run_equivalence_lane,
    run_relative_advantage_lane,
    run_undesirable_space_lane,
    run_cpo_margin_lane,
    run_soft_margin_lane,
)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def build_evidence(project_root: Path) -> dict[str, object]:
    source = load_source_record(project_root / "sources/paper.json")
    lane_results = tuple(lane() for lane in LANES)
    return {
        "schema_version": 1,
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "snapshot_id": SNAPSHOT_ID,
        "upstream_revision": UPSTREAM_REVISION,
        "source": source["paper"],
        "claims": _bind_results(source["claims"], lane_results),
        "limitations": (
            "No language model was trained or evaluated.",
            "The benchmark SOTA claim was not reproduced.",
            "The advertised author repository was unavailable during assessment.",
            "Only the challenge can issue official verdict labels.",
        ),
        "commands": (
            "conditional-dpo-repro generate --project-root . --output evidence.json",
            "conditional-dpo-repro validate --project-root . --evidence evidence.json",
        ),
    }
```

The schema validator is project-owned and recursive. It rejects unknown
top-level fields, a wrong schema version, wrong identity, missing claims,
wrong claim order/hash, invalid outcomes, booleans where numbers are expected,
and any non-finite float:

```python
ALLOWED_OUTCOMES = {
    "consistent", "contradiction", "mixed", "not_reproduced"
}


def _walk_finite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_finite(item, f"{path}[{index}]")
```

Store the corresponding closed structural contract in
`schema/evidence-v1.schema.json`, including `"additionalProperties": false`
for every project-owned object.

- [ ] **Step 4: Implement atomic CLI behavior**

```python
def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def command_generate(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    value = build_evidence(project_root)
    validate_evidence(value, project_root / "schema/evidence-v1.schema.json")
    atomic_write(args.output, canonical_json_bytes(value))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
```

`run-lane` uses a fixed dictionary from the five accepted names to lane
functions; it never imports or evaluates a caller-provided module name.

- [ ] **Step 5: Run GREEN and commit**

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_evidence.py tests/test_cli.py
git diff --check
git add schema/evidence-v1.schema.json src/conditional_dpo_repro/evidence.py src/conditional_dpo_repro/cli.py tests/test_evidence.py tests/test_cli.py
git commit -m "feat: build canonical conditional DPO evidence"
```

Expected: schema checks pass, two builds are byte-identical, and failed writes
preserve prior output.

---

### Task 7: Root evidence pages and reviewer documentation

**Files:**

- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/README.md`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/index.html`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/poster.html`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/tests/test_pages.py`

**Interfaces:**

- Consumes: committed root `evidence.json`.
- Produces: a static Space root and two offline reviewer pages.

- [ ] **Step 1: Write failing page tests**

```python
def test_readme_has_exact_static_space_metadata(project_root):
    readme = (project_root / "README.md").read_text("utf-8")
    assert "sdk: static" in readme
    assert "app_file: index.html" in readme
    assert "  - paper-7UEBX1KU1y" in readme
    assert "  - icml2026-repro" in readme


@pytest.mark.parametrize("page_name", ["index.html", "poster.html"])
def test_pages_use_only_committed_root_evidence(project_root, page_name):
    page = (project_root / page_name).read_text("utf-8")
    assert 'fetch("./evidence.json")' in page
    assert "https://" not in page
    assert "http://" not in page
    assert "innerHTML" not in page


def test_pages_state_honest_limits(project_root):
    text = (
        (project_root / "index.html").read_text("utf-8")
        + (project_root / "poster.html").read_text("utf-8")
        + (project_root / "README.md").read_text("utf-8")
    ).lower()
    assert "no language model was trained" in text
    assert "benchmark sota claim was not reproduced" in text
    assert "official verdict" in text
```

- [ ] **Step 2: Run page tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_pages.py
```

Expected: tests fail because the root pages and README do not exist.

- [ ] **Step 3: Create static metadata and safe renderers**

Start `README.md` with the exact YAML from the approved design. Document:

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv sync --frozen
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run conditional-dpo-repro generate --project-root . --output evidence.json
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run conditional-dpo-repro validate --project-root . --evidence evidence.json
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q
```

Both pages use DOM text nodes:

```javascript
function cell(text) {
  const element = document.createElement("td");
  element.textContent = String(text);
  return element;
}

async function loadEvidence() {
  const response = await fetch("./evidence.json");
  if (!response.ok) throw new Error(`evidence fetch failed: ${response.status}`);
  const evidence = await response.json();
  const body = document.querySelector("#claims");
  for (const claim of evidence.claims) {
    const row = document.createElement("tr");
    row.append(
      cell(claim.challenge_claim),
      cell(claim.outcome),
      cell(claim.summary),
      cell((claim.limitations || []).join("; "))
    );
    body.append(row);
  }
}

loadEvidence().catch((error) => {
  document.querySelector("#error").textContent = error.message;
});
```

`index.html` shows the six-claim table, identity, source pin, lane parameters,
and limitations. `poster.html` shows one compact card per targeted lane plus
the unavailable benchmark claim. Each page links to the other with a relative
URL.

- [ ] **Step 4: Run GREEN and commit**

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_pages.py
git diff --check
git add README.md index.html poster.html tests/test_pages.py
git commit -m "docs: add conditional DPO evidence pages"
```

Expected: metadata and page tests pass without external URLs or unsafe HTML
injection.

---

### Task 8: Generate the evidence bundle and perform final proposal validation

**Files:**

- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/evidence.json`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/evidence/commands.json`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/evidence/environment.json`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/evidence/validation.json`
- Create:
  `submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/tests/test_committed_bundle.py`

**Interfaces:**

- Consumes: Tasks 1–7 and the exact locked environment.
- Produces: committed evidence bytes and non-authoritative validation metadata.

- [ ] **Step 1: Write the failing committed-bundle test**

```python
def test_committed_bundle_matches_fresh_build(project_root):
    committed = (project_root / "evidence.json").read_bytes()
    fresh = canonical_json_bytes(build_evidence(project_root))
    assert committed == fresh
    validate_evidence(
        json.loads(committed),
        project_root / "schema/evidence-v1.schema.json",
    )


def test_validation_binds_evidence_hash(project_root):
    payload = (project_root / "evidence.json").read_bytes()
    validation = json.loads(
        (project_root / "evidence/validation.json").read_text("utf-8")
    )
    assert validation["evidence_sha256"] == hashlib.sha256(payload).hexdigest()
    assert validation["schema_valid"] is True
    assert validation["deterministic"] is True
```

- [ ] **Step 2: Run the bundle test and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q tests/test_committed_bundle.py
```

Expected: failure because the committed bundle does not exist.

- [ ] **Step 3: Generate twice and record exact metadata**

Run from the project directory:

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv sync --frozen
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run conditional-dpo-repro generate --project-root . --output /tmp/conditional-dpo-evidence-first.json
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run conditional-dpo-repro generate --project-root . --output /tmp/conditional-dpo-evidence-second.json
cmp /tmp/conditional-dpo-evidence-first.json /tmp/conditional-dpo-evidence-second.json
cp /tmp/conditional-dpo-evidence-first.json evidence.json
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run conditional-dpo-repro validate --project-root . --evidence evidence.json
sha256sum evidence.json uv.lock
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run python -c 'import json,platform,sys; print(json.dumps({"python": platform.python_version(), "implementation": platform.python_implementation(), "platform": sys.platform}, sort_keys=True))'
```

Write the literal ordered commands and expected zero exit statuses to
`evidence/commands.json`. Write only the three printed runtime fields and the
`uv.lock` SHA-256 to `evidence/environment.json`. Write the evidence SHA-256,
`schema_valid: true`, `deterministic: true`, and focused/full test counts to
`evidence/validation.json`. Do not include environment variables, absolute
cache paths, credentials, or timestamps.

- [ ] **Step 4: Run project and workspace verification**

Run:

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run pytest -q
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run python -c 'import json; json.load(open("evidence.json", encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))'
git diff --check
cd /home/will/projects/icml-2026-reproductions
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q --ignore=submissions/nape
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache PRE_COMMIT_HOME=/tmp/icml-repro-pre-commit uv run pre-commit run -a
```

Expected: all project tests pass; strict JSON parsing succeeds; no whitespace
errors exist; the root suite excluding archival NAPE passes; and every
pre-commit hook passes. If unrelated concurrent workspace changes prevent the
root checks from passing, preserve their output as a concern and do not edit
those files.

- [ ] **Step 5: Commit only the paper project**

```bash
cd submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment
git add evidence.json evidence/commands.json evidence/environment.json evidence/validation.json tests/test_committed_bundle.py
git commit -m "evidence: publish finite conditional DPO results"
```

Expected: the commit contains only this paper project. Report the commit,
commands, evidence paths, exact evidence SHA-256, and concerns as a worker
proposal. Request controller validation; do not deploy, submit, mutate state,
or claim an official verdict.
