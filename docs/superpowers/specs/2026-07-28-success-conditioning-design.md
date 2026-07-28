# Success Conditioning Formal-Evidence Reproduction Design

## Purpose and authority

This design covers attempt `b2b5899c-43a1-4c91-8b1f-9122d746f4c6`,
challenge paper `FEmXFeqYNZ`, “Success-Conditioning as Policy Improvement:
The Optimization Problem Solved by Imitating Success.” The admitted snapshot
is `09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199`.
The worker produces a paper-scoped proposal only. The controller retains
design approval, validation, integration, deployment, submission, polling,
and verdict authority.

The project will independently recompute the paper's four admitted claims on
finite episodic MDPs. It will not treat the paper's proof, examples, figures,
or printed values as reproduced measurements. The paper is the specification
being tested; separate implementations, constrained solvers, exhaustive
enumeration, exact arithmetic, mutation controls, and independently generated
fixtures supply the evidence.

## Immutable claims and source

The target claims, in order, are:

1. `Success conditioning is proved to exactly solve a trust-region policy optimization problem with a chi-squared divergence constraint whose radius is determined by the data (Section 4.3).`
2. `At the optimum, relative policy improvement, chi-squared policy-change magnitude, and action-influence are shown to be exactly equal at every state (Section 4.4).`
3. `The analysis characterizes exact success conditioning as a conservative improvement operator that cannot degrade performance or induce dangerous distribution shift (Section 4.3).`
4. `The paper analyzes return thresholding as an extension that can amplify improvement but can misalign the learned policy with the true objective (Section 5).`

These strings, including their parenthetical locators, are immutable challenge
bindings and must not be corrected in place. The fourth binding contains a
source-locator mismatch: the challenge says Section 5, while pinned
`arxiv:2601.18175v2` places return thresholding in Section 7, states the
amplification/alignment identity in Proposition 7.1, and gives the Beta example
in Section 7.3. Evidence therefore records the challenge locator and pinned-v2
locator separately as `challenge_locator`, `source_locator`,
`identity_locator`, `example_locator`, and `locator_status`. If the substantive
thresholding observations are `supported`, the known locator mismatch combines
to local status `mixed`, not unqualified `supported`. A substantive
`not_supported` result remains `not_supported`; a substantive `inconclusive`
result remains `inconclusive`. The combination is derived from serialized
component outcomes and is never hard-coded by the builder or renderer. Every
rendered page must display both components and this limitation.

The sole upstream input is `arxiv:2601.18175v2`, licensed CC BY 4.0:

- URL: `https://arxiv.org/pdf/2601.18175v2`
- acquisition command: `curl -fsSL https://arxiv.org/pdf/2601.18175v2 -o "$PINNED_INPUT_DIR/2601.18175v2.pdf"`
- byte count: `652921`
- SHA-256: `f9194e48cadf5c13307eb6a523ed20e4fb787856b1fc8b19e4f628a8ac3ad672`

The downloaded PDF is an external validation input and is never committed.
The project commits reviewed UTF-8 transcriptions of only the definitions and
equations needed to state the tested properties. Each transcription records
page, section, exact excerpt bytes, and SHA-256. Transcriptions explain test
targets; no test passes merely because code matches transcription text.

## Considered approaches

Three approaches were considered:

1. A proof-text audit would be quick, but circular: it could only show that
   the paper states its own conclusions.
2. A single Monte Carlo toy MDP would be executable, but sampling noise and
   one small fixture would likely justify only `toy` evidence.
3. The selected approach combines a universal machine-checkable symbolic proof
   certificate, exact rational MDP semantics, exhaustive trajectory
   conditioning, an independently parameterized numerical optimizer,
   randomized and adversarial fixture families, and closed-form Beta survival
   probabilities. It remains CPU-only while providing independent paths for
   all four claims.

## Six-lane architecture

### Lane 1: immutable provenance and theorem contracts

`provenance.py` verifies the pinned PDF and the transcription manifest.
`claims.py` fixes the four claim strings and hashes. `schema.json` fixes the
canonical evidence contract. Any source, claim-order, excerpt, or schema drift
fails closed before computation.

### Lane 2: finite-MDP truth engine

`mdp.py` represents finite rational MDPs with explicit success and failure
terminal states. It supports:

- layered acyclic MDPs by backward dynamic programming;
- transient absorbing MDPs by exact `Fraction` Gaussian elimination;
- behavior-policy value, Q-value, advantage, occupancy, and success
  probability;
- Bayes success conditioning,
  `pi_plus(a|s) = pi0(a|s) * Q0(s,a) / V0(s)`; and
- exhaustive trajectory conditioning on bounded acyclic controls.

The dynamic-programming and trajectory-enumeration paths share data types but
not policy-conditioning logic. Exact rational comparisons decide theorem
truth. Floating point is used only by the independent optimizer and plotting.

### Lane 3: universal symbolic proof certificate

`proofs.py` emits and independently verifies a closed certificate for claims
1–3 over arbitrary finite states and each state's behavior-supported action
set, rather than over the bounded fixture collection. Its assumptions are
explicit symbolic premises: nonnegative occupancy weights `d_s`,
`p_{s,i} > 0` on active support, `sum_i p_{s,i} = 1`,
`0 <= q_{s,i} <= 1`, and
`v_s = sum_i p_{s,i} q_{s,i} > 0`. It derives
`pi_plus_{s,i} = p_{s,i} q_{s,i} / v_s` and checks:

- normalization, support inclusion, and the exact centered-change form;
- the aggregate state/action trust-region objective upper bound by weighted
  Cauchy–Schwarz, equality at `pi_plus`, and the corresponding primal
  feasibility, dual feasibility, stationarity, and complementary-slackness KKT
  equations;
- the universal identity
  `relative_improvement = chi2(pi_plus || p) = Var_p(q) / v^2`,
  with action influence reduced independently to the same normal form; and
- nonnegative one-step advantage as a sum of weighted squares, followed by the
  finite-MDP policy-improvement implication and the certified support/movement
  bound used by claim 3.

The certificate uses formal indexed sums and exact coefficient
canonicalization; it contains no fixture values, random seeds, SLSQP outputs,
or tolerances. Its verifier rejects a missing premise, altered exponent,
reversed divergence, incorrect radius, or any unproved inequality step. The
paper transcriptions identify the theorem being checked but are not inputs to
the algebraic reductions. The bounded exact enumeration and numerical solver
below are independent corroboration, not the universal proof.

### Lane 4: numerical trust-region corroboration

`trust_region.py` constructs the Section 4.3 objective and aggregate
chi-squared constraint from independently evaluated `pi0`, `Q0`, `V0`,
unconditioned occupancy, and success-conditioned occupancy. A SciPy SLSQP
solver sees neither the Bayes update nor `pi_plus`; it optimizes one simplex
per state from multiple deterministic starts. Evidence records objective gap,
constraint slack, policy distance for nondegenerate cases, solver success,
and KKT residuals. Small bandits additionally receive exhaustive rational-grid
upper-bound checks. Mutation controls alter the radius, occupancy weights, or
divergence direction and must be detected.

### Lane 5: statewise identities and conservative improvement

`identities.py` independently computes, at each reachable state:

- relative one-step advantage;
- `chi2(pi_plus || pi0)`; and
- action influence, from the variance of `Q0` under `pi0`.

All three must agree exactly as `Fraction`. `conservative.py` then evaluates
`pi_plus` afresh and checks statewise and initial-distribution monotonicity,
the paper's lower bound, aggregate movement/radius equality, and lack of
out-of-support mass. “No dangerous distribution shift” is reported only as
the proved finite-MDP chi-squared support/movement bound; the reproduction
must not turn it into a real-world safety guarantee. Reversed-success and
out-of-support mutations must fail.

### Lane 6: thresholding, evidence, and root pages

`thresholding.py` uses `scipy.special.betainc` to compute Beta tail
probabilities without Monte Carlo. It evaluates the paper's 100-arm family
over 64 fixed, recorded seeds and a fixed threshold grid. It compares faithful
Bernoulli reward conditioning with threshold proxy conditioning, verifies the
Proposition 7.1 amplification/alignment identity, and records both a moderate
threshold that improves beyond faithful conditioning and a severe threshold
that harms the true objective. It does not compare against pixels read from
Figure 2. Its evidence record explicitly says that these items occur in
Section 7 / Proposition 7.1 / Section 7.3 of pinned v2, not Section 5 as the
immutable challenge binding states.

`evidence.py` orchestrates all lanes, rejects malformed, incomplete, or
integrity-inconsistent records, and writes canonical JSON plus per-case JSONL.
Scientifically null, contradictory, failed-solver, and counterexample outcomes
are valid evidence records and are never discarded merely because they do not
support a claim. Root
`README.md`, `report.md`, `poster.html`, `poster_embed.html`, and `app.py`
render only schema-valid evidence. They never embed manually copied
measurements.

## Exact project boundary and file map

The project root is:

`submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success/`

It contains:

- `pyproject.toml`, `uv.lock`, `LICENSE`, `LICENSES/CC-BY-4.0.txt`, and
  `NOTICE.md`;
- `paper_transcriptions/manifest.json` and `paper_transcriptions/excerpts/`;
- `src/success_conditioning_repro/{claims,provenance,types,linear,mdp,fixtures,proofs,trust_region,identities,conservative,thresholding,evidence,render,cli}.py`;
- `evidence/schema.json`, generated `evidence/evidence.json`,
  `evidence/cases.jsonl`, `evidence/provenance.json`,
  `evidence/manifest.json`, `evidence/tdd-log.jsonl`, and
  `evidence/repro-bundle.tar.gz`;
- root pages `README.md`, `report.md`, `poster.html`, `poster_embed.html`, and
  `app.py`; and
- focused tests under `tests/`.

Original code and schema are MIT. Source excerpts and attribution material are
CC BY 4.0. `NOTICE.md` records the boundary.

## Pinned interfaces

```python
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

Probability = Fraction
State = str
Action = str
Policy = Mapping[State, Mapping[Action, Probability]]

@dataclass(frozen=True)
class FiniteMDP:
    states: tuple[State, ...]
    actions: Mapping[State, tuple[Action, ...]]
    transitions: Mapping[tuple[State, Action], tuple[tuple[State, Probability], ...]]
    initial: Mapping[State, Probability]
    success_states: frozenset[State]
    failure_states: frozenset[State]

    @property
    def nonterminal_states(self) -> tuple[State, ...]:
        return tuple(
            state
            for state in self.states
            if state not in self.success_states and state not in self.failure_states
        )

@dataclass(frozen=True)
class Evaluation:
    rho: Fraction
    value: Mapping[State, Fraction]
    q_value: Mapping[tuple[State, Action], Fraction]
    occupancy: Mapping[State, Fraction]

@dataclass(frozen=True)
class TrustRegionProblem:
    states: tuple[State, ...]
    actions: Mapping[State, tuple[Action, ...]]
    behavior: Policy
    objective_coefficients: Mapping[tuple[State, Action], Fraction]
    movement_weights: Mapping[State, Fraction]
    radius: Fraction

@dataclass(frozen=True)
class SolverResult:
    success: bool
    policy: Policy
    objective: float
    constraint_violation: float
    simplex_residual: float
    stationarity_residual: float
    starts: tuple[Mapping[str, object], ...]

def validate_mdp(mdp: FiniteMDP, policy: Policy) -> None: ...
def evaluate_policy(mdp: FiniteMDP, policy: Policy) -> Evaluation: ...
def condition_on_success(mdp: FiniteMDP, policy: Policy, evaluation: Evaluation) -> Policy: ...
def enumerate_conditioned_policy(mdp: FiniteMDP, policy: Policy) -> Policy: ...
def success_conditioned_occupancy(evaluation: Evaluation) -> Mapping[State, Fraction]: ...
def build_universal_certificate() -> "UniversalCertificate": ...
def verify_universal_certificate(certificate: "UniversalCertificate") -> None: ...
def solve_trust_region(problem: "TrustRegionProblem", starts: int = 8) -> "SolverResult": ...
def relative_objective_gap(problem: TrustRegionProblem, candidate: Policy, optimum: Policy) -> float: ...
def statewise_identity(mdp: FiniteMDP, pi0: Policy, pi_plus: Policy) -> tuple["IdentityRow", ...]: ...
def conservative_checks(mdp: FiniteMDP, pi0: Policy, pi_plus: Policy) -> "ConservativeResult": ...
def beta_threshold_sweep(seed: int, thresholds: Sequence[float]) -> "ThresholdResult": ...
def build_evidence(output_dir: Path, pinned_pdf: Path) -> dict[str, object]: ...
def validate_evidence(evidence_path: Path, schema_path: Path, evidence_root: Path) -> None: ...
```

Validation must open and hash file-backed provenance, cases, and manifests.
Validating only an in-memory dictionary is forbidden.

## Deterministic fixture domains

The canonical suite uses:

- the paper's two-arm values as a transcription smoke test, never as the sole
  evidence;
- at least 256 independently generated full-support layered rational MDPs;
- at least 64 transient absorbing rational MDPs;
- at least 32 bounded acyclic MDPs with exhaustive trajectory conditioning;
- at least 64 independently solved trust-region instances with eight starts;
- exhaustive rational-grid checks for at least 128 two- and three-arm
  bandits;
- rare-action probabilities down to `1/1_000_000`, near-zero influence,
  deterministic-transition, stochastic-transition, and repeated-state cases;
  and
- 64 Beta-bandit seeds, 100 arms per seed, and every threshold `i / 1000`
  for integer `i` from 1 through 999.

Every seed, full instance tensor, solver start, tolerance, and case result is
serialized. Generation refuses reduced domains. Evidence labels each domain
as exact exhaustive, exact finite, or numerical non-exhaustive.

## Acceptance rules and error handling

- Claims 1–3 are `supported` only when the universal symbolic certificate is
  well formed and validates and the complete bounded corroboration domains
  contain zero exact-arithmetic violations. A valid exact counterexample yields
  `not_supported`. Missing numerical convergence without an exact
  counterexample, including complete SLSQP failure, yields `inconclusive` and
  preserves every failed start.
- Claim 1's numerical corroboration records feasibility against `1e-9` and
  relative objective gap against `1e-8`; violations affect the derived status
  but do not invalidate an otherwise well-formed evidence bundle.
- Claim 2 derives its result from the statewise equality observations on every
  reachable positive-value state. Claim 3 derives its result from value and
  success-rate monotonicity, aggregate movement/radius equality, support, and
  negative-control observations. Any valid counterexample is serialized.
- Claim 4's substantive component is `supported` only when the analytic proxy
  identity is within `1e-10` and both amplification and harmful-misalignment
  regimes occur in at least 60 of 64 seeds. A complete null/counterexample
  domain yields substantive `not_supported`; insufficient valid observations
  yield substantive `inconclusive`. The aggregate status is computed from that
  component and the pinned-v2 `mismatch` locator using the rule above.
- Invalid probability simplices, malformed or nonabsorbing declared inputs,
  singular required exact systems, missing canonical cases, nonfinite values,
  schema violations, manifest/hash mismatches, or a malformed/tampered proof
  certificate abort evidence generation as integrity failures. Scientific
  violations, null effects, counterexamples, and numerical solver failures do
  not abort generation.
- Local evidence statuses are `supported`, `mixed`, `not_supported`, or
  `inconclusive`; they are derived from observations and are never called
  official verdicts.

## Testing, commands, and reproducibility

Every production change follows red-green-refactor. Before implementation,
the worker runs the focused missing-behavior test and appends its command,
timestamp, test node, expected failure, and observed failure to
`evidence/tdd-log.jsonl`.

Canonical commands are:

```bash
PROJECT=submissions/success-conditioning-as-policy-improvement-the-optimization-problem-solved-by-imitating-success
PINNED_INPUT_DIR=/tmp/success-conditioning-inputs
uv sync --project "$PROJECT" --frozen
uv run --project "$PROJECT" python -m pytest -q
uv run --project "$PROJECT" success-conditioning-repro recompute \
  --pinned-pdf "$PINNED_INPUT_DIR/2601.18175v2.pdf" \
  --output-dir "$PROJECT/evidence"
uv run --project "$PROJECT" success-conditioning-repro validate \
  --evidence "$PROJECT/evidence/evidence.json" \
  --schema "$PROJECT/evidence/schema.json"
uv run --project "$PROJECT" success-conditioning-repro render \
  --evidence "$PROJECT/evidence/evidence.json" \
  --project-root "$PROJECT"
uv run pre-commit run -a
```

Two clean recomputations must produce byte-identical canonical artifacts.
Canonical files omit wall-clock time, absolute paths, host names, and
timestamps. Runtime observations belong only in controller validation.

## Scope exclusions

There is no GPU work, paid API, model training, large-scale language-model
experiment, copied contributor artifact, or autonomous external mutation.
The worker does not edit state, skills, controller documents, another
submission, or Hub resources. The Space is a CPU proposal exposing generated
evidence and bounded recomputation; only the controller may publish it.
