# Conditional DPO/RLHF Reproduction Design

**Paper:** Conditional Equivalence of DPO and RLHF: Assumptions, Failure
Modes, and Provable Alignment

**Paper ID:** `7UEBX1KU1y`

**Attempt ID:** `933665ed-b7ed-4d73-9b07-35704660a184`

**Pinned paper:** `arxiv:2605.20834v1`

**Admitted snapshot:**
`09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199`

**Design date:** 2026-07-28

**Approval:** The controller explicitly authorized the approved five-lane
design: finite response grids, equivalence and counterexample checks,
CPO-margin identities, test-driven development, root presentation pages, a
pinned arXiv source, and a deterministic evidence bundle.

**Phase gate:** The attempt is `selected`. This document is a worker-authored
proposal. It does not record or approve a design in coordinator state, grant a
writer lease, authorize implementation, attest evidence, deploy a Space,
submit an entry, or import a verdict.

---

## 1. Objective and scope

Build a small, independently executable CPU reproduction of the paper's five
selected mathematical claims. The implementation will enumerate finite
two-response policies and evaluate the paper's equations directly. It will
not train or load a language model.

The five evidence lanes are:

1. conditional DPO/RLHF equivalence;
2. relative advantage versus absolute preference;
3. undesirable-space counterexamples;
4. constrained-RLHF and stationary CPO margin identities; and
5. DPO as soft-margin ranking with possibly negative targets.

The sixth live challenge claim, state-of-the-art benchmark performance, is
outside scope. The evidence bundle and both presentation pages will retain it
in live-claim order as `not_reproduced`, with the explicit reason that
reproducing it requires model training, inference, external evaluators, and
unavailable upstream code.

All evidence computation is:

- CPU-only;
- deterministic;
- offline after project installation;
- standard-library-only at runtime;
- free of paid APIs;
- bounded to finite grids that complete in under 30 minutes; and
- independent of the paper's reported experimental values.

The advertised repository, `https://github.com/visitworld123/CPO`, returned
HTTP 404 during assessment. It is not a required input. The reproduction must
not fabricate a repository revision, reconstruct the authors' training code,
or treat missing code as evidence for or against a mathematical claim.

## 2. Approaches considered

### Selected: independent finite-state equation auditor

Represent one prompt with two responses, preferred `w` and dispreferred `l`.
The policy is fully determined by `p_w in (0, 1)`, with
`p_l = 1 - p_w`, and
`delta = log(p_w) - log(p_l) = logit(p_w)`. Evaluate the RLHF, DPO, CPO, and
ranking equations over exact, declared grids and analytic identities.

This approach is selected because it isolates the claims' mathematical
content, makes counterexamples inspectable, and requires neither author code
nor model weights.

### Rejected: reproduce the paper's Llama-3 experiments

This would require GPU training or inference, judge-backed model evaluation,
datasets, and the unavailable repository. It violates the CPU-only scope and
cannot complete within the attempt's bounded budget.

### Rejected: prose-only proof audit

A source audit could identify equations and internal inconsistencies but would
not independently recompute observations. It is useful as context, but is not
sufficient evidence by itself.

## 3. Critical semantic distinction

The reproduction must not assume that all uses of “DPO objective” are
equivalent. It will calculate three separate objects:

1. **Finite RLHF optimum.** For finite reward difference `reward_gap`, full
   support reference policy, and `beta > 0`,

   ```text
   delta_rlhf = delta_ref + reward_gap / beta
   ```

2. **One-sided empirical DPO loss from paper Equation 8.**

   ```text
   L_positive(delta) =
       softplus(-beta * (delta - delta_ref))
   ```

   This loss is strictly decreasing in `delta` and has no finite minimizer on
   the unconstrained real line. Its infimum occurs as `delta -> +infinity`.

3. **Full Bradley–Terry population cross-entropy.** If the preferred outcome
   occurs with probability `q = sigmoid(reward_gap)`, then

   ```text
   L_population(delta) =
       -q * log(sigmoid(beta * (delta - delta_ref)))
       -(1-q) * log(1-sigmoid(beta * (delta - delta_ref)))
   ```

   Its finite stationary point is
   `delta_ref + logit(q) / beta = delta_rlhf`.

The evidence will report these outcomes separately. In particular, a
positive `delta_rlhf` does not turn the one-sided Equation 8 loss into a
finite-optimum objective. Conversely, the population cross-entropy identity
holds algebraically regardless of the sign of `delta_rlhf`. This is a
reproduction result to compute and expose, not a predetermined verdict.

## 4. Immutable identity and claim bindings

The project path is:

```text
submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/
```

The evidence must bind all six live claims to these immutable, ordered
constants from the admitted attempt and snapshot. The first five are targeted;
the sixth is retained as `not_reproduced`.

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

Tests must embed the same six strings and hashes as immutable expected
constants. A test that merely hashes strings loaded from `sources/paper.json`
and compares them with hashes from that same file is self-consistent but does
not protect the admitted binding; it is insufficient.

`sources/paper.json` will record:

- arXiv identifier and immutable version `2605.20834v1`;
- abstract URL `https://arxiv.org/abs/2605.20834v1`;
- HTML URL `https://arxiv.org/html/2605.20834v1`;
- source URL `https://arxiv.org/e-print/2605.20834v1`;
- upstream revision token `arxiv:2605.20834v1`;
- the arXiv perpetual non-exclusive license label exposed by the paper page;
- the unavailable advertised repository URL and observed access limitation;
- the exact six live challenge strings and hashes; and
- equation identifiers used by each lane.

The paper itself will not be vendored because its arXiv license is not a
general public redistribution license. Versioned URLs and equation
transcriptions are provenance, while recomputed numbers are evidence.

## 5. Project structure and interfaces

```text
pyproject.toml
uv.lock
README.md
index.html
poster.html
evidence.json
sources/
  paper.json
schema/
  evidence-v1.schema.json
src/conditional_dpo_repro/
  __init__.py
  math.py
  grids.py
  claims.py
  evidence.py
  cli.py
tests/
  conftest.py
  test_identity.py
  test_math.py
  test_equivalence.py
  test_relative_advantage.py
  test_undesirable_space.py
  test_cpo_margin.py
  test_soft_margin.py
  test_evidence.py
  test_cli.py
  test_pages.py
evidence/
  commands.json
  environment.json
  validation.json
```

The public Python interfaces are:

```python
@dataclass(frozen=True)
class TwoResponsePolicy:
    preferred: float
    dispreferred: float

    @property
    def delta(self) -> float: ...

def policy_from_delta(delta: float) -> TwoResponsePolicy: ...
def rlhf_optimal_delta(delta_ref: float, reward_gap: float, beta: float) -> float: ...
def dpo_loss(delta: float, delta_ref: float, beta: float) -> float: ...
def dpo_loss_derivative(delta: float, delta_ref: float, beta: float) -> float: ...
def bt_population_loss(
    delta: float, delta_ref: float, reward_gap: float, beta: float
) -> float: ...
def constrained_rlhf_objective(
    policy: TwoResponsePolicy,
    reference: TwoResponsePolicy,
    reward_gap: float,
    beta: float,
    gamma: float,
) -> float: ...
def solve_exact_constrained_rlhf(
    reference: TwoResponsePolicy,
    reward_gap: float,
    beta: float,
    gamma: float,
) -> dict[str, object]: ...
def cpo_optimal_policy_margin(policy: TwoResponsePolicy, gamma: float) -> float: ...
def cpo_reference_margin(reference: TwoResponsePolicy, gamma: float) -> float: ...
def cpo_loss(
    delta: float, delta_ref: float, beta: float, margin: float
) -> float: ...
def cpo_loss_derivative(
    delta: float, delta_ref: float, beta: float, margin: float
) -> float: ...
def scaled_dpo_soft_margin(
    delta: float, delta_ref: float, beta: float
) -> float: ...

def run_equivalence_lane() -> dict[str, object]: ...
def run_relative_advantage_lane() -> dict[str, object]: ...
def run_undesirable_space_lane() -> dict[str, object]: ...
def run_cpo_margin_lane() -> dict[str, object]: ...
def run_soft_margin_lane() -> dict[str, object]: ...
def build_evidence(project_root: Path) -> dict[str, object]: ...
def validate_evidence(value: object, schema_path: Path) -> None: ...
def canonical_json_bytes(value: object) -> bytes: ...
```

Inputs must reject non-finite values, probabilities outside `(0, 1)`,
probabilities that do not sum to one within `1e-12`, `beta <= 0`, and
`gamma < 0`. Numerical comparisons use explicit absolute tolerances recorded
with each observation; no test may rely on platform-default approximate
comparison.

## 6. Canonical finite grids

The complete evidence run uses declared Cartesian products rather than random
sampling.

### Equivalence grid

```text
p_ref_w    = [0.05, 0.10, 0.25, 0.40, 0.50, 0.75, 0.90]
reward_gap = [0.10, 0.50, 1.00, 2.00]
beta       = [0.25, 0.50, 1.00, 2.00]
```

For each of the 112 cases, compute `delta_ref`, `delta_rlhf`, the sign
condition, the analytic population-loss stationary point, a centered
finite-difference derivative at that point, and the one-sided DPO derivative.
Record counts by positive, zero-within-tolerance, and negative
`delta_rlhf`.

### Relative-advantage and undesirable-space grid

```text
delta_ref = [-4.0, -2.0, -1.0, -0.25, 0.25]
offset    = [0.10, 0.25, 0.50, 1.00, 2.00]
beta      = [0.50, 1.00, 2.00]
```

Set `delta = delta_ref + offset`. The relative criterion is `offset > 0`;
the absolute criterion is `delta > 0`. A witness belongs to the undesirable
space exactly when `delta_ref < delta < 0`. For every witness, compare the DPO
loss at `delta` with the loss at `delta_ref`, and record the corresponding
preferred-response probabilities.

### CPO grid

```text
p_ref_w    = [0.10, 0.25, 0.50, 0.75, 0.90]
reward_gap = [0.10, 0.50, 1.00]
beta       = [0.50, 1.00, 2.00]
gamma      = [0.00, 0.01, 0.05, 0.10]
```

For each case, lane 4 is a three-stage audit whose stages may not be collapsed:

1. Solve or classify the exact finite two-response constrained-RLHF objective.
   Report whether a finite global optimum exists, its policy and objective
   value when it does, and independent first-order, curvature, and boundary
   certificates. If the objective is boundary-seeking or unbounded for a
   parameter choice, record that result rather than manufacturing an
   `optimal_policy`.
2. Only for a certified finite optimum, evaluate the Equations 13–16
   optimal-policy margin
   `gamma * (1/p_star_w + 1/p_star_l)` and its implicit optimality residual.
   This is the exact-result record and must use `p_star`, never `p_ref`.
3. Separately audit the Equation 17 reference-policy approximation
   `gamma * (1/p_ref_w + 1/p_ref_l)`, its stationary CPO logit, predicted
   shift `reference_margin / beta`, and held-fixed loss derivative. The output
   must call this an approximation and report its error against the exact
   optimal-policy margin only when the latter exists.

Numerically perturb the policy delta while holding the precomputed reference
margin fixed to verify that the approximation contributes zero parameter
derivative. The exact constrained-RLHF result, Equations 13–16
optimal-policy-margin result, and Equation 17 reference-policy approximation
must occupy separate named objects in every row and in the aggregate summary.

### Soft-margin grid

```text
delta_ref = [-3.0, -1.0, -0.25, 0.00, 0.50]
delta     = [-3.5, -2.0, -0.50, 0.00, 0.75, 2.00]
beta      = [1.0, 4.0, 16.0, 64.0, 256.0]
```

Compare `dpo_loss / beta` with
`max(0, delta_ref - delta)`. Record the maximum absolute error at each beta
and demonstrate convergence without asserting that finite-beta DPO loss is
literally a hinge loss or reaches zero. Negative target examples require
`delta_ref < 0`; at least one canonical example must also have
`delta_ref < delta < 0`.

## 7. Lane outcomes

Each lane returns observations and one of these local evidence outcomes:

- `consistent`: every declared identity and invariant passed;
- `contradiction`: a recomputed observation conflicts with the literal target
  claim;
- `mixed`: separable parts support and conflict with the claim;
- `not_reproduced`: the lane was intentionally not executed.

These are local evidence labels, not official challenge verdicts. The
implementation must never emit `verified`, `falsified`, `toy`, or
`inconclusive` as if the challenge had judged the submission.

The lane outcome is derived from observations, never hard-coded to a desired
answer. In particular:

- Lane 1 must expose whether the one-sided DPO objective has a finite optimum
  and whether the population likelihood identity depends on the sign
  condition.
- Lane 2 must separately count relative improvement and absolute preference.
- Lane 3 must list concrete policy witnesses, not only count them.
- Lane 4 must first solve or classify the exact constrained-RLHF problem and,
  where a finite optimum exists, test the Equations 13–16 optimal-policy
  margin. It must then audit Equation 17's reference substitution as a
  separately named approximation. A passing stationary-loss identity for the
  substituted margin cannot stand in for either preceding exact check.
- Lane 5 must test the scaled high-beta limit and retain finite-beta errors.

## 8. Evidence bundle

`evidence.json` is canonical UTF-8 JSON with sorted keys, two-space
indentation, a trailing newline, and no NaN or infinity. It contains:

- schema version;
- paper, attempt, snapshot, and upstream identity;
- all six live claims in exact order;
- target and challenge hashes;
- declared parameters and tolerances;
- per-lane observations and derived outcomes;
- explicit paper-context equations;
- limitations and unavailable artifacts; and
- the exact commands needed to regenerate and validate it.

It excludes timestamps, random identifiers, host paths, environment dumps,
Git credentials, network responses, and paper-reported benchmark values.

The companion bundle records:

- `evidence/commands.json`: ordered commands and expected exit status;
- `evidence/environment.json`: Python version, implementation, platform, and
  `uv.lock` SHA-256 from the validation run, with no environment variables;
- `evidence/validation.json`: evidence SHA-256, schema result, test counts, and
  determinism comparison.

The CLI is:

```text
conditional-dpo-repro generate --project-root PATH --output PATH
conditional-dpo-repro validate --project-root PATH --evidence PATH
conditional-dpo-repro run-lane --project-root PATH --lane {equivalence,relative,undesirable,cpo,soft-margin}
```

Writes use a same-directory temporary file followed by `os.replace`.
Validation and lane execution are read-only. Failures return nonzero, preserve
an existing output, print one concise error to stderr, and do not expose a
traceback by default.

## 9. Root presentation pages

The deployable Space source is the project root. `README.md` begins with:

```yaml
---
title: Conditional DPO/RLHF Equation Reproduction
emoji: ⚖️
colorFrom: indigo
colorTo: amber
sdk: static
app_file: index.html
pinned: false
tags:
  - paper-7UEBX1KU1y
  - icml2026-repro
---
```

`index.html` is the reviewer-facing evidence explorer. `poster.html` is a
single-screen summary. Both fetch only `./evidence.json`, render all six live
claims in order, display local evidence outcomes and computed observations,
and link to each other. They use no external scripts, fonts, images, analytics,
or network services.

The pages must state prominently:

- no language model was trained or evaluated;
- no paper benchmark was reproduced;
- the author repository was unavailable during assessment;
- paper equations are context, while grid results are recomputed evidence;
- the exact constrained-RLHF result and Equations 13–16 optimal-policy margin
  are distinct from the Equation 17 reference-policy approximation;
  and
- only the challenge can issue official verdict labels.

## 10. Testing and failure behavior

Every production behavior starts with a failing pytest. Focused tests cover:

- identity and claim hashing;
- policy and stable-softplus validation;
- analytic RLHF and population-loss stationary points;
- negative derivatives for the one-sided DPO loss;
- relative/absolute criterion separation;
- exact undesirable-space membership and witnesses;
- exact constrained-RLHF optimum classification and certificates;
- Equations 13–16 optimal-policy margin and residual, when defined;
- Equation 17 reference-policy approximation, predicted shift, derivative,
  and approximation error when comparable;
- soft-margin convergence and negative targets;
- closed evidence schema and deterministic bytes;
- atomic CLI writes and sanitized errors; and
- static root metadata, offline assets, and honest limitations.

The full project suite must pass offline. Two complete evidence generations
must be byte-identical. `git diff --check`, strict JSON parsing, project
pytest, root pytest excluding archival NAPE, and repository pre-commit are
controller validation inputs. A worker may run tests and report them, but only
the controller can attest validation.

## 11. Security, licensing, and authority

- No GPU, paid API, model weights, datasets, or Hub credentials are needed.
- No subprocess accepts user-controlled shell text.
- The numerical core reads no files and performs no network access.
- Pages render strings through DOM text nodes, not `innerHTML`.
- The project is original code. Paper equations and facts are attributed to
  immutable arXiv v1 URLs.
- The missing GitHub repository is neither downloaded nor redistributed.
- Workers may edit only the controller-assigned paper worktree and project.
- Workers must not modify coordinator state, this design after approval, the
  reproduction-loop skill, another submission, or `docs/HANDOFF.md`.
- Workers must not deploy, submit, poll, import verdicts, merge, or claim
  external phases.

## 12. Success criteria

The implementation proposal is ready for controller review when:

1. all five selected lanes execute from explicit finite grids;
2. lane 1 preserves the distinction between one-sided and population DPO;
3. every reported number is recomputed rather than copied from the paper;
4. all six live claim strings and hashes equal immutable expected constants in
   admitted order, and the SOTA claim is explicitly not reproduced;
5. the evidence schema rejects missing identity, non-finite values, unknown
   fields, and invalid outcomes;
6. two generation runs are byte-identical;
7. the project test suite passes offline;
8. root `index.html` and `poster.html` render the committed evidence without
   external assets;
9. the evidence bundle records commands, environment, validation, source
   version, and limitations; and
10. the worker returns a commit and evidence paths only as a proposal for
    controller validation.
