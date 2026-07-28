# RACO Score Reproduction Design

## Authority and immutable binding

This design covers attempt `97e213a5-7ca3-4a1b-a500-1ec52d94d87a`,
paper `vSzRJyg6k0`, **Reward-free Alignment for Conflicting Objectives**.
It is bound to admitted snapshot
`09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199`
and immutable upstream
`arxiv:2602.02495v3+github:PeterLauLukChen/RACO@84a943c34f38520c7e0c9dd3066517c111b3c8fa`.

The attempt is `design-pending`. This document is a design proposal only. It
does not approve itself, grant a writer lease, attest evidence, publish a
Space, submit an entry, or establish an official verdict.

## Exact live claims

The evidence bundle and reviewer pages contain all ten live claims in this
exact order. SHA-256 is over the exact UTF-8 challenge text.

| # | Exact live claim | SHA-256 | Scope |
|---:|---|---|---|
| 1 | RACO is an offline, reward-free preference-alignment method that accepts user-specified objective weights and explicitly handles conflicting objectives (Table 1). | `e9a35e34b57a7273bf84d3d5981ab19f8ff1088adef8363f4640dcf436183944` | source and algorithm audit |
| 2 | The method uses CAGrad-Clip to limit correction gradients so updates better respect preferred objective trade-offs (Figure 1, Algorithm 1). | `7c0aa54e034d03f2d0905417a024af4db41338cd9a803a0b42e441945c307cf9` | source and executable algorithm audit |
| 3 | On TL;DR summarization, RACO achieves better Pareto frontiers for conciseness-quality and faithfulness-quality trade-offs than AMoPO and weighted-loss DPO baselines (Figure 2, Figure 3). | `85abbc8a21d5c4537409f6e9f2af6bffc7e4c15e2311dfa078bf816ea0cffc9e` | GPU-scale; not recomputed |
| 4 | On BeaverTails safety alignment, RACO improves harmlessness-helpfulness Pareto trade-offs across Qwen3 and Gemma3 setups (Figure 4). | `dac93f364ac0469302894920781b034bfcd205816fbe16537c2f8e7c10d8995d` | GPU-scale; not recomputed |
| 5 | Ablations show clipping and the correction-radius constant affect validation margins and Pareto frontiers (Figure 5, Figure 6). | `269d8a5053e224206036399bccb2435455565149086de6439a9046d89682772b` | paper context only |
| 6 | RACO directly applies conflict-averse gradient descent to objective-specific pairwise preference losses instead of relying on explicit reward models (Section 3). | `0d457572ea8a502fa8a489fef3e15da21b13cc39dd3a3730843d1cbe833059b0` | selected executable claim |
| 7 | The clipped CAGrad update is introduced to stabilize multi-objective LLM alignment while respecting user-specified objective weights (Section 3.2). | `50719d645042a500f9c4d53fbdfedf719ee56429ddb73a231912f1eaeadb1b31` | selected executable claim |
| 8 | The paper proves convergence of clipped CAGrad to Pareto-critical points that respect user-specified weights in nonconvex smooth settings (Theorem 3.1). | `5ec835ce150ff60d1e2bbd4fbdf7d1ebacf91bb2b6b2d65f72c44c3b3ed65229` | selected theorem audit |
| 9 | For two objectives, the analysis shows clipping can strictly improve the convergence rate (Theorem 3.2). | `b74a0ea75967144b210934fd40fd23449d3ef985df878d5a9e14c4b04025ba4b` | selected descent-certificate audit |
| 10 | Experiments on multi-objective summarization and safety alignment across Qwen 3, Llama 3, and Gemma 3 report better Pareto trade-offs than reward-free baselines (Section 4). | `58b31f527bb5e1bccb05c0dab775a74c2f2bdcd8e92ef2c0dd578733b5fb058e` | GPU-scale; not recomputed |

Claims 6–9 are the four admitted target claims. Claims 1–2 receive supporting
artifact observations from the same algorithm audit. Claims 3–5 and 10 remain
in the bundle and pages with local outcome `limited`; their paper-reported
numbers are never measurement fields.

## Scope and selected approach

Build independently executable CPU evidence for:

1. objective-specific DPO-style pairwise losses without reward models;
2. the exact weighted two-objective CAGrad subproblem;
3. coordinate-wise CAGrad-Clip, including singular cases;
4. Theorem 3.1's deterministic descent and stationarity inequalities; and
5. Theorem 3.2's one-step descent-certificate identity and strictness
   conditions.

The implementation is dependency-light, deterministic, offline after pinned
inputs are acquired, and uses no GPU or paid API. It does not train an LLM,
rerun TL;DR or BeaverTails, or reproduce model-family Pareto frontiers.

The selected approach is a source-faithful two-objective solver plus analytic
finite-dimensional audits. Wrapping the upstream training stack would exceed
the CPU boundary; copying paper tables would be self-report rather than
reproduction evidence.

## Components

- `provenance.py`: fail-closed manifest, SHA-256, Git-blob, safe-path, paper
  version, repository commit, snapshot, and live-claim verification.
- `pairwise.py`: stable objective-specific DPO losses and gradients.
- `cagrad_clip.py`: weighted two-objective simplex solver, coordinate clipping,
  singular-case behavior, and diagnostics.
- `theorem_audit.py`: Theorem 3.1 inequalities and Theorem 3.2 per-step
  certificate calculations on analytic gradients.
- `evidence.py`: closed schema, all ten ordered claims, local outcomes,
  measured observations, commands, limitations, and canonical JSON.
- `cli.py`: offline `audit` and `validate` commands with atomic output.
- root `pages/*.md`, `README.md`, and `app.py`: read-only reviewer surfaces
  over committed evidence.

No presentation component calculates results independently.

## Weighted two-objective CAGrad-Clip

### Inputs and validation

For gradients `g1,g2`, user weights `w=(w1,w2)`, and radius `c`:

```text
w1 >= 0
w2 >= 0
w1 + w2 = 1
0 <= c < 1
g0 = w1*g1 + w2*g2
```

All tensors must have equal shape and finite values. The implementation keeps
the user weights in the solver interface because both the anchor `g0` and the
subproblem depend on them.

### Exact one-dimensional subproblem

Let `p=(alpha,1-alpha)` with `alpha in [0,1]`,
`Gp(alpha)=alpha*g1+(1-alpha)*g2`, and `s=c*||g0||`. Solve:

```text
argmin h(alpha)
h(alpha) = <Gp(alpha),g0> + s*||Gp(alpha)||
```

Using the paper's Appendix B.1 notation:

```text
b1    = <g1,g0>
b2    = <g2,g0>
delta = b1-b2
H11   = <g1,g1>
H12   = <g1,g2>
H22   = <g2,g2>
q2    = H11+H22-2*H12
q1    = 2*(H12-H22)
q0    = H22
Q(a)  = q2*a^2+q1*a+q0
h(a)  = b2+delta*a+s*sqrt(Q(a))
```

Candidate points are endpoints, real in-range roots of `Q(alpha)=0`, and real
in-range stationary roots of:

```text
(delta^2*q2-s^2*q2^2)*alpha^2
+(delta^2*q1-s^2*q1*q2)*alpha
+delta^2*q0-s^2*q1^2/4 = 0.
```

Every candidate is evaluated in the original unsquared objective `h`; this
prevents a root introduced by squaring the first-order condition from being
accepted automatically. Ties use the candidate closest to `w1`, then the
smaller alpha, so output is deterministic and weight-preserving.

### Singular cases

The solver handles these cases explicitly rather than dividing by a vanishing
quadratic coefficient or norm:

- identical gradients (`q2=0`): choose `p=w`, because every mixture direction
  is identical;
- `g0=0`: choose `p=w`, set the correction radius to zero, and return update
  direction zero;
- `c=0`: minimize the remaining linear objective on `[0,1]`, using `w1` on a
  tie;
- colinear, opposite, or zero gradients: evaluate endpoints, zero-mixture
  points, and all valid stationary candidates through the same original
  objective;
- degenerate quadratic: solve the remaining linear equation, or record no
  stationary candidate if both coefficients vanish;
- `||Gp||=0` or `||G_tilde||=0`: use a zero correction vector, never normalize
  it.

All singular-case decisions and tolerances appear in evidence.

### Coordinate-wise clipping and update

After solving `p`, apply exactly:

```text
p_tilde_i = min(p_i,w_i)       for i in {1,2}
G_tilde   = p_tilde_1*g1 + p_tilde_2*g2
```

Do not renormalize `p_tilde`; renormalization would undo the paper's weight
budget. The update direction is:

```text
G0 = g0 + c*||g0||*G_tilde/||G_tilde||   if ||G_tilde|| > 0
G0 = g0                                   otherwise.
```

Diagnostics record `w`, `g0`, `p`, `p_tilde`, active clipped coordinates,
`Gp`, `G_tilde`, correction norm, and the final update.

## Theorem 3.1 audit

The audit records, rather than assumes, the paper's deterministic
preconditions:

```text
w in the probability simplex
each Li has li-Lipschitz gradient
lw = sum_i w_i*li
0 < eta <= 1/lw
0 <= c < 1
Li >= 0 for the DPO fixture
```

For analytic smooth objectives it recomputes:

```text
M(theta) = min_{lambda in simplex} ||sum_i lambda_i*grad Li(theta)||
||grad Lw|| = ||sum_i w_i*grad Li||
M(theta) <= ||grad Lw||
Gamma(rho) >= (1-c^2)/2
Lw(theta_next) <= Lw(theta)-eta*||g0||^2*Gamma(rho)
```

It also checks the finite-horizon bound:

```text
min_{0<=t<T} M(theta_t)^2
<= min_{0<=t<T} ||grad Lw(theta_t)||^2
<= 2*Lw(theta_0)/(eta*(1-c^2)*T).
```

Finite examples do not prove the general theorem. They test its algebra,
preconditions, and executable update, and their local outcome is `supported`,
`not-supported`, or `limited`.

## Theorem 3.2 audit

Theorem 3.2 is not an empirical iteration-count comparison. It compares the
per-step descent certificate of unclipped and clipped correction directions.

For `g0 != 0`, define:

```text
u       = Gp/||Gp||                 if ||Gp||>0, else 0
u_tilde = G_tilde/||G_tilde||       if ||G_tilde||>0, else 0
rho       = <g0,u>/||g0||
rho_tilde = <g0,u_tilde>/||g0||
Gamma(rho) =
    (1+c*rho) - (lw*eta/2)*(1+c^2+2*c*rho)
```

The primary reproduced identity is:

```text
Gamma(rho_tilde)-Gamma(rho)
= c*(1-lw*eta)*(rho_tilde-rho).
```

Under `m=2`, `w1,w2>0`, `c>0`, and `eta<1/lw`, the difference is
nonnegative. It is strictly positive only when all additional conditions hold:

```text
g1 and g2 are not colinear
p1>0 and p2>0
p != w
g0 != 0
```

The audit records each Boolean separately. Cases outside the theorem domain,
including `g0=0`, are `limited`/not applicable rather than forced into the
identity. It may record realized objective decreases as secondary diagnostics,
but it must not translate the theorem into “fewer iterations.”

## Evidence semantics and schema

Local evidence outcomes are exactly:

- `supported`;
- `not-supported`;
- `limited`.

The words `verified`, `falsified`, and `toy` are reserved for official
challenge verdicts imported by the controller. They must not appear as local
claim outcomes, headings, badges, or conclusions.

`evidence/results.json` uses a closed schema and contains:

- attempt, paper, admitted snapshot, and upstream identity;
- all ten exact live claims and hashes in order;
- target status for claims 6–9;
- paper context separated from measured observations;
- solver inputs, singular-case decisions, theorem preconditions, tolerances,
  raw rows, and local outcomes;
- unavailable GPU/API operations and limitations;
- pinned artifact hashes and acquisition lineage; and
- exact regeneration and validation commands.

Canonical JSON rejects non-finite values, sorts keys, uses stable separators,
and ends with one newline. Two offline generations from the same verified
inputs must be byte-identical.

## Judge-readable root pages

The project root contains committed pages:

```text
pages/00-summary.md
pages/01-objective-losses.md
pages/02-cagrad-clip.md
pages/03-theorem-31.md
pages/04-theorem-32.md
pages/05-limitations-and-provenance.md
```

Each page includes the exact relevant live claim text and hash, local outcome,
computed observation, formula, source pin, command, and limitation.
`00-summary.md` lists all ten claims in order. `app.py` loads these files and
committed evidence without network access; `README.md` has valid Space
metadata and links directly to every page.

The pages prominently state that local outcomes are not official verdicts,
that empirical Pareto claims were not recomputed, and that no LLM, GPU, paid
API, or reward model was used.

## Testing and acceptance

Every production behavior starts with a failing pytest. Tests cover exact
claim text/hash/order, snapshot identity, manifest tampering, unsafe paths,
weighted anchors, asymmetric weights, zero weights, coordinate-wise clipping,
no post-clip renormalization, identical/colinear/opposite/zero gradients,
`g0=0`, `c=0`, degenerate stationary equations, Theorem 3.1 preconditions,
Theorem 3.2 identity and strictness, invalid local verdict words, canonical
evidence, page completeness, offline app import, and deterministic generation.

The complete evidence run is CPU-only, uses USD 0.00 in APIs, and must finish
within 30 minutes on an ordinary workstation. The worker may modify only
`submissions/reward-free-alignment-for-conflicting-objectives/` in its
controller-assigned worktree. Its commits and test reports remain proposals
until separate controller validation and attestations.
