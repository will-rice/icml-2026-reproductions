# Graph Dataset Pruning Formal-Evidence Reproduction Design

## Authority, attempt, and phase

- Attempt: `e485c086-6fa5-4ff6-a3c3-1f31c79bbae6`
- Challenge paper: `a3GdvuPItd`
- Author and current writer: `codex-graph-pruning-writer`
- Pinned paper: Dongyue Wu et al., *Selecting Samples on Graphs: A Unified
  Dataset Pruning Framework for Lossless Training Acceleration*,
  `arxiv:2606.12913v2`, dated 2026-07-03.
- License shown by arXiv: CC BY-NC-SA 4.0.
- Phase covered by this document: design. The live claim refresh and immutable
  assessed snapshot that admitted this attempt remain coordinator provenance;
  this task must not refresh live state or mutate coordinator files. After this
  design is committed, a separate actor records it with the attempt's current
  owner and fencing token, and a different reviewer records approval. Only then
  may the attempt enter `implementing`.
- Approval: the user's 2026-07-24 instruction gives standing autonomous
  approval for this design. It does not waive independent review, fencing, TDD,
  validation, deployment verification, or live-refresh requirements.

This design deliberately produces independent formal evidence. It does not
restate reported CIFAR or ImageNet numbers as measurements, and it does not use
small successful examples as proof of universal claims.

## Target claims and verdict boundaries

The reproduction has exactly two positive target areas:

1. **objective-to-MWCP formulation** — whether the paper's dataset-pruning
   objective is exactly the fixed-cardinality maximum vertex-and-edge-weight
   clique objective it states; and
2. **submodularity and greedy approximation guarantee** — whether the stated
   premises imply diminishing returns, monotonicity, and the claimed
   \(1-1/e\) greedy/optimum bound.

Each area receives claim-level observations, but no claim is declared verified
merely because a corrected formulation works. Results for the literal paper
formulation, a conventional single-counted objective, and any repaired
premises are separate records. A universal counterexample to the literal
statement is a contradiction witness. A generated positive instance is only a
synthetic smoke test.

The following are explicitly unavailable:

- CIFAR-10/100 accuracy and comparison claims;
- ImageNet-1k accuracy, training-time, and lossless-acceleration claims;
- results requiring unreleased training code, pretrained features, or GPU
  training; and
- semantic-segmentation and object-detection experiments.

They appear in limitations and context only, never in reproduced outputs.

## Primary-source transcription and provenance

Implementation will include a hand-audited, immutable transcription manifest.
Every expression stores paper revision, PDF page, section or appendix, equation
number, a normalized symbolic form, and the transcription checksum. The
rendered report links to `https://arxiv.org/pdf/2606.12913v2`; it must not
silently follow a newer arXiv revision.

The required transcription is:

\[
w_i=\alpha I_{\mathrm{in}}(x_i),\qquad
a_{ij}=g(D(x_i,x_j)) \tag{2, PDF p. 2}
\]

\[
F_{\mathrm{MWCP}}(C)=
\sum_{v_i\in C}w_i+
\sum_{\{v_i,v_j\}\subseteq C}a_{ij},
\qquad |C|=b \tag{3, PDF p. 3}
\]

\[
f_{\mathrm{lit}}(S)=
\sum_{x_i\in S}
\left[\alpha I_{\mathrm{in}}(x_i)+I_{\mathrm{ex}}(x_i\mid S)\right],
\qquad |S|=b \tag{4, PDF p. 3}
\]

\[
I_{\mathrm{ex}}(x_i\mid S)=
\sum_{x_j\in S\setminus\{x_i\}}a_{ij}
=\sum_{x_j\in S\setminus\{x_i\}}g(D(x_i,x_j)) \tag{5, PDF p. 3}
\]

\[
\Delta^-(v_i\mid G)=w_i+
\sum_{v_j\in C\setminus\{v_i\}}a_{ij} \tag{6, PDF p. 3}
\]

\[
I(x_i\mid S)=\Delta(x_i\mid S)=
\alpha I_{\mathrm{in}}(x_i)+I_{\mathrm{ex}}(x_i\mid S) \tag{7, PDF p. 3}
\]

\[
x^\star\in\arg\max_{x_i\in T\setminus S_t}I(x_i\mid S_t),
\qquad S_{t+1}=S_t\cup\{x^\star\} \tag{8, PDF p. 3}
\]

\[
\Delta(x\mid A)=f(A\cup\{x\})-f(A),\qquad
A\subseteq B,\ x\notin B:
\Delta(x\mid A)\geq\Delta(x\mid B) \tag{10--11, PDF p. 4}
\]

and the paper's Eq. (12)--(14) marginal-difference identity under
\(D\geq0\) and \(g:\mathbb R_{\geq0}\to\mathbb R_{\leq0}\).

Appendix E is transcribed literally, including

\[
I_{\mathrm{in}}^{\mathrm{revised}}(x_i)
=I_{\mathrm{in}}(x_i)+\sum_{j=1}^{|\hat S|}\eta,
\tag{Appendix E, Eq. 26, PDF p. 15}
\]

and

\[
\eta\geq\frac1\alpha\max_{x_i,x_j}|g(D(x_i,x_j))|.
\tag{Appendix E, Eq. 27, PDF p. 15}
\]

Appendix F's proof chain, Eq. (28)--(38), is transcribed with special
attention to its use of \((b-t)\), the product
\(\prod_{k=1}^{b}(1-1/k)\), normalization, monotonicity, non-negativity,
and the final \(1-1/e\) ratio. The transcription records what the PDF says
without repairing notation in place.

## Considered approaches

Three approaches were considered:

1. **Positive finite examples only.** This is cheap but cannot establish a
   universal theorem and invites a `toy` verdict. Rejected.
2. **Computer algebra only.** Symbolic expansion finds coefficient mistakes
   cleanly but does not independently exercise greedy behavior or find minimal
   finite witnesses. Rejected as incomplete.
3. **Layered literal audit with independent oracles and exhaustive bounded
   search.** This combines exact transcription, symbolic identities,
   independently implemented numerical oracles, and smallest-first witness
   search. Selected because disagreements remain diagnosable and all universal
   claims are actively falsifiable.

## Formal model variants

The evidence engine exposes named, non-interchangeable objectives:

- `paper_mwcp`: Eq. (3), each undirected pair counted once.
- `paper_samplewise_literal`: Eq. (4) composed with Eq. (5), with no inserted
  factor.
- `single_counted_pairwise`: the conventional set function
  \[
  F(S)=\sum_{i\in S}w_i+\sum_{\{i,j\}\subseteq S}a_{ij}.
  \]
- `half_corrected_samplewise`: Eq. (4) with
  \(\tfrac12 I_{\mathrm{ex}}\), used only to explain a possible correction.
- `appendix_shift_literal`: Eq. (26) interpreted exactly as written for the
  current conditioning set.
- `modular_shift_candidate`: a separately labeled repaired objective that
  adds a fixed per-selected-element constant.

No repaired variant can overwrite the verdict for a literal variant. Every
result carries its `model_variant`.

## Independent evidence oracles

Each oracle has a narrow interface and must not call another oracle's objective
implementation.

### Objective-equivalence oracle

One implementation enumerates unordered edges from a graph adjacency map and
computes Eq. (3). A second traverses selected samples and computes Eq. (4)--(5)
literally. A symbolic oracle expands both into coefficients of \(w_i\) and
\(a_{ij}\). It reports coefficient differences and a concrete evaluation.

Search begins at \(n=1\) and increases by \(n\), selected-set size, and
lexicographic integer weights. It seeks the smallest nonzero-edge witness for
Eq. (3) versus Eq. (4), then separately verifies the half-corrected identity.
If a mismatch is found, the full graph, selected set, exact rational weights,
both totals, and symbolic coefficient delta are persisted.

The clique terminology is also audited: because the constructed graph is
fully connected, or sparsified missing edges are assigned zero weight, every
fixed-cardinality subset is feasible. The report distinguishes “an MWCP on a
complete graph with edge weights” from any nontrivial clique-feasibility
constraint.

### Diminishing-returns oracle

The first oracle directly enumerates all triples
\(A\subseteq B\subseteq T,\ x\notin B\) and calculates set-function
differences. The second computes the claimed closed-form marginal polynomial.
It checks:

- literal Eq. (4)--(5);
- the single-counted objective;
- symmetric and deliberately asymmetric interaction tables;
- \(g(D)\leq0\), zero, and one-premise-at-a-time violations; and
- sparse zero-weight edges.

This separates “the function is submodular” from “Eq. (12) is its actual
marginal.” A universal claim passes bounded exhaustive testing only if no
witness exists; the report calls this exhaustive evidence over the declared
finite domain, not a proof over all reals.

### Monotonicity and shift oracle

For every \(S\subseteq T\) and \(x\notin S\), this oracle checks
\(\Delta(x\mid S)\geq0\). It audits unshifted objectives, Eq. (26)--(27)
literally, and the separately named modular-shift candidate. The search varies
\(n\), \(\alpha>0\), nonnegative intrinsic weights, negative pair weights,
and \(\eta\) at, below, and above the stated threshold.

It specifically tests whether a bound on a single edge magnitude suffices when
the marginal contains several incident negative edges, and whether the
set-size-dependent term in Eq. (26) is truly an equal constant shift that
leaves greedy decisions unchanged. Any failure stores the minimal
\((T,S,x,\alpha,I_{\mathrm{in}},a,\eta)\) witness and the exact negative
marginal.

### Greedy-versus-optimum oracle

The greedy implementation uses only the selected objective's independently
computed marginal. The optimum implementation enumerates all size-\(b\)
subsets directly. Deterministic tie handling evaluates all tied greedy paths:
the evidence reports best, worst, and canonical lexicographic greedy values.
This prevents an arbitrary tie break from hiding a counterexample.

For each normalized, monotone, submodular instance it records

\[
\rho =
\begin{cases}
F(S_{\mathrm{greedy}})/F(S^\star),&F(S^\star)>0,\\
1,&F(S_{\mathrm{greedy}})=F(S^\star)=0,
\end{cases}
\]

and flags negative or undefined objective regimes rather than presenting a
misleading ratio. It searches smallest-first for:

- a violation of the claimed \(1-1/e\) bound under the paper's exact premises;
- a case showing the paper's \((b-t)\) proof step is invalid;
- a mismatch between Eq. (7)'s score and the literal Eq. (4) marginal; and
- smoke instances satisfying the standard repaired theorem premises.

### Appendix-premise oracle

This oracle is a proof ledger rather than a numerical shortcut. Each transition
in Eq. (28)--(38) records its required premise and is checked by exact rational
arithmetic on enumerated instances. At minimum it audits:

- normalization \(f(\varnothing)=0\);
- non-negativity and monotonicity, not submodularity alone;
- whether adding an objective constant preserves normalization and ratios;
- the bound on \(|S^\star\setminus S_t|\), which is at most \(b\) but need not
  be at most \(b-t\);
- the multiplication/product indices, including the \(k=1\) factor; and
- the logical relationship between a repaired standard greedy theorem and the
  theorem actually stated.

The output is a row per proof step with `supported`, `contradicted`, or
`not_applicable`, plus witness references.

## Exhaustive search domain and minimization

Default exhaustive search uses \(1\leq n\leq6\), all nonempty cardinalities,
integer vertex weights in \(\{0,1,2\}\), symmetric pair weights in
\(\{-2,-1,0\}\) for theorem-premise searches, and small signed values for
premise-violation controls. Fractions needed for \(\alpha\) and \(\eta\) use
`fractions.Fraction`; floating point is not used for truth decisions.

The runner estimates each Cartesian product before executing it and uses
property-specific pruning without sharing conclusions across oracles. It
records the complete domain, cases examined, deterministic enumeration order,
and early-stop policy. A found witness is minimized by:

1. vertex deletion;
2. selected-set/cardinality reduction;
3. zeroing unnecessary weights;
4. reducing absolute weight magnitudes; and
5. lexicographic canonicalization.

The minimized witness and the pre-minimization discovery are both retained.
Regression fixtures are generated from canonical witness JSON, never copied
from prose.

## TDD sequence

Implementation follows failing-test-first development:

1. Transcription-schema and checksum tests fail before the manifest exists.
2. Independent objective-oracle tests fail before either evaluator exists.
3. A two-vertex nonzero-edge regression test is introduced as a neutral
   equality expectation derived from the paper; its observed outcome determines
   the verdict, not a hard-coded desired contradiction.
4. Diminishing-returns enumeration and closed-form agreement tests fail before
   those oracles are implemented.
5. Monotonicity/shift boundary tests fail before Appendix E support exists.
6. Greedy, exhaustive optimum, all-ties, and ratio tests fail before solvers
   exist.
7. Proof-ledger and minimal-witness persistence tests fail before Appendix F
   auditing exists.
8. Evidence-bundle, report, poster, and Space rendering tests fail before their
   producers exist.

For every red phase, the log records command, timestamp, test identifier, and
the expected missing behavior. Tests assert mathematical identities and schema
invariants; they do not assert that the paper must pass or fail.

## Evidence bundle and schema

`submissions/graph-pruning/evidence/evidence.json` is the canonical computed
artifact. A proposed schema version `1` contains:

```json
{
  "schema_version": 1,
  "attempt_id": "e485c086-6fa5-4ff6-a3c3-1f31c79bbae6",
  "paper": {
    "challenge_id": "a3GdvuPItd",
    "revision": "arxiv:2606.12913v2",
    "source_url": "https://arxiv.org/pdf/2606.12913v2",
    "license": "CC BY-NC-SA 4.0"
  },
  "environment": {},
  "transcriptions": [],
  "searches": [],
  "witnesses": [],
  "claim_results": [],
  "unavailable_claims": [],
  "commands": [],
  "artifacts": []
}
```

Each transcription includes `equation`, `pdf_page`, `section`,
`normalized_expression`, `source_excerpt_sha256`, and `reviewed_by`.
Each search includes `oracle`, `model_variant`, exact domain, case count,
completion status, and code revision. Each witness includes exact rational
inputs as numerator/denominator strings, all intermediate values,
`universal_claim_falsified`, `minimality_checks`, and artifact SHA-256.
Each claim result includes claim ID, expected observation, computed
observations, witness links, status, and limitations.

The human report is generated from this JSON and cannot introduce new numeric
claims. JSON Schema validation, stable sorting, deterministic serialization,
and a second clean-run byte comparison are required.

## Attribution and licensing

The repository and Space identify the paper, all seven authors, title, exact
arXiv v2 URL, and CC BY-NC-SA 4.0 license. Transcribed equations are attributed
at point of use with equation, page, and revision. The reproduction's original
code and evidence state their own compatible license; any adapted paper
material is marked and distributed under CC BY-NC-SA 4.0 with the license text,
attribution, noncommercial notice, and ShareAlike notice. No paper figures,
tables, or experimental images are copied.

## Poster and Space

The poster and dedicated Hugging Face Space lead with the two theorem audits,
not experimental marketing claims. They show:

- provenance and literal equations;
- a table comparing literal and repaired model variants;
- independently computed pass/fail status per oracle;
- canonical minimal witnesses in readable form;
- greedy and optimum values with exact ratios;
- proof-ledger failures or supported premises;
- the exhaustive finite domain and its limits; and
- a prominent unavailable panel for ImageNet, CIFAR, segmentation, and
  detection claims.

The Space exposes downloadable canonical JSON and witness files and a
deterministic “recompute formal audit” path suitable for CPU execution. It
must not label repaired-theorem evidence as verification of the literal paper
claim.

## Failure handling and limitations

- A transcription ambiguity blocks the affected oracle until two independent
  readings agree; both readings remain recorded.
- Search interruption writes an incomplete checkpoint, but incomplete search
  never yields a pass.
- Oracle disagreement is a test failure and cannot be averaged or resolved by
  choosing the favorable result.
- Exhaustive bounded enumeration can refute a universal statement but cannot
  prove it over arbitrary real weights; symbolic identities supply the general
  evidence where possible.
- A repaired standard theorem can explain author intent but cannot erase a
  literal counterexample.
- No released implementation is available to resolve whether code used
  single- or double-counted edges or a different monotonicity shift.
- Empirical accuracy and acceleration remain unavailable because the required
  code/artifacts and autonomous GPU budget are unavailable.

## Validation and completion gates

The implementation plan must require:

1. clean-environment installation from locked dependencies;
2. the complete deterministic evidence command twice with byte-identical
   canonical JSON;
3. JSON Schema validation and report-to-JSON agreement;
4. the submission's full pytest suite;
5. root `uv run pytest -q`, while excluding the archival NAPE tree according
   to repository policy;
6. `uv run pre-commit run -a`;
7. credential, mutable-URL, cache, and unrelated-diff review;
8. `superpowers:verification-before-completion`;
9. commit of the exact validated source and evidence configuration;
10. deployment to a paper-specific Space and exact deployed-SHA verification;
11. exercise of the live Space recomputation and machine-readable download;
12. a fresh assessed live refresh immediately before submission, with
    cancellation if eligibility changed; and
13. fenced writes for `validated`, `deployed`, `submitted`, and bounded
    `judging`, followed by exact-claim verdict handling.

This design task stops after committing this document. It does not implement
the submission, record or approve the design in coordinator state, update
`docs/HANDOFF.md`, deploy a Space, submit to the challenge, or touch NAPE.

## Design self-review checklist

- No placeholder, TODO, or unresolved design choice remains.
- Literal, conventional, and repaired objectives have distinct identifiers.
- Each universal claim has an independent oracle and a falsification path.
- Positive synthetic cases are labeled smoke tests, never universal evidence.
- Minimal witnesses and complete search domains are preserved.
- Paper context, computed outputs, and unavailable evidence are distinct.
- Licensing, poster, Space, TDD, evidence schema, and validation are explicit.
- The scope is one CPU-bounded formal-evidence submission and is suitable for
  a single later implementation plan.
