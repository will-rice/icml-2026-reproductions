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

The reproduction has exactly two scheduler targets. These strings are copied
verbatim from the admitted attempt and are immutable identifiers:

1. `The paper casts dataset pruning as a graph problem with node weights for intrinsic importance and edge weights for extrinsic diversity/interaction, yielding a Maximum Weight Clique formulation (Section 3.3).`
2. `Under mild conditions, the unified objective becomes submodular and admits a greedy approximation guarantee (Section 3.6; Appendix F).`

The first target asks whether the paper's dataset-pruning objective is exactly
the fixed-cardinality maximum vertex-and-edge-weight clique objective it
states. The second asks whether the stated premises imply diminishing returns,
monotonicity, and the claimed \(1-1/e\) greedy/optimum bound.

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

Appendix E contains two distinct statements that must not be normalized into
one another. Its inline prose defines

\[
I_{\mathrm{in}}^{\mathrm{revised}}(x_i)
=I_{\mathrm{in}}(x_i)+\sum_{j=1}^{|\hat S|}\eta
\tag{Appendix E inline definition, PDF p. 15}
\]

but displayed Eq. (26) instead states

\[
\Delta_{\mathrm{paper\text{-}26}}(x_i\mid\hat S)
=\alpha I_{\mathrm{in}}(x_i)
+\alpha\sum_{j=1}^{|\hat S|}\eta
+\sum_{x_j\in\hat S}g(D(x_i,x_j)).
\tag{Appendix E, Eq. 26, PDF p. 15}
\]

For symmetric \(a_{ij}=g(D(x_i,x_j))\), the natural literal substitution into
Eq. (4)--(5) defines

\[
f_{\mathrm{appendix\text{-}inline}}(S)
=f_{\mathrm{lit}}(S)+\alpha\eta|S|^2.
\]

Its independently derived actual marginal is

\[
f_{\mathrm{appendix\text{-}inline}}(S\cup\{x\})
-f_{\mathrm{appendix\text{-}inline}}(S)
=\alpha I_{\mathrm{in}}(x)+2\sum_{j\in S}a_{xj}
+\alpha\eta(2|S|+1),
\]

not displayed Eq. (26). For the separately repaired single-counted objective,
the corresponding marginal is
\(\alpha I_{\mathrm{in}}(x)+\sum_{j\in S}a_{xj}
+\alpha\eta(2|S|+1)\).
The manifest stores the inline definition, displayed Eq. (26), and both derived
marginals as distinct records. It also transcribes

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
- `appendix_inline_shift_literal`: the inline Appendix E replacement
  \(I_{\mathrm{in}}+|S|\eta\) applied to literal Eq. (4)--(5), yielding the
  cardinality term \(\alpha\eta|S|^2\).
- `appendix_eq26_score`: displayed Eq. (26), retained as a score rather than
  silently treated as the marginal of a defined set function.
- `modular_shift_candidate`: a separately labeled repaired objective that
  adds a fixed per-selected-element constant, with its coefficient stated for
  the chosen single- or double-counted objective.

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

The symbolic reduction is primary: for symmetric interactions,

\[
F_{\mathrm{MWCP}}(S)=\sum_{i\in S}w_i+\sum_{\{i,j\}\subseteq S}a_{ij},
\qquad
f_{\mathrm{lit}}(S)=\sum_{i\in S}w_i
+2\sum_{\{i,j\}\subseteq S}a_{ij}.
\]

Thus equality for arbitrary weights reduces to the edge-coefficient identity
\(1=2\), while the half-corrected expression reduces symbolically to matching
coefficients for every \(n\). A labeled **exhaustive finite witness search**
then covers only \(n\in\{1,2\}\), every nonempty selected set, vertex weights
in \(\{0,1\}\), and the sole symmetric edge weight (when present) in
\(\{0,1\}\): at most
\(2+3\cdot2^2\cdot2=26\) objective cases. It seeks the smallest nonzero-edge
witness. If found, the full graph, selected set, exact weights, independently
computed totals, and symbolic coefficient delta are persisted. This finite
search is exhaustive over its stated domain; the all-\(n\) coefficient result
is symbolic, not an enumeration claim.

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

For symmetric interactions the symbolic oracles independently derive

\[
\Delta_{\mathrm{lit}}(x\mid S)=w_x+2\sum_{j\in S}a_{xj},\qquad
\Delta_{\mathrm{single}}(x\mid S)=w_x+\sum_{j\in S}a_{xj}.
\]

Their diminishing-return differences are respectively
\(-2\sum_{j\in B\setminus A}a_{xj}\) and
\(-\sum_{j\in B\setminus A}a_{xj}\). This establishes the general sign result
symbolically while also testing whether Eq. (12) is the actual marginal.

The labeled **exhaustive symmetric control** covers \(1\leq n\leq4\), every
triple \(A\subseteq B\), \(x\notin B\), and every symmetric edge assignment in
\(\{-1,0,1\}\). Vertex terms are omitted only after the symbolic cancellation
is recorded. Its exact ceiling is
\(\sum_{n=1}^4 n3^{n-1}3^{\binom n2}=79{,}480\) triple-assignment cases. A
separate labeled **exhaustive asymmetric diagnostic**, which is outside the
paper's metric premise, covers \(1\leq n\leq3\) and directed weights in
\(\{-1,0,1\}\), with ceiling
\(\sum_{n=1}^3 n3^{n-1}3^{n(n-1)}=19{,}738\). No-witness outcomes are called
exhaustive only over these finite domains; the arbitrary-real conclusion comes
only from the displayed symbolic reductions.

### Monotonicity and shift oracle

For every \(S\subseteq T\) and \(x\notin S\), this oracle checks
\(\Delta(x\mid S)\geq0\). It audits unshifted objectives, the Appendix E inline
shift, displayed Eq. (26), Eq. (27), and the separately named modular-shift
candidate. Symbolically, nonnegative intrinsic terms can only increase a
marginal, and positive \(\alpha\) permits normalization of the zero-intrinsic
worst case to \(\alpha=1\). With \(|a_{ij}|\leq M\) and Eq. (27), displayed
Eq. (26)'s \(\alpha|S|\eta\) term covers its \(|S|\) incident penalties. The
actual Appendix-inline marginal's \(\alpha\eta(2|S|+1)\) term covers the
literal objective's at most \(2|S|M\) penalty. By contrast, a repaired *fixed*
modular shift needs a coefficient of at least \(|S|M\) for the single-counted
objective or \(2|S|M\) for the literal objective; it cannot reuse Eq. (27)'s
single-edge bound without the Appendix's cardinality multiplier.

The labeled **exhaustive shift boundary search** covers \(1\leq n\leq4\),
zero intrinsic weights, \(\alpha=1\), all symmetric edge assignments in
\(\{-1,0\}\), every \((S,x)\), and the deduplicated values immediately below,
at, and above the assignment's Eq. (27) threshold. Before deduplication its
ceiling is
\(3\sum_{n=1}^4 2^{\binom n2}n2^{n-1}=6{,}459\) marginal cases. A separate
**non-exhaustive boundary control** uses positive intrinsic values and rational
\(\alpha\) to check scale handling; it has a fixed ceiling of 256 generated
cases and cannot support a universal pass.

It specifically verifies that the paper's single-edge bound suffices only in
combination with its cardinality multiplier, and that both the displayed and
actual set-size-dependent terms shift all candidates equally at a fixed greedy
iteration. It separately falsifies any fixed-shift reinterpretation that is too
small. Any failure stores the minimal
\((T,S,x,\alpha,I_{\mathrm{in}},a,\eta)\) witness and the exact negative
marginal.

### Greedy-versus-optimum oracle

There are two mandatory greedy paths:

- `paper_eq7_score_greedy` implements Eq. (8) using exactly Eq. (7),
  \(w_x+\sum_{j\in S}a_{xj}\), without calling an objective marginal; and
- `true_marginal_greedy` computes
  \(F(S\cup\{x\})-F(S)\) from the selected objective's independent evaluator.
  For `paper_samplewise_literal`, this is
  \(w_x+2\sum_{j\in S}a_{xj}\).

Neither path may call the other. The optimum implementation enumerates all
size-\(b\) subsets directly. Deterministic tie handling evaluates all tied
paths for each greedy implementation; the evidence reports best, worst, and
canonical lexicographic values separately. This prevents an arbitrary tie
break or the Eq. (7)/true-marginal mismatch from being hidden.

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

The labeled **exhaustive greedy domain** uses \(1\leq n\leq4\),
\(1\leq b\leq\min(3,n)\), vertex weights in \(\{0,1,2\}\), and symmetric edge
weights in \(\{-1,0\}\). It contains exactly
\(\sum_{n=1}^4\min(3,n)3^n2^{\binom n2}=16{,}239\) weighted-cardinality
instances. Each optimum enumerates at most six selected sets, and each all-ties
greedy traversal has at most \(P(4,3)=24\) terminal paths, so each greedy
implementation evaluates at most 389,736 terminal paths and all optimum calls
together evaluate at most 97,434 selected sets. Claims outside this domain rely
on symbolic proof-ledger reasoning, not an exhaustive label.
Additional seeded examples, if any, are explicitly **non-exhaustive smoke
tests** with a ceiling of 100.

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

The arbitrary-set cardinality checks and algebraic transitions are labeled
**symbolic**. A separate **exhaustive finite proof-ledger control** reuses the
16,239 weighted-cardinality instances from the greedy domain rather than
opening another Cartesian product. It emits at most one row for each of the 11
numbered steps Eq. (28)--(38), for a ceiling of 178,629 rows. The ledger records
`not_applicable` rather than manufacturing a numerical check when a symbolic
step has no instance-level predicate.

## Search accounting and minimization

There is no global all-assignments-through-\(n=6\) run. Each oracle uses the
symbolic reduction and bounded domain stated above. `fractions.Fraction` is
used whenever \(\alpha\), \(\eta\), or a ratio is non-integral; floating point
is not used for truth decisions. Before execution, the runner checks the
declared formula and ceiling, refuses an undeclared domain expansion, and
records the exact domain, formula, cases examined, deterministic order,
early-stop policy, completion status, and `exhaustive_finite`, `symbolic`, or
`non_exhaustive` label. Early-stopped witness searches are not labeled
exhaustive even if their enclosing domain is finite.

Property-specific reductions do not transfer a result between oracles. A found
witness is minimized by:

1. vertex deletion;
2. selected-set/cardinality reduction;
3. zeroing unnecessary weights;
4. reducing absolute weight magnitudes; and
5. lexicographic canonicalization.

The minimized witness and the pre-minimization discovery are both retained.
Regression fixtures are generated from canonical witness JSON, never copied
from prose.

Counting weighted greedy instances, optimum subsets, and both greedy
implementations separately, all declared finite controls have an aggregate
ceiling of 1,177,833 case, path, subset, or ledger-row evaluations.
They use exact arithmetic, at most four vertices, and no GPU, network call, or
model training. This preserves the assessed CPU-only, under-30-minute scope;
the evidence records wall time and fails the lifecycle gate rather than silently
shrinking a domain if that bound is exceeded.

## TDD sequence

Implementation follows failing-test-first development:

1. Transcription-schema and checksum tests fail before the manifest exists.
2. Independent objective-oracle tests fail before either evaluator exists.
3. A two-vertex nonzero-edge test fails before structured objective comparison
   exists. Its expectations are independently derived exact totals: for
   \((w_1,w_2,a_{12})=(1,2,-1)\), Eq. (3) totals \(2\), literal Eq. (4)--(5)
   totals \(1\), and `samplewise_minus_mwcp` is \(-1\), with edge coefficients
   `2` and `1`. The test asserts this structured mismatch record, not an
   impossible neutral equality and not a preselected claim verdict.
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
  "target_claims": [
    "The paper casts dataset pruning as a graph problem with node weights for intrinsic importance and edge weights for extrinsic diversity/interaction, yielding a Maximum Weight Clique formulation (Section 3.3).",
    "Under mild conditions, the unified objective becomes submodular and admits a greedy approximation guarantee (Section 3.6; Appendix F)."
  ],
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
Each search includes `oracle`, `model_variant`, `greedy_path` when applicable,
its exhaustive/symbolic label, exact domain, ceiling formula and value, actual
case count, completion status, and code revision. Each witness includes exact
rational inputs as numerator/denominator strings, all intermediate values,
`universal_claim_falsified`, `minimality_checks`, and artifact SHA-256.
Each claim result includes a stable local claim ID, `target_claim` equal to one
of the two exact strings above, expected observation, computed observations,
witness links, status, and limitations. Every result belongs to exactly one
target. Submission and improvement verdict payloads are generated only from
these two strings, preserve their order and spelling byte-for-byte, and reject
missing, additional, or rewritten claim text before any coordinator mutation.

The human report is generated from this JSON and cannot introduce new numeric
claims. JSON Schema validation, stable sorting, deterministic serialization,
and a second clean-run byte comparison are required.

## Attribution and licensing

The parent convention is MIT for root-authored work (`LICENSE` and `README.md`)
and separate licensing for bundled material. The submission follows that
convention with explicit file boundaries:

- `submissions/graph-pruning/LICENSE` contains the MIT License and covers
  original software in `src/`, `tests/`, `app.py`, `pyproject.toml`, `uv.lock`,
  and original JSON Schema files.
- `submissions/graph-pruning/LICENSES/CC-BY-NC-SA-4.0.txt` contains the Creative
  Commons Attribution-NonCommercial-ShareAlike 4.0 International legal code.
  It covers adapted/transcribed material in `paper_transcriptions/`,
  `evidence/`, `README.md`, `poster.html`, `poster_embed.html`, and explanatory
  Space assets.
- `submissions/graph-pruning/NOTICE.md` names all seven authors, paper title,
  exact arXiv v2 URL, source license, adaptation status, both licenses, and the
  file-boundary map. The Space exposes the same notice.

Transcribed equations are attributed at point of use with equation, PDF page,
and revision. Generated evidence containing those transcriptions remains under
CC BY-NC-SA 4.0; original executable code remains MIT and does not absorb the
paper license. No paper figures, tables, experimental images, or unreleased
code are copied.

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
