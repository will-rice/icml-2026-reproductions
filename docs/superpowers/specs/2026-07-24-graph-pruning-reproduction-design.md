# Graph Dataset Pruning Formal-Evidence Reproduction Design

## Authority, attempt, and phase

- Attempt: `64bfe193-333b-4b37-9683-9ac25ca5ac27`
- Challenge paper: `a3GdvuPItd`
- Design author: `codex-graph-pruning-design-author-v2`
- Pinned paper: Dongyue Wu et al., *Selecting Samples on Graphs: A Unified
  Dataset Pruning Framework for Lossless Training Acceleration*,
  `arxiv:2606.12913v2`, dated 2026-07-03.
- License shown by arXiv: CC BY-NC-SA 4.0.
- Phase covered by this document: `design-pending`. The schema-v6 authoritative
  attempt shard pins immutable assessed snapshot
  `35d2104cb8462a652d933aa5a776f9b166e8c2724df12da7b35f54cbe19c883d`.
  Coordinator index, attempt, lease, judgment, transaction, and snapshot shards
  remain outside this design-author task. This task must not refresh live
  state, mutate coordinator files, implement the submission, or commit. The
  controller may record this proposal only after reviewing the focused
  two-document diff; a different reviewer identity must independently approve
  it before implementation.
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
premises are separate records. A greedy-guarantee violation is a theorem
counterexample only when the same finite instance has been exhaustively
certified to satisfy the theorem's global non-negativity, normalization,
monotonicity, and submodularity premises. A failure outside that premise set is
an `out_of_premise_diagnostic`, never a counterexample or guarantee violation.

The following are explicitly unavailable:

- CIFAR-10/100 accuracy and comparison claims;
- ImageNet-1k accuracy, training-time, and lossless-acceleration claims;
- results requiring unreleased training code, pretrained features, or GPU
  training; and
- semantic-segmentation and object-detection experiments.

They appear in limitations and context only, never in reproduced outputs.

## Primary-source transcription and provenance

Implementation will include a hand-audited, immutable transcription manifest.
Every record has exactly `record_id`, `equation`, `pdf_page`, `section`,
`normalized_expression`, `source_excerpt_path`,
`source_excerpt_byte_count`, `source_excerpt_sha256`, and `reviewed_by`.
Equation excerpts are stored as literal UTF-8 bytes under
`paper_transcriptions/excerpts/<record_id>.txt`; the Algorithm 1 record points
to `paper_transcriptions/algorithm1.txt`. Paths are unique canonical POSIX
paths, may not traverse or escape `paper_transcriptions/`, and each excerpt
file is referenced exactly once. Hashes and byte counts are computed from
`Path.read_bytes()` without newline, Unicode, or whitespace normalization.
The manifest never embeds a second `source_excerpt` string. A pinned
`TRANSCRIPTION_SET_SHA256` in `provenance.py` hashes the ordered tuples
`(record_id, source_excerpt_path, source_excerpt_byte_count,
source_excerpt_sha256)` and supplies a separately reviewed aggregate pin.
The rendered report links to `https://arxiv.org/pdf/2606.12913v2`; it must not
silently follow a newer arXiv revision.

The exact PDF acquisition and verification command is:

```bash
curl --fail --location --proto '=https' --tlsv1.2 --output /tmp/2606.12913v2.pdf https://export.arxiv.org/pdf/2606.12913v2 && test "$(wc -c < /tmp/2606.12913v2.pdf)" -eq 683737 && printf '%s  %s\n' 26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090 /tmp/2606.12913v2.pdf | sha256sum --check --strict
```

The versioned URL plus mandatory digest check identifies immutable source
bytes. Provenance and the canonical evidence must retain the command verbatim,
the full byte count `683737`, and SHA-256
`26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090`;
recording only the arXiv identifier, URL, or revision is insufficient. The PDF
itself is an acquisition input and need not be committed.

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

Algorithm 1 is a separate primary-source object, not an implementation gloss on
Eq. (7)--(8). Its literal line transcription (PDF p. 5; only PDF line wrapping
is normalized) is:

```text
Algorithm 1 Overall pipeline with Greedy Selection and Structured Graph Sparsification
Require: Training dataset T = {x_i}_{i=1}^N, pruning ratio p,
Ensure: Pruned subset S.
1: Build a fully connected graph G = (V, E) from T.
2: Compute intrinsic importance I^in(x_i) for all x_i ∈ T.
3: Perform Structured Graph Sparsification and get neighborhood clusters {N_k}_{k=1}^K.
4: Compute {D(x_i, x_j) | ∀x_i, x_j ∈ N_k} for all N_k.
5: Initialize S_0 ← ∅, I^ex(x_i | S_0) ← 0 for all x_i.
6: for t = 1 to (1 - p)N do
7:   for all x_i ∈ T \ S_{t-1} do
8:     if x⋆ ∈ N(x_i) then
9:       Fetch I^in(x_i) and D(x_i, x⋆).
10:      I^ex(x_i | S_t) ← I^ex(x_i | S_{t-1}) + g(D(x_i, x⋆))
11:      I(x_i | S_t) ← α I^in(x_i) + I^ex(x_i | S_t)
12:    end if
13:  end for
14:  Select x⋆ ← arg max_{x_i ∈ T \ S} I(x_i | S_t).
15:  Update S_t ← S_{t-1} ∪ {x⋆}.
16: end for
17: Return S ← S_{(1-p)N}.
```

The manifest retains a checksum of the source excerpt and this line-by-line
transcription. The literal audit must report, rather than repair, the apparent
operational ambiguities: line 5 initializes `S_0` and extrinsic values but no
unified `I(x_i | S_0)` scores; line 8 reads `x*` before line 14 first selects it;
lines 10--11 write values indexed by `S_t` before line 15 constructs `S_t`;
line 14 uses unindexed `S`, although line 5 initialized only `S_0`; and
candidates outside the line-8 neighborhood do not receive a score update before
the argmax. Any executable resolution (score initialization,
select-before-update, carry-forward scores, a chosen meaning of `S`, or a
first-iteration special case) is a separately named interpretation and cannot
be attributed to literal Algorithm 1.

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
without repairing notation in place. Eq. (28) is stored as two independently
statused conclusions within its one numbered ledger row:

1. `eq28_union_submodular_bound`:
   \(f(S^\star)\leq f(S_t)+
   \sum_{x\in S^\star\setminus S_t}\Delta(x\mid S_t)\), which requires the
   exact monotonicity/union and submodularity premises used to derive it; and
2. `eq28_b_minus_t_bound`: replacing that sum by
   \((b-t)\max_x\Delta(x\mid S_t)\), which additionally requires
   \(|S^\star\setminus S_t|\leq b-t\) and nonnegative candidate marginals.

Acceptance must not merge those conclusions or let the status of one stand in
for the other.

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
  cardinality term \(\alpha\eta|S|^2\). This is the literal Appendix E set
  function, not a repaired modular shift.
- `appendix_eq26_score`: displayed Eq. (26), retained as a score rather than
  silently treated as the marginal of a defined set function.
- `modular_shift_candidate`: a separately labeled repaired objective that
  uses the literal double-counted base and adds the fixed modular term
  \[
  F_{\mathrm{mod}}(S)=f_{\mathrm{lit}}(S)+\eta_{\mathrm{mod}}|S|.
  \]
  Here `Instance.eta` is exactly \(\eta_{\mathrm{mod}}\) for this variant. In
  the exhaustive greedy/proof domain it is fixed before evaluation to
  \(\eta_{\mathrm{mod}}=2(n-1)M\), where
  \(M=\max_{i\ne j}|a_{ij}|\). It never uses the single-counted base and never
  substitutes the Appendix-inline quadratic term.

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
differences. The second computes a separately implemented closed-form
marginal for each of the exact six set-function variants charged in the
symmetric accounting:

- `paper_mwcp`;
- `paper_samplewise_literal`;
- `single_counted_pairwise`;
- `half_corrected_samplewise`;
- `appendix_inline_shift_literal`; and
- `modular_shift_candidate`.

`appendix_eq26_score` is explicitly rejected because it is not a set function.
The oracle also checks:

- symmetric and deliberately asymmetric interaction tables;
- \(g(D)\leq0\), zero, and one-premise-at-a-time violations; and
- sparse zero-weight edges.

For any stored directed interaction table define
\[
d_x(S)=\sum_{j\in S}(a_{xj}+a_{jx}),\qquad
u_x(S)=\sum_{j\in S}a_{\min(x,j),\max(x,j)}.
\]
The latter uses the same canonical unordered-pair lookup as Eq. (3).
The six exact closed forms are
\[
\begin{array}{c|l}
\text{variant} & \Delta(x\mid S)\\ \hline
\texttt{paper\_mwcp} & w_x+u_x(S)\\
\texttt{paper\_samplewise\_literal} & w_x+d_x(S)\\
\texttt{single\_counted\_pairwise} & w_x+u_x(S)\\
\texttt{half\_corrected\_samplewise} & w_x+\tfrac12d_x(S)\\
\texttt{appendix\_inline\_shift\_literal}
  & w_x+d_x(S)+\alpha\eta(2|S|+1)\\
\texttt{modular\_shift\_candidate}
  & w_x+d_x(S)+\eta_{\mathrm{mod}}.
\end{array}
\]
`direct_marginal` obtains each value only by two calls to the independent
objective evaluator; `closed_form_marginal` may call neither that evaluator
nor `direct_marginal`. Tests compare both implementations for all six
variants on symmetric and asymmetric fixtures and assert the formula-specific
values, so agreement cannot be obtained by dispatching only the two old
literal/single formulas.

For symmetric interactions \(d_x(S)=2\sum_{j\in S}a_{xj}\) and
\(u_x(S)=\sum_{j\in S}a_{xj}\). The symbolic oracles therefore independently
derive

\[
\Delta_{\mathrm{lit}}(x\mid S)=w_x+2\sum_{j\in S}a_{xj},\qquad
\Delta_{\mathrm{single}}(x\mid S)=w_x+\sum_{j\in S}a_{xj}.
\]

The diminishing-return differences for `paper_mwcp`,
`single_counted_pairwise`, and `half_corrected_samplewise` are
\(-\sum_{j\in B\setminus A}a_{xj}\); those for
`paper_samplewise_literal` and `modular_shift_candidate` are
\(-2\sum_{j\in B\setminus A}a_{xj}\), because the fixed modular term cancels.
This establishes the general sign result symbolically while also testing
whether Eq. (12) is the actual marginal.

For `appendix_inline_shift_literal`, the independently derived difference is

\[
\Delta_{\mathrm{appendix}}(x\mid A)
-\Delta_{\mathrm{appendix}}(x\mid B)
=-2\sum_{j\in B\setminus A}a_{xj}
-2\alpha\eta(|B|-|A|).
\]

The audit must persist the minimal two-element falsification: let
\(T=\{x,y\}\), \(A=\varnothing\), \(B=\{y\}\), set both vertex weights and
the sole edge weight to zero, and set \(\alpha=\eta=1\). Then the shifted
literal objective is \(|S|^2\), so
\(\Delta(x\mid\varnothing)=1\) and
\(\Delta(x\mid\{y\})=3\), contradicting diminishing returns. Two elements are
minimal because a strict chain \(A\subset B\) with \(x\notin B\) cannot exist
on a one-element ground set. This witness applies only to
`appendix_inline_shift_literal`; it is not attributed to
`modular_shift_candidate` or any other repaired objective.

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
modular shift on the chosen literal base needs a coefficient at least
\(2|S|M\) at a particular set size. The exact
`modular_shift_candidate` fixes one coefficient for the whole \(n\)-vertex
domain:
\[
F_{\mathrm{mod}}(S)=f_{\mathrm{lit}}(S)+2(n-1)M|S|,
\quad
\Delta_{\mathrm{mod}}(x\mid S)
=w_x+2\sum_{j\in S}a_{xj}+2(n-1)M.
\]
Thus its `Instance.eta` is \(2(n-1)M\). It cannot reuse Eq. (27)'s
single-edge bound without a cardinality/degree multiplier, and it is never
implemented as `paper_mwcp + eta * |S|`.

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
iteration. The modular channel evaluates the exact literal-base objective
above; separate below-threshold diagnostics vary a candidate coefficient but
may not relabel those smaller coefficients as `modular_shift_candidate`. Any
failure stores the minimal
\((T,S,x,\alpha,I_{\mathrm{in}},a,\eta)\) witness and the exact negative
marginal.

### Greedy-versus-optimum oracle

There are three mandatory, non-interchangeable audit paths:

- `paper_algorithm1_literal` is a line-indexed state-machine audit of the exact
  Algorithm 1 transcription above. Undefined reads and ambiguous state are
  first-class results, so this path is expected to stop at the first unresolved
  use rather than borrowing Eq. (8) initialization or silently reordering lines;
- `paper_eq7_score_greedy` implements Eq. (8) using exactly Eq. (7),
  \(w_x+\sum_{j\in S}a_{xj}\), without calling an objective marginal; and
- `true_marginal_greedy` computes
  \(F(S\cup\{x\})-F(S)\) from the selected objective's independent evaluator.
  For `paper_samplewise_literal`, this is
  \(w_x+2\sum_{j\in S}a_{xj}\).

No path may call another. The optimum implementation enumerates all size-\(b\)
subsets directly. Each proposed executable resolution of Algorithm 1 receives
its own identifier, records every departure from a literal line, and is never
reported as `paper_algorithm1_literal`. Deterministic tie handling evaluates
all tied paths for the two equation-defined executable greedy implementations;
the evidence reports best, worst, and canonical lexicographic values
separately. This prevents an arbitrary tie break, the Algorithm 1 ambiguities,
or the Eq. (7)/true-marginal mismatch from being hidden.

The true-marginal path must also run with
`appendix_inline_shift_literal`, using
\(w_x+2\sum_{j\in S}a_{xj}+\alpha\eta(2|S|+1)\). At each fixed iteration its
cardinality term is equal for every candidate, and at fixed budget \(b\) its
\(\alpha\eta b^2\) objective term is equal for every feasible set. Those facts
may establish equality of selected sets and optima with the unshifted literal
variant, but they do not establish submodularity or transfer a multiplicative
approximation ratio: an additive cardinality term changes objective ratios.
The greedy-guarantee audit therefore records the two-element `1`-then-`3`
witness as a failed theorem premise and must not report the repaired standard
greedy guarantee as a guarantee for this literal Appendix E variant.

The theorem-premise and ratio audit has six set-function variants:
`paper_mwcp`, `paper_samplewise_literal`, `single_counted_pairwise`,
`half_corrected_samplewise`, `appendix_inline_shift_literal`, and
`modular_shift_candidate`. `appendix_eq26_score` is not a set function and
receives one explicit `not_applicable` premise record with zero finite-domain
work. Every canonical graph fixes
\(M=\max_{\{i,j\}}|a_{ij}|\), with \(M=0\) when the graph has no edges, before
any objective or outcome is evaluated. The one canonical parameter function
for the symmetric diminishing-returns, premise, greedy, optimum, and finite
proof-ledger domains is:

| set-function variant | `alpha` | `eta` |
| --- | ---: | ---: |
| `paper_mwcp` | \(1\) | \(0\) |
| `paper_samplewise_literal` | \(1\) | \(0\) |
| `single_counted_pairwise` | \(1\) | \(0\) |
| `half_corrected_samplewise` | \(1\) | \(0\) |
| `appendix_inline_shift_literal` | \(1\) | \(M\) |
| `modular_shift_candidate` | \(1\) | \(2(n-1)M\) |

Thus the modular objective remains exactly
\(f_{\mathrm{lit}}(S)+\eta_{\mathrm{mod}}|S|\), with
\(\eta_{\mathrm{mod}}=2(n-1)M\), and never switches to a single-counted base.
The parameter function creates no enumeration axis and therefore changes no
case count or ceiling.

The graph ID is exactly
`n=<n>;vw=<w0>,...,<w(n-1)>;ew=<a01>,<a02>,...`, where vertices are canonical
indices, every value is a normalized `p/q`, unordered edges are in
lexicographic pair order, and an empty edge vector is `ew=-`. The symmetric
diminishing-returns control records the all-zero vertex vector in that ID even
though its symbolic proof cancels vertex terms. Every parameterized instance
ID is then exactly
`graph=<graph-id>::variant=<variant>::alpha=1/1::eta=<p>/<q>`, where the final
`<p>/<q>` is the normalized nonnegative rational returned by the table.
Budget, subset, marginal, greedy-path, or conclusion suffixes may extend that
base ID, but may not replace it. Different parameter tuples may never share an
ID. The separately required minimal Appendix falsification is a fixed
symbolic diagnostic, outside the canonical per-graph finite enumeration: its
ID includes `alpha=1/1::eta=1/1::diagnostic=appendix-minimal`, so its
zero-edge, \(1\)-then-\(3\) result is preserved without selecting a parameter
after observing an outcome.

Premises are budget-independent, so for each of the
\[
G=\sum_{n=1}^{4}3^n2^{\binom n2}=5{,}421
\]
weighted graphs, each of the six set-function variants is checked once over
the entire power set before any theorem ratio is classified:

- global non-negativity: \(F(S)\geq0\) for every \(S\subseteq V\);
- normalization: \(F(\varnothing)=0\);
- global monotonicity: \(\Delta(x\mid S)\geq0\) for every
  \(S\subseteq V\) and \(x\notin S\); and
- global submodularity: \(\Delta(x\mid A)\geq\Delta(x\mid B)\) for every
  \(A\subseteq B\subseteq V\) and \(x\notin B\).

The result stores all four booleans and canonical failing witness IDs. Only an
instance for which all four are true is `theorem_eligible`. For each eligible
instance it records

\[
\rho =
\begin{cases}
F(S_{\mathrm{greedy}})/F(S^\star),&F(S^\star)>0,\\
1,&F(S_{\mathrm{greedy}})=F(S^\star)=0,
\end{cases}
\]

and compares the exact rational value to \(1-1/e\) through an exact certified
inequality, never floating point. `guarantee_violations` may contain only
eligible instances and must include their complete premise certificate.
Negative, undefined, or poor ratios for ineligible instances are stored only
under `out_of_premise_diagnostics`, with failed-premise witness IDs, and are
never counted or described as theorem counterexamples. If the approved finite
domain contains no eligible instance, the guarantee status is
`not_evaluated`, not supported or contradicted. It searches smallest-first for:

- a violation of the claimed \(1-1/e\) bound under the paper's exact premises;
- a symbolic cardinality witness showing the paper's \((b-t)\) step is
  unavailable without an additional nesting premise;
- a mismatch between Eq. (7)'s score and the literal Eq. (4) marginal; and
- failure of the submodularity premise for the literal Appendix E shifted
  objective, including the minimal two-element witness.

The labeled **exhaustive greedy domain** uses \(1\leq n\leq4\),
\(1\leq b\leq\min(3,n)\), vertex weights in \(\{0,1,2\}\), and symmetric edge
weights in \(\{-1,0\}\). It contains exactly
\(\sum_{n=1}^4\min(3,n)3^n2^{\binom n2}=16{,}239\) weighted-cardinality
instances. Each optimum enumerates at most six selected sets, and each all-ties
greedy traversal has at most \(P(4,3)=24\) terminal paths. Exact
domain-sensitive ceilings replace those loose maxima:

- objective values needed for every optimum are
  \[
  O=\sum_n 3^n2^{\binom n2}
  \sum_{b=1}^{\min(3,n)}\binom nb=74{,}145
  \]
  per set-function variant;
- terminal paths per greedy selector are
  \[
  P=\sum_n 3^n2^{\binom n2}
  \sum_{b=1}^{\min(3,n)}\frac{n!}{(n-b)!}=210{,}675;
  \]
- candidate score/look-up operations per selector are
  \[
  C=\sum_n3^n2^{\binom n2}
  \sum_{b=1}^{\min(3,n)}
  \sum_{k=1}^{b}\frac{n!}{(n-k)!}=316{,}983.
  \]

The optimum table is computed once per variant and reused for both greedy
families' terminal objective values. The global premise marginal table is also
reused by true-marginal greedy; a cache lookup/selection comparison is counted
in \(C\), and no hidden marginal recomputation is permitted. Claims outside
this domain rely on symbolic proof-ledger reasoning, not an exhaustive label.
There is no undeclared 100-evaluation smoke allowance. Adding a future smoke
domain requires a new named accounting component and independent design review
before execution.

### Appendix-premise oracle

This oracle is a proof ledger rather than a numerical shortcut. Eq. (28) keeps
one numbered row for accounting but two separately statused conclusions as
defined above. Every Eq. (28)--(38) conclusion stores its exact prerequisites,
`blocked_by` links, and exact-rational check. A failed or unavailable
prerequisite makes the downstream conclusion `not_applicable`; it is not
automatically a contradiction. Finite rows for `modular_shift_candidate`
consume only the literal-base
\(f_{\mathrm{lit}}+\eta_{\mathrm{mod}}|S|\) objective and its
\(w_x+d_x(S)+\eta_{\mathrm{mod}}\) marginal defined above; the proof ledger
has no alternate single-counted modular interpretation. The dependency gates
are:

- Eq. (28a) requires its monotonicity/union and submodular telescoping steps.
- Eq. (28b) additionally requires nonnegative marginals and
  \(|S^\star\setminus S_t|\leq b-t\). A mandatory symbolic witness uses
  \(V=\{a,b,c\}\), \(b=2\), \(t=1\), \(S_t=\{a\}\), and
  \(S^\star=\{b,c\}\), so \(2=|S^\star\setminus S_t|>b-t=1\). It concerns
  cardinality only and is never weight-dependent or reported as an Eq. (28a)
  counterexample.
- Eq. (29) requires that the selected element be a true-marginal maximizer and
  that \(S_{t+1}\) be defined. Eq. (7) score selection does not satisfy this
  prerequisite merely by sharing a greedy label.
- Eq. (30) requires both Eq. (28) conclusions and Eq. (29); Eq. (31) requires
  the defined algebraic quantities; Eq. (32) requires Eq. (30) and
  \(b-t>0\); Eq. (33) requires Eq. (32) and its residual definition.
- Eq. (34) requires the Eq. (33) recurrence for every \(t=0,\ldots,b-1\) and
  exact product reindexing. Eq. (35) requires a well-defined product and
  positive integer \(b\).
- Eq. (36)'s conclusion is split from its logarithmic derivation. For \(b=1\),
  \((1-1/b)^b=0\leq1/e\) is supported, but the `ln(1-1/b)` step is
  `not_applicable` because \(\ln 0\) is outside its domain. For integer \(b>1\),
  the log derivation uses the exact symbolic lemma
  \(\ln(1-x)\leq-x\); \(b=0\) is `not_applicable`.
- Eq. (37) requires the complete Eq. (34)--(36) chain and the theorem
  premises; Eq. (38) requires Eq. (37), the exact objective relation, and a
  nonnegative optimum.

At minimum it also audits:

- normalization \(f(\varnothing)=0\);
- non-negativity and monotonicity, not submodularity alone;
- whether adding an objective constant preserves normalization and ratios;
- whether the Appendix E term \(\alpha\eta|S|^2\), which is not an objective
  constant across cardinalities, preserves diminishing returns and permits the
  standard greedy theorem;
- the bound on \(|S^\star\setminus S_t|\), which is at most \(b\) but need not
  be at most \(b-t\);
- the multiplication/product indices, including the \(k=1\) factor; and
- the logical relationship between a repaired standard greedy theorem and the
  theorem actually stated.

The proof-ledger schema is uniform. Every numbered row, including rows with
only one conclusion, has exactly:

```json
{
  "row_id": "eq28",
  "equation": "28",
  "model_variant": "paper_samplewise_literal",
  "instance_id": "symbolic-or-canonical-instance-id",
  "conclusions": []
}
```

Every member of `conclusions` has exactly `conclusion_id`, `statement`,
`required_premise_ids`, `prerequisite_conclusion_refs`, `check_id`,
`evidence_kind`, `status`, `blocked_by`, and `witness_ids`. Row objects never
carry a shadow `status`, premise list, blocker list, or witness list. Eq. (28)
has two conclusion members; Eq. (29)--(38) each have one, for 12 conclusions
per variant/instance. A conclusion reference is the canonical string
`<model_variant>/<instance_id>/<row_id>/<conclusion_id>`. Acceptance rebuilds
the exact acyclic prerequisite adjacency map and rejects missing, duplicate,
unknown, replaced, or cyclic references.

Conclusion statuses are `supported`, `contradicted`, or `not_applicable`. For
`appendix_inline_shift_literal`, the ledger must link the minimal two-element
`1`-then-`3` witness wherever Appendix F requires submodularity, distinguish
its supported normalization and any supported monotonicity facts, and reject
ratio transfer from a repaired objective. It must never substitute
`modular_shift_candidate` for this literal variant.

The arbitrary-set cardinality checks and algebraic transitions are labeled
**symbolic**. A separate **exhaustive finite proof-ledger control** reuses the
16,239 weighted-cardinality instances from the greedy domain rather than
opening another Cartesian product. It evaluates 12 conclusions for each of
the six set-function variants and each instance, for a ceiling of
\(12\cdot6\cdot16{,}239=1{,}169{,}208\) conclusion-status operations.
`appendix_eq26_score` receives 12 symbolic `not_applicable` conclusions but no
finite-instance expansion. The ledger records `not_applicable` rather than
manufacturing a numerical check when a symbolic step has no instance-level
predicate.

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

The runner increments a named counter for every primitive subset evaluation,
marginal/score evaluation, comparison, path record, summary classification,
or proof conclusion. A component may finish below its ceiling because actual
ties are fewer than the all-tie bound, but acceptance requires
`0 <= actual <= declared_ceiling`; equality is never required or fabricated.
The complete generation ceiling is:

| Component | Formula | Ceiling |
| --- | --- | ---: |
| objective-equivalence objective values | \(26\cdot2\) | 52 |
| symmetric diminishing-return primitives | \(79{,}480\cdot6\text{ exact set-function variants}\cdot(4\text{ subset}+2\text{ independently closed-form marginal})\) | 2,861,280 |
| asymmetric literal diagnostic primitives | \(19{,}738\cdot(4+2)\) | 118,428 |
| shift marginal/score values | \(6{,}459\cdot7\text{ channels}\) | 45,213 |
| rational-\(\alpha\) values | \(256\cdot7\text{ channels}\) | 1,792 |
| premise subset values | \(84{,}750\cdot6\) | 508,500 |
| premise marginal values | \(168{,}555\cdot6\) | 1,011,330 |
| premise submodularity comparisons | \(565{,}815\cdot6\) | 3,394,890 |
| Eq. (7) candidate scores | \(C\) | 316,983 |
| Eq. (7) terminal paths | \(P\) | 210,675 |
| true-marginal candidate lookups/comparisons | \(C\cdot6\) | 1,901,898 |
| true-marginal terminal paths | \(P\cdot6\) | 1,264,050 |
| optimum subset objective values | \(O\cdot6\) | 444,870 |
| best/worst/canonical classifications | \(16{,}239\cdot6\cdot2\cdot3\) | 584,604 |
| finite Appendix F conclusions | \(16{,}239\cdot6\cdot12\) | 1,169,208 |
| symbolic Appendix F conclusions | \(7\cdot12\) | 84 |
| literal Algorithm 1 audit | fixed | 1 |
| mandatory Appendix E witness marginals | fixed | 2 |
| **Generation ceiling** | | **13,833,860** |

The premise formulas are independently checked before execution:

\[
\sum_n3^n2^{\binom n2}2^n=84{,}750,\quad
\sum_n3^n2^{\binom n2}n2^{n-1}=168{,}555,\quad
\sum_n3^n2^{\binom n2}n3^{n-1}=565{,}815.
\]

The earlier `1_177_735` figure is withdrawn: it hid per-variant premise
subsets/marginals and used loose path maxima as though they were exact runtime
counts. It is neither an accepted total nor an equality target. Semantic
validation performs one full deterministic replay with the same
`13_833_860` ceiling, so controller generation plus replay has a declared
ceiling of `27_667_720` work units. Each pass records its own component actuals,
and both must remain at or below every component ceiling. All controls use
exact arithmetic, at most four vertices, and no GPU, network call, or model
training. Canonical `evidence.json`, canonical witness files, and canonical
command records contain no measured wall time, start/end timestamp, duration,
elapsed seconds, or host-clock value. They retain only deterministic domain
formulas, declared ceilings, actual work counts, return/completion facts, and
completion status. Every byte of every canonical evidence and witness file is
included in the two-run byte comparison; there is no normalization or excluded
runtime field.

The controller measures generation and replay externally. It records the
measurement only in its validation attestation or in a noncanonical log under
`/tmp`; that measurement is never copied into the submission, committed,
hashed into canonical evidence, rendered, uploaded, or used in canonical byte
comparison. The 30-minute limit is an operational controller gate: if either
generation or replay exceeds 1,800 measured seconds, the controller withholds
validation rather than shrinking a domain. The deterministic work ceilings
and completion facts remain in evidence regardless of that external timing
decision.

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
5. Appendix E tests fail before shift support exists: they require the literal
   \(f_{\mathrm{lit}}(S)+\alpha\eta|S|^2\) variant to remain distinct from all
   repaired shifts and require its zero-weight, \(\alpha=\eta=1\), two-element
   marginals to be exactly `1` then `3` in the diminishing-returns,
   greedy-guarantee, and proof-ledger outputs.
6. Greedy, exhaustive optimum, all-ties, and ratio tests fail before solvers
   exist. Adversarial cases require a poor-ratio instance with one failed
   global premise to remain only in `out_of_premise_diagnostics`, reject any
   ineligible record in `guarantee_violations`, and produce `not_evaluated`
   when no theorem-eligible instance exists.
7. Literal Algorithm 1 transcription, undefined-read, state-order, and
   no-silent-repair tests fail before its independent audit path exists.
8. Proof-ledger and minimal-witness persistence tests fail before Appendix F
   auditing exists. They require separately statused Eq. (28a)/(28b), the
   uniform nested-conclusion schema on every row, the symbolic \(b-t\)
   witness, exact acyclic prerequisite references, downstream blocking, and
   Eq. (36) cases for \(b=0\), \(b=1\), and \(b>1\).
9. Evidence-bundle acceptance tests fail before their producer exists. They
   delete, duplicate, and replace full-domain records; alter stored statuses,
   nested conclusions, and prerequisite edges; tamper excerpt paths, key sets,
   byte counts, hashes, witness IDs, witness bytes, and links; and remove or add
   witness files. All must be rejected by canonical full-domain replay even
   when the mutated bundle is internally self-consistent.
10. Renderer tests fail first on an invalid ID-as-array-index JSON pointer and
    require every emitted RFC 6901 pointer to resolve to the displayed value.
    Report, poster, and Space tests also require visible/downloadable
    `NOTICE.md`, MIT `LICENSE`, and CC BY-NC-SA legal text.

For every red phase, the log records command, timestamp, test identifier, and
the expected missing behavior. Tests assert mathematical identities and schema
invariants; they do not assert that the paper must pass or fail.

## Evidence bundle and schema

`submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/evidence/evidence.json`
is the canonical computed artifact. A proposed schema version `1` contains:

```json
{
  "schema_version": 1,
  "attempt_id": "64bfe193-333b-4b37-9683-9ac25ca5ac27",
  "source_revision": "40-hex final source-integration commit",
  "paper": {
    "challenge_id": "a3GdvuPItd",
    "revision": "arxiv:2606.12913v2",
    "source_url": "https://arxiv.org/pdf/2606.12913v2",
    "pdf_acquisition_command": "curl --fail --location --proto '=https' --tlsv1.2 --output /tmp/2606.12913v2.pdf https://export.arxiv.org/pdf/2606.12913v2 && test \"$(wc -c < /tmp/2606.12913v2.pdf)\" -eq 683737 && printf '%s  %s\\n' 26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090 /tmp/2606.12913v2.pdf | sha256sum --check --strict",
    "pdf_byte_count": 683737,
    "pdf_sha256": "26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090",
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
  "guarantee_violations": [],
  "out_of_premise_diagnostics": [],
  "proof_ledger": [],
  "claim_results": [],
  "unavailable_claims": [],
  "commands": [],
  "artifacts": []
}
```

Each transcription has the exact manifest keys defined above and repeats the
authenticated byte count/hash, not excerpt text.
Each search includes `oracle`, `model_variant`, `greedy_path` when applicable,
its exhaustive/symbolic label, exact domain, ceiling formula and value, actual
case count, completion status, the canonical parameter rule and realized
parameterized case-ID grammar, and `source_revision`. Each witness includes `id`,
`artifact_path`, `artifact_sha256`, `model_variant`, the audited property,
exact rational inputs as numerator/denominator strings, all intermediate
values, `universal_claim_falsified`, and `minimality_checks`. `artifact_path`
is a canonical relative path under `evidence/witnesses/`, cannot traverse
outside the evidence root, and is never inferred from untrusted input.
Each claim result includes a stable local claim ID, `target_claim` equal to one
of the two exact strings above, expected observation, computed observations,
witness links, status, and limitations. Every result belongs to exactly one
target. Submission and improvement verdict payloads are generated only from
these two strings, preserve their order and spelling byte-for-byte, and reject
missing, additional, or rewritten claim text before any coordinator mutation.

The human report is generated from this JSON and cannot introduce new numeric
claims. Arrays remain arrays; canonical references to them use numeric,
RFC 6901-escaped indices such as `/witnesses/0/id`, never pseudo-pointers such
as `/witnesses/{id}`. Rendering builds a deterministic ID-to-index lookup and
every `data-evidence-path` or report footnote must resolve with an RFC 6901
evaluator to the exact displayed value. JSON Schema validation, stable sorting,
deterministic serialization, and a second clean-run byte comparison are
required.

Schema acceptance is a full deterministic replay, not validation of predicates
selected by the candidate bundle. Given the evidence JSON path, project root,
and `source_revision`, it:

1. opens `paper_transcriptions/manifest.json`, requires the exact key set,
   exact expected record IDs, unique safe paths, and exact referenced-file set;
   reads every excerpt/Algorithm file as bytes; verifies byte count, SHA-256,
   UTF-8 decoding, the aggregate `TRANSCRIPTION_SET_SHA256`, and the reviewed
   normalized expression;
2. independently reconstructs the canonical graph, \(M\), six-row parameter
   tuple, parameterized base ID, budget, subset, marginal, greedy, and
   proof-conclusion domains from source constants rather than candidate
   `searches`, record IDs, parameters, counts, or prerequisite links;
3. reruns the entire `13_833_860`-ceiling generation into an isolated temporary
   root, producing expected canonical `evidence.json` and witness files;
4. compares canonical bytes and exact relative-file sets between expected and
   candidate evidence roots, then separately validates every RFC 6901 render
   pointer against the replayed evidence; and
5. rejects path traversal, missing/extra/duplicate/replaced domain records,
   missing/extra/tampered witness files, stale hashes, duplicate or
   cross-variant witness IDs, a missing or altered `alpha`, any canonical
   `eta` (zero unshifted, Appendix, or modular), parameterized case ID, and any
   missing, unknown, replaced, or cyclic proof prerequisite edge.

`guarantee_violations` and `out_of_premise_diagnostics` are required
top-level arrays, and the displayed schema object has an exact top-level key
set with no additional properties. Replay independently regenerates both from
the canonical premise certificates and exact ratio classifications, then
compares their complete canonical order and bytes. A missing, extra, reordered,
or misclassified entry fails acceptance, including moving an ineligible result
into `guarantee_violations` or removing its out-of-premise diagnostic.

The replay must not call a `validate_*` method on candidate records or derive
an expected domain from their contents. Tests delete one middle graph/subset
record, duplicate another record in its place while preserving array length,
replace a record under the original ID, alter Eq. (28)'s nested conclusion,
remove and redirect prerequisite edges, alter `alpha`, any table-derived
`eta`, or a parameterized ID, mutate or move an entry between the two top-level
classification arrays, insert a measured runtime field, mutate excerpt bytes
and manifest keys, and remove/replace witness files. Every mutation must fail
even when stored counts, statuses, and hashes are edited to remain internally
self-consistent.

Acceptance additionally requires one canonical symbolic-diagnostic
`appendix_inline_shift_literal` witness with two elements, zero vertex and edge
weights, `alpha=1`, `eta=1`, and exact marginal values `1` and `3`. Its stable
witness ID contains
`alpha=1/1::eta=1/1::diagnostic=appendix-minimal` and must be referenced by
the diminishing-returns result, the greedy-guarantee premise result, and every
applicable Appendix F proof-ledger row. Acceptance fails if any of those
records is absent, if a repaired variant uses that literal variant's
identifier, or if results for
`appendix_inline_shift_literal` and `modular_shift_candidate` are merged.

Revision provenance has two non-self-referential authorities:

- `source_revision` is embedded in canonical evidence and names the final
  commit containing all executable source, schemas, transcriptions, tests,
  application code, dependency files, and renderer templates.
- `artifact_revision` is the controller-observed current Git `HEAD` containing
  the generated evidence, witnesses, report, and poster. It is recorded in the
  validation attestation, not embedded in those files, because embedding a
  commit's own SHA would be self-referential.

The controller requires `source_revision` to be an ancestor of
`artifact_revision` and requires every path changed in
`source_revision..artifact_revision` to match only:
`evidence/evidence.json`, `evidence/witnesses/*.json`, `report.md`,
`poster.html`, or `poster_embed.html` within this submission. The range may
contain no source, schema, transcription, test, lockfile, application,
configuration, README, notice, or license change. It reruns the full replay at
`artifact_revision`; because executable paths are unchanged from
`source_revision`, the replayed bytes must equal the committed generated
artifacts. Any later executable or template change requires a new
`source_revision`, regeneration, and a new artifact-only range.

## Attribution and licensing

The parent convention is MIT for root-authored work (`LICENSE` and `README.md`)
and separate licensing for bundled material. The submission follows that
convention with explicit file boundaries:

- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/LICENSE`
  contains the MIT License and covers original software in `src/`, `tests/`,
  `app.py`, `pyproject.toml`, `uv.lock`, and original JSON Schema files.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/LICENSES/CC-BY-NC-SA-4.0.txt`
  contains the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International legal code. It covers adapted/transcribed material in
  `paper_transcriptions/`, `evidence/`, `README.md`, `poster.html`,
  `poster_embed.html`, and explanatory Space assets.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/NOTICE.md`
  names all seven authors, paper title, exact arXiv v2 URL, source license,
  adaptation status, both licenses, and the file-boundary map. The Space
  exposes the same notice.

`README.md`, `report.md`, the poster, and the Space each surface the attribution
and license boundary without requiring repository browsing. The Space provides
visible links and downloads for `NOTICE.md`, the MIT `LICENSE`, and
`LICENSES/CC-BY-NC-SA-4.0.txt`; acceptance rejects a deployment bundle missing
any of them.

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

The Space exposes downloadable canonical JSON and witness files, the three
notice/license files above, and a deterministic “recompute formal audit” path
suitable for CPU execution. It must not label repaired-theorem evidence as
verification of the literal paper claim.

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
  literal formulation failure; an out-of-premise failure remains a diagnostic,
  not a theorem counterexample.
- No released implementation is available to resolve whether code used
  single- or double-counted edges or a different monotonicity shift.
- Empirical accuracy and acceleration remain unavailable because the required
  code/artifacts and autonomous GPU budget are unavailable.

## Validation and completion gates

The implementation plan must require:

1. clean-environment installation from locked dependencies;
2. the complete deterministic evidence command twice with byte-identical
   canonical JSON and witness trees, component actuals no greater than the
   `13_833_860` generation ceiling, and no equality assumption;
3. JSON Schema validation plus full canonical-domain replay, byte comparison of
   the complete evidence/witness tree, excerpt-byte authentication,
   report-to-JSON pointer agreement, and acceptance of the required Appendix E
   literal witness and its three audit linkages;
4. the submission's full `python -m pytest` suite;
5. root `uv run python -m pytest -q`, while excluding the archival NAPE tree according
   to repository policy;
6. `uv run pre-commit run -a`;
7. credential, mutable-URL, cache, and unrelated-diff review;
8. `superpowers:verification-before-completion`;
9. integration of all source changes as `source_revision`, then regeneration
   and an artifact-only descendant commit observed by the controller as
   `artifact_revision`; every intervening path must match the generated
   artifact allowlist, and any later executable/template change requires a new
   source revision and regeneration;
10. delivery of the worker branch and evidence as an untrusted proposal. The
    worker does not mutate coordinator state or any external service;
11. controller revalidation followed by immutable `attest-validation`;
12. controller deployment to a paper-specific Space, exact deployed-SHA/live
    verification, and `publish-deployment`;
13. controller-only fresh assessed live refresh immediately before submission,
    cancellation on drift, exact-claim submission, and `attest-submission`;
14. controller-only post-submission refresh and queued/live presence proof,
    followed by bounded `watch-attempt`; and
15. controller-only official verdict import through `sync-verdict`. Each
    controller phase uses the exact attempt ID, current owner, and fencing
    token, and persists the corresponding HANDOFF milestone. Workers never
    deploy, submit, poll, import verdicts, or claim those phases.

This design-author task stops with an uncommitted two-document proposal for
independent review. It does not implement the submission, record or approve the
design in coordinator state, commit, update
`docs/HANDOFF.md`, deploy a Space, submit to the challenge, or touch NAPE.

## Design self-review checklist

- No placeholder, TODO, or unresolved design choice remains.
- Literal, conventional, and repaired objectives have distinct identifiers.
- `modular_shift_candidate` has one literal-base
  \(f_{\mathrm{lit}}+\eta_{\mathrm{mod}}|S|\) definition across objective,
  marginal, shift, greedy, optimum, premise, proof-ledger, and accounting
  paths, with greedy/proof \(\eta_{\mathrm{mod}}=2(n-1)M\).
- All six canonical graph variants derive `alpha=1` and their exact `eta`
  (`0`, \(M\), or \(2(n-1)M\)) from one pre-outcome table, encode normalized
  fractions in every case ID, and add no enumeration dimension.
- All six charged set-function variants have independent closed-form marginal
  formulas and direct-versus-closed formula-specific regression tests.
- The Appendix E shifted literal objective has the exact minimal two-element
  `1`-then-`3` falsification linked across diminishing returns, greedy-guarantee,
  and proof-ledger audits without being conflated with a repaired shift.
- Literal Algorithm 1, Eq. (7)--(8) score greedy, and true-marginal greedy have
  distinct identifiers, with ambiguities preserved rather than repaired.
- Each universal claim has an independent oracle and a falsification path.
- Out-of-premise diagnostics are separate from guarantee violations and are
  never universal counterexamples; both are required top-level arrays and are
  regenerated by replay.
- Measured time is controller-only and noncanonical; deterministic work and
  completion facts remain in evidence, whose complete bytes are compared
  without exclusions or normalization.
- Six set-function variants have fixed parameters and fully counted subset,
  marginal, path, classification, and proof-conclusion ceilings; actual work is
  bounded above rather than required to equal a ceiling.
- Semantic acceptance reconstructs canonical domains and byte-compares a full
  replay, rather than trusting candidate membership or prerequisite graphs.
- Every proof row uses the same nested-conclusion schema.
- Excerpt bytes have unique authenticated paths and an aggregate transcription
  pin.
- `source_revision` and controller-observed `artifact_revision` are distinct,
  with only generated artifacts allowed between them.
- Minimal witnesses and complete search domains are preserved.
- Paper context, computed outputs, and unavailable evidence are distinct.
- Licensing, poster, Space, TDD, evidence schema, and validation are explicit.
- The scope is one CPU-bounded formal-evidence submission and is suitable for
  a single later implementation plan.
