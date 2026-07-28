# Mechanistic Data Attribution Score Design

## Objective

Correct and complete the existing proposal for attempt
`3a44d506-d7a0-4bb8-abf7-d51a55c0018c`, paper `PQaxfoEcRc`, against arXiv
`2601.21996v2` and upstream commit
`faa0890bc2d7961a0f177a422849b4e0801943c0`.
The immutable admission snapshot is
`09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199`.

The three targets are sample-level influence, targeted-versus-random causal
intervention, and concentration of influential samples in repetitive
structural domains. Larger-model generalization and downstream ICL claims are
explicitly unreplicated.

## Immutable Claim Binding

The evidence bundle preserves all five live claims in this exact order. The
first three are executable toy-scale targets; the final two are included as
`unreplicated` limitations rather than omitted.

1. `e470ef476fc89a065b017210926164b7846e3e594eb9ff9eaa499ec49d446152` —
   “Mechanistic Data Attribution quantifies individual training-sample
   influence on targeted interpretable LLM units such as induction and
   previous-token heads (Figure 1, Table 3).”
2. `539cd0a95b83e0bd75e6133c439813d5f88ada9dbfe34d9816cc29dcb7e53aae`
   — “Targeted deletion or augmentation of high-influence samples causally
   modulates induction-head and previous-token-head emergence more than random
   interventions (Figure 2).”
3. `29c953f86480ef5d8b883b872f43314dad58713830d6e095b81ce33612949df4`
   — “High-influence samples for induction heads are concentrated in
   repetitive structural domains, with top-ranked examples including LaTeX,
   HTML, and repeated text patterns (Table 1, Figure 4).”
4. `da0162afe1e0066da413c79d64446196ceb6698edb5527a594dbb774ef4a642d`
   — “Synthetic data patterns selected from Pythia-14M attribution generalize
   to larger model scales when augmenting induction-head formation (Table 2).”
5. `a7efd796a097ed339af48c027b5e9baa7a07ac7ef460c9c424cd92a1d394a3d0`
   — “Interventions that alter induction-head strength also shift in-context
   learning scores, supporting a causal link between the mechanism and ICL
   behavior (Figure 5).”

Tests bind the strings and hashes to immutable constants and recompute every
SHA-256. A different string, order, hash, snapshot, paper ID, or upstream pin
fails closed.

## Selected Approach

Preserve only the existing exact scalar-Hessian influence fixture, then replace
the circular category and score-subtraction demonstrations with two independent
experiments:

1. Train a deterministic tiny attention model on a controlled corpus; compare
   actual retraining after attribution-ranked removal/augmentation with matched
   random interventions.
2. Label structural categories before computing influence, then evaluate
   enrichment among top-ranked samples with permutation confidence intervals.

This provides causal and pattern evidence without pretending that a tiny model
reproduces Pythia-scale magnitudes. It is preferable to static upstream-code
inspection and remains CPU-feasible unlike the released Pythia workflow.

## Components

- `corpus.py`: seeded token corpora and labels assigned independently of model
  gradients.
- `tiny_model.py`: small attention model with deterministic initialization,
  training, and induction/previous-token probes.
- `attribution.py`: exact influence scores and rank output with no category
  inputs.
- `intervention.py`: full retraining from identical initialization for
  targeted and random controls.
- `patterns.py`: enrichment, effect size, and permutation interval from
  predeclared labels.
- `cli.py`: evidence/provenance/measurement generation and direct root page.

The experiment is fixed before execution: vocabulary 64, sequence length 16,
four categories with 64 samples each, a one-layer/two-head attention model
with model width 32 and feed-forward width 64, AdamW at learning rate `1e-3`,
batch size 16, and 160 steps. Seeds are `[11, 23, 37, 53, 71]`. Each probe is
evaluated on 256 sequences from the disjoint seed `training_seed + 10_000`.
For the previous-token probe, 128 positives are sampled uniformly from
vocabulary 64 and then receive four uniformly selected nonadjacent copied
predecessors; 128 negatives are iid-uniform sequences rejected if any adjacent
tokens match. For the induction probe, 128 positives have an iid-uniform
distinct-token first half copied exactly into positions 8–15; 128 negatives
are iid-uniform and rejected if their halves match. Held-out token hashes must
not equal a training token hash.
Previous-token accuracy is the fraction of positions 1–15 where head 0's
attention argmax is exactly `t-1`. Induction-copy accuracy uses period-eight
sequences with independently drawn distinct first halves and is the fraction
of positions 8–15 where head 1's attention argmax is exactly `t-8`. These two
predeclared alignments define toy-scale head emergence; no post-hoc probe is
allowed.

Influence freezes the trained transformer and fits two 32-dimensional
float64 logistic readouts. For head 0, its value slice is projected through
its slice of `W_O` and averaged over positions 1–15; for head 1 the analogous
32-vector is averaged over positions 8–15. Binary targets are exactly the two
probe constructors' positive/negative labels above, not semantic category
labels. With the
64-vector concatenated readout parameter `theta`, per-sample mean binary
cross-entropy `ell_i`, training Hessian
`H = grad²_theta mean_i ell_i`, and held-out probe-loss gradient
`g = grad_theta L_probe`, where `L_probe` is the equally weighted mean BCE
over both 256-example held-out probe sets, the score is
`I_i = -gᵀ(H + 1e-3 I)⁻¹ grad_theta ell_i / n`. The only solve is the exact
dense float64 `torch.linalg.solve`; labels and categories are unavailable to
this API.
Targeted removal and augmentation each affect the top 32 samples (12.5%);
matched random controls use the identical count, initialization, batching,
optimizer, and step schedule.

Claim 2 is `supported` only if, for both probes, targeted removal lowers
accuracy at least 0.02 more than matched random removal and targeted
augmentation raises it at least 0.02 more than matched random augmentation,
with the lower endpoint of a fixed-seed 10,000-resample paired bootstrap 95%
interval above zero in all four comparisons. A valid null or reversed effect
is `not-supported`; failed convergence or insufficient valid seeds is
`limited`. Raw seed pairs and bootstrap indices are serialized.

Pattern enrichment uses the same predeclared 12.5% cutoff, 2,000 seeded
permutations, odds ratio as effect size, and a predeclared positive result only
when the odds ratio exceeds 1 and the 95% permutation interval excludes 1.
Every seed and raw permutation summary is retained. No threshold is adjusted
after seeing results.

The current uncommitted worktree is a proposal, not authority. Workers retain
only claim-relevant changes. Local evidence uses `supported`, `not-supported`,
`limited`, or `unreplicated`; `verified`, `falsified`, and `toy` are reserved
for the official challenge verdict.

## Provenance and Judge Surface

Acquisition writes a committed manifest for the exact arXiv v2 bytes and every
used file at Git commit `faa0890...`, including URL, size, SHA-256, and Git blob
where applicable. Every offline command rechecks that manifest before reading
an input and fails on missing, extra, or changed bytes.

The Space root exposes `pages/reproduction.md` through both `app.py` and
`poster.html`; neither recomputes results. README metadata pins Gradio, names
`app.py`, and includes `paper-PQaxfoEcRc` and `icml2026-repro`. An offline
judge-view test imports the app with networking disabled and checks that both
root surfaces contain the claim table, limitations, commands, and at least 200
substantive characters.

## Evidence and Failure Semantics

All intervention comparisons share model initialization, training steps, and
sample counts. Random controls use a fixed seed list and are fully serialized.
Pattern labels cannot call attribution functions or use influence ranks.
Outputs report raw per-seed rows, aggregate confidence intervals, input hashes,
commands, and limitations.

Tests fail on circular labels, missing retraining, mismatched controls,
nondeterminism, reserved official-verdict labels, or absent root `pages/*.md`.
Matched-control tests cover removal and augmentation and compare sample counts,
initialization, exact ordered batch-index schedule, optimizer hyperparameters,
step count, and exact ordered training sample IDs with multiplicity through
one canonical training fingerprint.
Counterexamples and null effects remain valid evidence and are not hidden.

Two independent evidence builds are written to separate directories and
compared byte-for-byte. JSON/CSV ordering is canonical. The tarball is built
with sorted member paths and fixed mode, uid, gid, uname, gname, and mtime, so
archive equality is meaningful rather than merely printing two hashes.

## Validation

Controller validation uses a clean worktree, repeats evidence generation,
runs project and workspace tests, validates the skill, runs pre-commit, and
compares hashes. Deployment is to a dedicated Space from the exact attested
tree. No GPU, paid API, paper-reported measurement, or unsafe external code is
used.
