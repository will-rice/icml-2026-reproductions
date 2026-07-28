# Mechanistic Data Attribution Score Design

## Objective

Correct and complete the existing proposal for attempt
`3a44d506-d7a0-4bb8-abf7-d51a55c0018c`, paper `PQaxfoEcRc`, against arXiv
`2601.21996v2` and upstream commit
`faa0890bc2d7961a0f177a422849b4e0801943c0`.

The three targets are sample-level influence, targeted-versus-random causal
intervention, and concentration of influential samples in repetitive
structural domains. Larger-model generalization and downstream ICL claims are
explicitly unreplicated.

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

The current uncommitted worktree is a proposal, not authority. Workers must
first snapshot its diff, retain only claim-relevant changes, and ensure every
reported status is `toy`, `verified`, `falsified`, or `unreplicated` according
to the actual computation.

## Evidence and Failure Semantics

All intervention comparisons share model initialization, training steps, and
sample counts. Random controls use a fixed seed list and are fully serialized.
Pattern labels cannot call attribution functions or use influence ranks.
Outputs report raw per-seed rows, aggregate confidence intervals, input hashes,
commands, and limitations.

Tests fail on circular labels, missing retraining, mismatched controls,
nondeterminism, unsupported “verified” labels, or absent root `pages/*.md`.
Counterexamples and null effects remain valid evidence and are not hidden.

## Validation

Controller validation uses a clean worktree, repeats evidence generation,
runs project and workspace tests, validates the skill, runs pre-commit, and
compares hashes. Deployment is to a dedicated Space from the exact attested
tree. No GPU, paid API, paper-reported measurement, or unsafe external code is
used.
