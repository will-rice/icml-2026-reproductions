# RACO Score Reproduction Design

## Objective

Produce independently executable CPU evidence for attempt
`97e213a5-7ca3-4a1b-a500-1ec52d94d87a` and paper `vSzRJyg6k0`,
using arXiv `2602.02495v3` and upstream commit
`84a943c34f38520c7e0c9dd3066517c111b3c8fa`.

The evidence targets the four live challenge claims covering objective-specific
pairwise losses, CAGrad-Clip, Theorem 3.1, and Theorem 3.2. GPU-scale TL;DR,
BeaverTails, and model-family comparisons remain explicitly unreplicated.

## Selected Approach

Replace the existing local toy update with a source-faithful, dependency-light
implementation of the exact upstream two-objective CAGrad solver and
independent coefficient clipping. Add deterministic scalar and vector
objective families that check the update equations and theorem preconditions
over finite parameter grids. This is preferable to either wrapping the
training stack, which would pull in GPU-only work, or merely auditing source,
which would leave theorem claims untested.

## Components

- `src/reward_free_alignment/pairwise.py`: numerically stable, per-objective
  DPO-style pairwise losses and gradients without a reward model.
- `src/reward_free_alignment/cagrad_clip.py`: exact two-objective CAGrad
  coefficient solution, clipping, and diagnostic return values.
- `src/reward_free_alignment/theorem_audit.py`: deterministic smooth
  objectives, stationarity residuals, Pareto-criticality checks, and clipped
  versus unclipped convergence sweeps.
- `src/reward_free_alignment/generate_evidence.py`: one command that writes
  measured JSON, CSV, provenance, and a root judge page.

Inputs are limited to the pinned paper source and upstream repository blobs.
Every derived record names its formula, parameters, tolerance, and outcome.
Paper-reported empirical values are never copied into reproduced outputs.

## Evidence and Failure Semantics

The command fails if source hashes drift, any theorem precondition is omitted,
any numerical residual exceeds its declared tolerance, output schemas are
invalid, or a rerun changes deterministic output bytes. A theorem audit may
report a counterexample; such an outcome is evidence, not a test failure, when
the implementation and preconditions are valid.

The root `pages/reproduction.md` distinguishes verified, falsified, toy, and
unreplicated claim paths and exceeds the judge's direct-page threshold.

## Validation

Tests are written red-first for each public interface. Controller validation
runs evidence generation twice, the project pytest suite, the workspace test
suite, skill validation, pre-commit, and hash comparison. Deployment uses a
dedicated Space and the exact controller-attested source tree. No GPU, paid
API, inherited Hub credential, or external mutation is part of evidence
generation.
