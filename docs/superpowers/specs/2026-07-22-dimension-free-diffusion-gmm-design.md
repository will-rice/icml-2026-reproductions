# Dimension-Free Diffusion/GMM Reproduction Design

## Scope

Reproduce three claims from ICML 2026 submission 2493, pinned to
`arxiv:2504.05300v1`, as independent CPU numerical audits:

1. DDPM discretization error is effectively independent of ambient dimension
   for isotropic Gaussian-mixture targets when other parameters are fixed.
2. Controlled score-estimation error contributes a separable degradation term.
3. The Gaussian-mixture score-Jacobian trace quantity remains controlled by
   mixture/time complexity rather than ambient dimension.

These experiments audit finite instances; they do not replace the proofs.

## Architecture

The submission is an isolated Python project under
`submissions/dimension-free-diffusion-gmm/`. A small NumPy implementation will
define isotropic GMM density, score, score Jacobian, the paper's DDPM schedule,
and deterministic Monte Carlo diagnostics. A single CLI will run all audits and
write JSON plus CSV evidence. Tests will cover analytic identities, deterministic
execution, claim thresholds, provenance, and failure controls before each
implementation increment.

## Evidence design

- Dimension audit: sweep dimensions while holding mixture geometry in a fixed
  low-dimensional subspace; estimate a seeded distribution discrepancy across
  DDPM step counts and compare the fitted error curves.
- Score-error audit: add seeded, norm-calibrated score perturbations and verify
  monotone/separable degradation relative to the exact-score baseline.
- Jacobian audit: compare the analytic trace with finite differences across
  dimensions and mixture counts.
- Controls: use anisotropic covariance and deliberately misspecified scores to
  demonstrate degradation when theorem conditions are relaxed.

Every result records seeds, parameters, tolerances, source URLs, the pinned
paper revision, software versions, and measured values. Paper statements remain
context fields and are never copied into measured-output fields.

## Logbook and deployment

The Trackio logbook uses exactly: Index, Executive summary, one page per target
claim, and Conclusion. The executive summary and strict-polish poster are pinned
in that order. The evidence bundle is linked from claim pages. The dedicated
Space is validated, published, and verified by exact deployed SHA before
submission. No GPU Job or paid API is used; estimated API cost is USD 0.00.

## Error handling and acceptance

The CLI rejects nonpositive dimensions, component counts, steps, samples, or
covariance scales; it never emits partial evidence on failure. Acceptance
requires deterministic reruns, passing submission/root pytest suites, passing
pre-commit, schema-valid evidence, and agreement between JSON summaries and CSV
rows. Unsupported outcomes are reported as inconclusive or unavailable.
