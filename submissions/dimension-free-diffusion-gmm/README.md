# Dimension-Free Diffusion/GMM Numerical Audit

This independent, deterministic CPU audit targets three claims from
`arxiv:2504.05300v1`. It does not replace the paper's proofs, and it never
presents paper-reported values as reproduced measurements. The audit uses
analytic isotropic-Gaussian/GMM identities, seeded numerical diagnostics, and
controls that violate the stated variance-floor assumption.

## Reproduce

From this directory:

```bash
uv sync --frozen
uv run pytest -q
uv run python -m diffusion_gmm_repro.cli --output-dir evidence
```

The command atomically writes `evidence/results.json` and
`evidence/measurements.csv`. Input identity and acquisition metadata are in
`evidence/provenance.json`.

The three exact claim identifiers are:

- `dimension-free-ddpm-discretization`
- `robustness-to-score-error`
- `dimension-free-score-jacobian-trace`

All result statuses describe these finite numerical audits only.
