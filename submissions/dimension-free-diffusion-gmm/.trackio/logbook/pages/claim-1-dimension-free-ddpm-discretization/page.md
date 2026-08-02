# Claim 1: Dimension-free DDPM discretization


---
<!-- trackio-cell
{"type": "code", "id": "cell_8eb7b7718971", "created_at": "2026-07-22T13:58:59+00:00", "title": "Deterministic evidence rerun", "command": ["uv", "run", "python", "-m", "diffusion_gmm_repro.cli", "--output-dir", "evidence"], "exit_code": 0, "duration_s": 1.09}
-->
````bash
$ uv run python -m diffusion_gmm_repro.cli --output-dir evidence
````

exit 0 · 1.1s


````output

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_91b75903e060", "created_at": "2026-07-22T13:59:31+00:00", "title": "Setup and result"}
-->
### Setup

Toy scope: one Euler/DDPM-like step at a stationary standard Gaussian with exact score `-x`, step size 0.125, seed 2026, 2,048 samples, and dimensions 1, 4, 16, 64. The analytic identity is `Var(x prime)=1+h squared`; empirical moments are diagnostics only.

### Result

The computed analytic per-coordinate variance error is 0.015625 at every dimension, giving range 0 against the threshold 1e-12. **Status: supports within this toy scope.** This does not reproduce the paper’s complete multi-step total-variation guarantee. Machine-readable records are in `evidence/results.json` and `evidence/measurements.csv`; paper context is pinned to [arXiv v1](https://arxiv.org/abs/2504.05300v1).
