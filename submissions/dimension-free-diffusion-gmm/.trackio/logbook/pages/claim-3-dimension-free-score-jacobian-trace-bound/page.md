# Claim 3: Dimension-free score-Jacobian trace bound


---
<!-- trackio-cell
{"type": "code", "id": "cell_4522d7bdedf4", "created_at": "2026-07-22T13:59:02+00:00", "title": "Deterministic evidence rerun", "command": ["uv", "run", "python", "-m", "diffusion_gmm_repro.cli", "--output-dir", "evidence"], "exit_code": 0, "duration_s": 0.985}
-->
````bash
$ uv run python -m diffusion_gmm_repro.cli --output-dir evidence
````

exit 0 · 1.0s


````output

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2330861410c4", "created_at": "2026-07-22T13:59:32+00:00", "title": "Setup and result"}
-->
### Setup

A symmetric two-component, unit-covariance GMM occupies one fixed active coordinate and is embedded in dimensions 1, 4, 16, and 64. The code evaluates the paper-aligned quantity `tr(I+J)` at 2,048 deterministic points and checks the analytic trace by centered finite differences. A component variance of 0.05 is the unit-covariance violation control.

### Result

The baseline maximum `tr(I+J)` is 3.9999389 at every tested dimension; its range is 3.55e-15 and maximum finite-difference disagreement is 9.21e-10. The covariance-violation control becomes strongly dimension-dependent. **Status: supports the fixed-subspace finite GMM trace behavior; the paper’s high-probability theorem remains unreplicated.**
