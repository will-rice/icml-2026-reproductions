# Claim 2: Robustness to score-estimation error


---
<!-- trackio-cell
{"type": "code", "id": "cell_c80a911f97a0", "created_at": "2026-07-22T13:59:00+00:00", "title": "Deterministic evidence rerun", "command": ["uv", "run", "python", "-m", "diffusion_gmm_repro.cli", "--output-dir", "evidence"], "exit_code": 0, "duration_s": 1.156}
-->
````bash
$ uv run python -m diffusion_gmm_repro.cli --output-dir evidence
````

exit 0 · 1.2s


````output

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_463d979e49a7", "created_at": "2026-07-22T13:59:32+00:00", "title": "Setup and result"}
-->
### Setup

Toy scope: one Gaussian step with a constant, norm-calibrated additive score perturbation. Error norms 0, 0.05, 0.10, and 0.25 are evaluated at step size 0.125; a state-dependent cubic perturbation is recorded separately as an assumption-breaking control.

### Result

Computed analytic mean shifts are 0, 0.00625, 0.0125, and 0.03125: monotone and exactly proportional to the perturbation norm. The misspecified-score control has empirical second moment 1.393 and is not used as supporting evidence. **Status: supports separability/monotonicity within this toy scope; full time-averaged score-error robustness is unreplicated.**
