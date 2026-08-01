# Rex Evidence For ICML 2026 Judge

Paper: `7pQIzVNctu`
Attempt: `11b90d4c-61f2-4d93-949e-8d4618aca972`
Validated upstream revision: `arxiv:2502.08834+github:zblasingame/Rex-solver@e39b57415d5608b18d7c5631595f1d38f06813b8`
Results digest: `dcbccca6aff60ef6edc6b60f001f826357f50342991307595ddda33f8b441683`

Every number on this page is recomputed deterministically on CPU by `evidence.py` in this Space (regenerate: `python evidence.py`; full records in `evidence/results.json`). Nothing is copied from the paper. The GPU image-generation and Boltzmann sampling claims are outside this selected evidence target.

## Claim 1: verified — exact reversibility of the Rex coupling

Rex converts explicit Runge-Kutta and stochastic Runge-Kutta schemes into algebraically reversible exponential solvers for diffusion ODEs and SDEs (Section 3).

Challenge claim SHA-256: `06ee77e870a2c0447848e1f6159454496f17d144d02bc08fe44441f6b7ad332f`

Forward sweeps through the released reciprocal coupling, then backward replay, recover the initial state exactly (tolerance 1e-12):

| coupling | steps | round-trip max abs error |
| --- | --- | --- |
| 0.9 | 8 | 4.219e-15 |
| 0.9 | 64 | 8.376e-13 |
| 0.93 | 8 | 1.554e-15 |
| 0.93 | 64 | 2.112e-13 |
| 0.999 | 8 | 1.332e-15 |
| 0.999 | 64 | 3.331e-15 |

The stochastic path is reversible as well: a frozen-noise Euler-Maruyama increment paired through the same coupling round-trips with max abs error 4.441e-15 over 32 steps.

## Claim 2: verified — order inheritance measured

The ODE Rex construction inherits arbitrary order of convergence and a non-zero linear stability region from the base McCallum-Foster method (Theorem A.1).

Challenge claim SHA-256: `69eedf49ae10686f77613801c126d0825e1a2ea7198e4d9f31c945e00670b8e0`

Rex sweeps built from base increments of orders 1, 2, and 4 converge to the fine-step limit (RK4 base, 8192 steps) at the base method's order — the coupling preserves convergence order:

| base increment | theoretical order | errors at steps [8, 16, 32, 64] | measured rate |
| --- | --- | --- | --- |
| euler | 1 | 4.381e-03, 2.179e-03, 1.074e-03, 5.210e-04 | 1.04 |
| exp_midpoint | 2 | 2.375e-05, 5.958e-06, 1.492e-06, 3.733e-07 | 2.00 |
| exp_rk4 | 4 | 2.859e-10, 1.778e-11, 1.110e-12, 7.057e-14 | 3.98 |

Independent scalar checks: Euler errors 1.525e-01, 8.035e-02, 4.129e-02, 2.094e-02 (rate 0.98), RK4 errors 4.984e-06, 3.281e-07, 2.105e-08, 1.333e-09 (rate 3.98). The measured RK4 negative-real-axis stability radius lower bound is 2.7852 (theory: 2.7853), confirming a non-zero linear stability region.

## Claim 3: verified — reversible adaptive stepping demonstrated

The ODE Rex construction inherits arbitrary convergence order and supports reversible adaptive step-size solvers (Section 3.3).

Challenge claim SHA-256: `be5532066024dda765f5b69ee4444b86c339c6adc9beeedeb4c995b2e61d0f13`

An embedded Heun/Euler estimator adaptively selects step sizes inside the Rex coupling; replaying the accepted step sequence backward recovers the initial state exactly:

- error tolerance: 1e-06
- accepted steps: 185
- step-size range: 3.933e-03 to 6.250e-03
- round-trip max abs error: 3.109e-15

The pinned canonical wrapper defaults to RK4 (fixed step) and DOPRI5 (adaptive), and rejects adaptive use for tableaus without embedded error coefficients (embedded set: bogacki_shampine, dopri5, fehlberg45, tsit5).

## Claim 4: verified — reversible DDIM recovered numerically

Rex is shown to recover reversible versions of diffusion-model solvers including DDIM, DPM-Solver, and SEEDS-1 (Section 3.3).

Challenge claim SHA-256: `311e73b22c834fd47107a52f11e90aa005067fa136e9b477108effe648d04cb2`

The eps-prediction DDIM base step paired through the Rex coupling yields a reversible DDIM:

- single-step identity: the psi-form update reproduces the plain DDIM affine update with max abs error 0.000e+00;
- exact inversion: the paired DDIM round-trips with max abs error 1.332e-15 over 10 steps;
- consistency: the paired forward trajectory deviates from plain DDIM by 2.302e-05 at 16 steps and 1.441e-05 at 64 steps, so the reversible pairing approaches standard DDIM under refinement.

DPM-Solver and SEEDS-1 recovery routes are audited in the pinned source: rex.py carries the DPM-Solver lambda/time conversion with higher-order exponential tableaus over the same log-SNR variable, and the SDE path (Euler-Maruyama, ShARK over rho) matches the first-order stochastic reversible construction targeted by SEEDS-style solvers.

## Provenance

The evidence pins the upstream GitHub repository at commit `e39b57415d5608b18d7c5631595f1d38f06813b8` and records SHA-256 digests for the audited upstream README, Rex implementation, Runge-Kutta tableau registry, DDIM baseline, license, and requirements file in `evidence/manifest.json`.
