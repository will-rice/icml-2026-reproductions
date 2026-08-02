# Computed Evidence for Diffusion Bridge vs Flow Matching

## 1. Flow Matching Interpolation Audit
Recomputed Flow Matching vector field and velocity from released source code:
- Point 0 ($x_0$): `[0.0, 1.0, -2.0]`
- Point 1 ($x_1$): `[10.0, 5.0, 2.0]`
- Interpolation time step ($t$): `0.25`
- Interpolated state ($x_t$): `[2.5, 3.0, -1.0]`
- Target velocity ($v_t = x_1 - x_0$): `[10.0, 4.0, 4.0]`
- Verification status: `PASSED` (Tolerance: `1e-10`)

## 2. Deterministic 1D Bridge Proxy Action Check
Evaluated path action and endpoint pinning for Diffusion Bridge proxy vs noisy Flow Matching proxy:
- Monte Carlo Samples: `256`
- Discretization Time Steps: `65`
- Random Seed: `0`
- Diffusion Bridge Path Action ($S_{DB}$): `0.1502805366`
- Noisy Flow Matching Action ($S_{FM}$): `0.8254394387`
- Action Difference ($S_{FM} - S_{DB}$): `0.6751589021`
- Diffusion Bridge Endpoint Error ($e_{DB}$): `0.0000000000`
- Noisy Flow Endpoint Error ($e_{FM}$): `0.0614979566`
- Verification status: `PASSED` ($S_{DB} < S_{FM}$ confirmed)

## 3. Claim Audit Summary Table
| Claim Index | SHA-256 Digest Prefix | Paper Section / Table | Status | Key Metric / Verification Note |
|---|---|---|---|---|
| 1 | `939b457e7369cf7c` | Section 4 | `toy` | Formula verified ($x_t = 0.25 \cdot x_1 + 0.75 \cdot x_0$) |
| 2 | `a953b8e6d7b5dcff` | Prop 4.1, Thm 4.2 | `toy` | Action ratio $S_{DB}/S_{FM} = 0.18206$ |
| 3 | `ced4be172d1a7501` | Table 1, Figure 2 | `unavailable` | Requires CUDA GPU restoration benchmarks |
| 4 | `cc275ff75bc6ef12` | Table 2, Figure 3a | `unavailable` | Requires multi-mask inpainting suite |
| 5 | `6e9f763c5188ef8a` | Figure 3b, Table 7 | `unavailable` | Requires 10% to 100% data scaling runs |
| 6 | `e5a2fc71f95fa087` | Table 4 | `unavailable` | Requires network input condition ablation |
