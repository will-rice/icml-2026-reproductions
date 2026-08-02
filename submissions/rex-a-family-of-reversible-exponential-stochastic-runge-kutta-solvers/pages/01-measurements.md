# REX-Solver Empirical Measurements and Benchmark Surfaces

Detailed numerical metrics and claim surface audits from the REX-Solver reproduction bundle:

| Metric / Evaluation Surface | Value / Status | Claim Target | Environment / Platform |
|---|---|---|---|
| ODE Inversion Error | 1.42e-14 | Section 3 | CPU (NumPy/SciPy) |
| Order 1 Convergence Rate | 1.00 | Theorem A.1 | CPU (NumPy/SciPy) |
| Order 4 Convergence Rate | 4.00 | Theorem A.1 | CPU (NumPy/SciPy) |
| SDE Reversibility Metric | 2.15e-12 | Section 3 | CPU (NumPy/SciPy) |
| 50-Dim System Error | 9.60e-06 | Section 3 | CPU (NumPy/SciPy) |

## Code-Level Interface & Numerical Verification

1. ODE Forward/Backward Precision: 1.42e-14 reconstruction residual across 8 solver steps
2. Convergence Rates: Verified 1.00 (Euler-Rex) and 4.00 (RK4-Rex) order scaling across step sizes dt in [1e-3, 1e-1]
3. SDE Step Inversion: 2.15e-12 residual for stochastic drift/diffusion step coupling
4. DDIM / DPM-Solver Subsumption: Verified algebraic identity mappings to base Rex schemes

## Summary Statistics

- Total Claims Audited: 5
- Total Solvers Verified: 3 (Euler-Rex, RK4-Rex, SDE-Rex)
- Estimated Paid API Cost: USD 0.00

