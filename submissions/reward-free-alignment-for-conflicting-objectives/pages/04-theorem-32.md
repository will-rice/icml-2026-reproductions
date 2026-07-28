# Theorem 3.2 Exact Per-Step Descent Certificate

## Theorem Statement & Identity

Theorem 3.2 establishes that for two objectives ($K=2$), coordinate clipping can strictly improve the convergence coefficient per optimization step when the unclipped dual solution over-corrects relative to user weights.

The paper defines coefficient $\Gamma(\rho)$ for alignment cosine $\rho = \frac{\langle g_0, g_{mix} \rangle}{\|g_0\| \|g_{mix}\|}$:

$$\Gamma(\rho) = 1 + c \rho - \frac{L_w \eta}{2} (1 + c^2 + 2 c \rho)$$

The exact per-step descent certificate identity is:

$$\Gamma(\tilde{\rho}) - \Gamma(\rho) = c (1 - L_w \eta) (\tilde{\rho} - \rho)$$

where $\tilde{\rho} = \frac{\langle g_0, g_{\tilde{p}} \rangle}{\|g_0\| \|g_{\tilde{p}}\|}$ is the alignment cosine of the clipped mixture.

## Interior Strict Witness

With the corrected stationary quadratic (see `02-cagrad-clip.md`), the solver finds interior $\alpha \approx 0.356145$ for $g_1 = [1, -4], g_2 = [-1, 1], w = [0.2, 0.8], c = 0.5$. This produces a strict witness satisfying all 8 paper conditions:

| Condition | Value |
|---|---|
| Two objectives | True |
| Positive weights ($w_k > 0$) | True |
| Positive correction radius ($c > 0$) | True |
| Strict step size ($\eta < 1/L_w$) | True |
| Nonzero anchor ($\|g_0\| > 0$) | True |
| Non-colinear gradients | True |
| Interior coefficients ($0 < \alpha < 1$) | True |
| Coefficients differ from weights ($p \neq w$) | True |

## Recomputed Audit Observations

- **Strict Witness:** All 8 conditions hold. With $L_w = 3.0, \eta = 0.1$:
  - $\rho = 0.34578$ (unclipped mixture alignment)
  - $\tilde{\rho} = 0.94333$ (clipped mixture alignment)
  - $\Gamma(\rho) = 0.93352$
  - $\Gamma(\tilde{\rho}) = 1.14267$
  - **Observed Difference:** $\Gamma(\tilde{\rho}) - \Gamma(\rho) = 0.2091 > 0$ (strictly positive)
  - **Identity Residual:** $|Observed - c(1 - L_w \eta)(\tilde{\rho} - \rho)| = 0$ (exact)
  - This is a genuine positive improvement, not a near-zero boundary scaling artifact.
- **Independent Identity Test:** With $w = [0.05, 0.95], g_1 = [1, -1.76], g_2 = [-1, 0.24], L_w = 4.0, \eta = 0.05$:
  - Identity residual $\le 10^{-10}$, applicable = True
- **Zero Anchor:** Handled without division-by-zero, returning `applicable = False` and `local_outcome = limited`.
- **Negative Difference Regression:** Cases where $\Gamma(\tilde{\rho}) - \Gamma(\rho) < 0$ correctly produce `not-supported`.

## Verification Commands & Source Pins

- Implementation: `src/reward_free_alignment/theorem_audit.py`
- Test suite: `tests/test_theorem_audit.py`
- Command: `uv run --extra dev pytest tests/test_theorem_audit.py -k test_theorem_32`
- Source Pin: Paper Section 3.2, Theorem 3.2

> **Notice:** Local outcomes (`supported`, `limited`, `not-supported`) are not an official verdict from challenge controllers.
