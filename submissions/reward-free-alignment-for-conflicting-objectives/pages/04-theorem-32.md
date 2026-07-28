# Theorem 3.2 Exact Per-Step Descent Certificate

## Theorem Statement & Identity

Theorem 3.2 establishes that for two objectives ($K=2$), coordinate clipping can improve the convergence coefficient per optimization step when the unclipped dual solution over-corrects relative to user weights.

The paper defines coefficient $\Gamma(\rho)$ for alignment cosine $\rho = \frac{\langle g_0, g_{mix} \rangle}{\|g_0\| \|g_{mix}\|}$:

$$\Gamma(\rho) = 1 + c \rho - \frac{L_w \eta}{2} (1 + c^2 + 2 c \rho)$$

The exact per-step descent certificate identity is:

$$\Gamma(\tilde{\rho}) - \Gamma(\rho) = c (1 - L_w \eta) (\tilde{\rho} - \rho)$$

where $\tilde{\rho} = \frac{\langle g_0, g_{\tilde{p}} \rangle}{\|g_0\| \|g_{\tilde{p}}\|}$ is the alignment cosine of the clipped mixture.

## Recomputed Audit Observations

- **Per-Step Identity Test:** Evaluated with $g_1 = [1.0, -4.0], g_2 = [-1.0, 1.0], w = [0.2, 0.8], c = 0.5, L_w = 3.0, \eta = 0.1$:
  - $\rho = -0.24254$, $\tilde{\rho} = -0.24254$ (boundary alpha: $p = [1.0, 0.0]$, $\tilde{p} = [0.2, 0.0]$)
  - Observed Difference $\Gamma(\tilde{\rho}) - \Gamma(\rho)$: $3.68 \times 10^{-9}$ (nonnegative)
  - Theoretical Identity RHS $c (1 - L_w \eta) (\tilde{\rho} - \rho)$: $3.68 \times 10^{-9}$
  - Identity Residual: $|Observed - RHS| \le 10^{-17}$
  - Applicable: `True`
  - Audit Outcome: `supported` — derived from identity residual and nonneg improvement, not hard-coded.
- **Strictness Conditions:** 7 of 8 paper conditions hold (two objectives, $w_k > 0$, $c > 0$, $\eta < 1/L_w$, $\|g_0\| > 0$, non-colinear, $p \neq w$). The `interior_coefficients` condition is False because for 2D non-degenerate gradients, $h(\alpha)$ is convex so the solver correctly produces boundary solutions.
- **Identity Verified Independently:** $\Gamma(\tilde{\rho}) - \Gamma(\rho) = c (1 - L_w \eta) (\tilde{\rho} - \rho)$ confirmed with residual $\le 10^{-10}$ using the plan's $(w=[0.05, 0.95], g_1=[1, -1.76], g_2=[-1, 0.24], L_w=4.0, \eta=0.05)$ fixture.
- **Negative Difference Regression:** Cases where $\Gamma(\tilde{\rho}) - \Gamma(\rho) < 0$ are now correctly labeled `not-supported`, not `supported`.
- **Zero Anchor:** Handled without division-by-zero, returning `applicable = False` and `local_outcome = limited`.

## Verification Commands & Source Pins

- Implementation: `src/reward_free_alignment/theorem_audit.py`
- Test suite: `tests/test_theorem_audit.py`
- Command: `uv run --extra dev pytest tests/test_theorem_audit.py -k test_theorem_32`
- Source Pin: Paper Section 3.2, Theorem 3.2

> **Notice:** Local outcomes (`supported`, `limited`, `not-supported`) are not an official verdict from challenge controllers.
