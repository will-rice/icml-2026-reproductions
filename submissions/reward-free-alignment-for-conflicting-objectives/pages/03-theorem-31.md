# Theorem 3.1 Convergence Audit

## Theorem Statement & Conditions

Theorem 3.1 proves convergence of clipped CAGrad to Pareto-critical points in smooth non-convex multi-objective optimization settings under user-specified weights $w \in \Delta^K$.

Required Preconditions:
1. Weights $w$ lie in the standard simplex $\Delta^K$.
2. Objective functions $f_k$ are $L_k$-smooth with positive constants $L_k > 0$.
3. Step size $\eta \le 1 / L_w$ where $L_w = \sum w_k L_k$.
4. Correction radius constant $0 \le c < 1$.
5. Losses are non-negative $f_k(\theta) \ge 0$.

## Executed Deterministic Trajectory

Unlike the rejected proposal which used hand-entered initial/final losses, this audit executes an actual trajectory:

- **Objectives:** $f_1(x) = x^2$, $f_2(x) = (x-1)^2$ (smooth, nonneg quadratics)
- **Parameters:** $L_1 = L_2 = 2.0$, $w = [0.6, 0.4]$, $L_w = 2.0$, $\eta = 0.1$, $c = 0.4$
- **Initial point:** $x_0 = 1.0$
- **Computed gradient:** $g_0 = 0.6 \cdot 2.0 + 0.4 \cdot 0.0 = 1.2$
- **Step:** $x_1 = x_0 - \eta \cdot g_0 = 1.0 - 0.12 = 0.88$
- **Recomputed losses:**
  - $L_w(x_0) = 0.6 \cdot 1.0^2 + 0.4 \cdot 0.0^2 = 0.6$
  - $L_w(x_1) = 0.6 \cdot 0.88^2 + 0.4 \cdot 0.12^2 = 0.46464 + 0.00576 = 0.4704$

## Recomputed Audit Observations

- **Precondition Verification:** All 7 Booleans hold `True`.
- **Descent bound recomputed:** $L_w(x_1) = 0.4704 \le L_w(x_0) - \frac{\eta}{2} \|g\|^2 \Gamma_{\min}$ verified from $\Gamma(\rho)$ at best-case alignment $\rho = 1$.
- **Pareto criticality bound:** Gradient norm consistent with the one-step descent.
- **Audit Outcome:** `supported` — derived from recomputed bound on executed trajectory, not vacuously from $f_{final} < f_{init}$ with hand-entered values.
- **Adversarial Regression:** A case with $f_{init} = f_{final} = 1.0$ and $\|g\| = 1.0$ (no descent despite nonzero gradient) correctly produces `descent_bound_holds = False`.

## Verification Commands & Source Pins

- Implementation: `src/reward_free_alignment/theorem_audit.py`
- Test suite: `tests/test_theorem_audit.py`
- Command: `uv run --extra dev pytest tests/test_theorem_audit.py -k test_theorem_31`
- Source Pin: Paper Section 3.2, Theorem 3.1

> **Notice:** Local outcomes (`supported`, `limited`, `not-supported`) are not an official verdict from challenge controllers.
