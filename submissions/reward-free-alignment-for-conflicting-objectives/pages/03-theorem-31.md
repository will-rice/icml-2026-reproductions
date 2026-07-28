# Theorem 3.1 Convergence Audit

## Theorem Statement & Conditions

Theorem 3.1 proves convergence of clipped CAGrad to Pareto-critical points in smooth non-convex multi-objective optimization settings under user-specified weights $w \in \Delta^K$.

Required Preconditions:
1. Weights $w$ lie in the standard simplex $\Delta^K$.
2. Objective functions $f_k$ are $L_k$-smooth with positive constants $L_k > 0$.
3. Step size $\eta \le 1 / L_w$ where $L_w = \sum w_k L_k$.
4. Correction radius constant $0 \le c < 1$.
5. Losses are non-negative $f_k(\theta) \ge 0$.

## Executed Deterministic T=10 Step Trajectory

This audit executes $T = 10$ RACO steps and persists every $M(\theta_t)$, $\|\nabla L_w(\theta_t)\|$, and $L_w(\theta_t)$:

- **Objectives:** $f_1(x) = x^2$, $f_2(x) = (x-1)^2$ (smooth, nonneg quadratics)
- **Parameters:** $L_1 = L_2 = 2.0$, $w = [0.6, 0.4]$, $L_w = 2.0$, $\eta = 0.1$, $c = 0.4$
- **Initial point:** $x_0 = 1.0$
- **Trajectory:**
  - $L_w(x_0) = 0.6000$, $\|\nabla L_w(x_0)\| = 1.2000$
  - $L_w(x_1) = 0.4704$, $\|\nabla L_w(x_1)\| = 0.9600$
  - $L_w(x_2) = 0.3875$, $\|\nabla L_w(x_2)\| = 0.7680$
  - ... (descent at every step) ...
  - $L_w(x_{10}) = 0.2442$, $\|\nabla L_w(x_{10})\| = 0.1289$
- **Observed minima over $t=0..9$:** $\min \|\nabla L_w(\theta_t)\| = 0.1611$

## Finite-Horizon Bound (Paper Equation)

$$\min_{t=0}^{T-1} \|\nabla L_w(\theta_t)\|^2 \le \frac{2 L_w(\theta_0)}{\eta (1-c^2) T}$$

- **LHS:** $\min_t \|\nabla L_w\|^2 = 0.0259$
- **RHS:** $2 \times 0.6000 / (0.1 \times 0.84 \times 10) = 1.4286$
- **Bound holds:** $0.0259 \le 1.4286$ ✅
- **Note:** Formula uses $2 L_w(\theta_0)$, NOT $L_w(\theta_0) / 2$. Does NOT assume best-case $\rho = 1$. Pareto bound verified independently from descent bound.

## One-Step Descent Bound

$$L_w(\theta_{t+1}) \le L_w(\theta_t) - \frac{\eta (1-c^2)}{2} \|\nabla L_w(\theta_t)\|^2$$

- $L_w(x_1) = 0.4704 \le 0.6000 - 0.042 \times 1.44 = 0.5395$ ✅
- Adversarial regression: $f_{init} = f_{final} = 1.0$ with $\|g\| = 1.0$ correctly produces `descent_bound_holds = False`.

## Verification Commands & Source Pins

- Implementation: `src/reward_free_alignment/theorem_audit.py`
- Test suite: `tests/test_theorem_audit.py`
- Command: `uv run --extra dev pytest tests/test_theorem_audit.py -k test_theorem_31`
- Source Pin: Paper Section 3.2, Theorem 3.1

> **Notice:** Local outcomes (`supported`, `limited`, `not-supported`) are not an official verdict from challenge controllers.
