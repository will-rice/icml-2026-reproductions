# Theorem 3.1 Convergence Audit

## Theorem Statement & Conditions

Theorem 3.1 proves convergence of clipped CAGrad to Pareto-critical points in smooth non-convex multi-objective optimization settings under user-specified weights $w \in \Delta^K$.

Required Preconditions:
1. Weights $w$ lie in the standard simplex $\Delta^K$.
2. Objective functions $f_k$ are $L_k$-smooth with positive constants $L_k > 0$.
3. Step size $\eta \le 1 / L_w$ where $L_w = \sum w_k L_k$.
4. Correction radius constant $0 \le c < 1$.
5. Losses are non-negative $f_k(\theta) \ge 0$.

## Recomputed Audit Observations

- **Precondition Verification:** Evaluated on smooth quadratic case $L = (2.0, 3.0), w = (0.6, 0.4), L_w = 2.4, \eta = 0.1, c = 0.4$.
  - All 7 precondition Booleans hold `True`.
  - Monotonic descent bound holds ($f_{final} = 1.2 \le f_{init} = 1.5$).
  - Stationarity / Pareto criticality bound holds.
  - Audit Outcome: `supported`.

## Verification Commands & Source Pins

- Implementation: `src/reward_free_alignment/theorem_audit.py`
- Test suite: `tests/test_theorem_audit.py`
- Command: `uv run --extra dev pytest tests/test_theorem_audit.py -k test_theorem_31`
- Source Pin: Paper Section 3.2, Theorem 3.1
