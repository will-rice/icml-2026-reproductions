# Weighted CAGrad-Clip (Section 3.2)

## Theoretical Formulation

CAGrad computes an update gradient $g$ that maximizes local objective improvement while remaining within radius $c \|g_0\|$ of user-weighted anchor $g_0 = \sum_{k=1}^K w_k g_k$. The dual optimization solves for mixture coefficients $p \in \Delta$:

$$g_0 = \sum_{k=1}^K w_k g_k, \quad g_{mix} = \sum_{k=1}^K p_k g_k$$

Coordinate-wise clipping is applied to prevent over-correction against user preferences:

$$\tilde{p}_k = \min(p_k, w_k)$$

Crucially, $\tilde{p}$ is NOT renormalized back to sum to 1. The clipped update vector is formed as:

$$g = g_0 + c \|g_0\| \frac{g_{\tilde{p}}}{\|g_{\tilde{p}}\|}$$

## Corrected Solver (Stationary Quadratic)

The stationarity condition $h'(\alpha) = 0$ leads to a quadratic $A\alpha^2 + B\alpha + C = 0$ where the key coefficient is:

$$B = \delta^2 q_1 - s^2 q_1 q_2$$

The previous implementation had $B = \delta^2 q_1 - 2s^2 q_2 q_1$ (a spurious factor of 2), which caused incorrect alpha solutions. The derivation from squaring $\delta_b \sqrt{Q(\alpha)} = -s \cdot Q'(\alpha)/2$ gives the correct coefficient without the extra factor.

## Recomputed Observations & Verification

- **Plan Witness:** For $g_1 = [1.0, -4.0], g_2 = [-1.0, 1.0]$, weights $w = [0.2, 0.8]$, $c = 0.5$:
  - Corrected Interior Dual Solution: $\alpha \approx 0.356145$ (interior, $0 < \alpha < 1$)
  - Previous Buggy Solution: $\alpha = 1.0$ (boundary, from incorrect quadratic)
  - Independent grid-search (100k points) confirms: $\alpha \approx 0.356$, $h \approx 0.4222$
  - Clipped Solution: $\tilde{p} = \min(p, w) = [0.2000, 0.6439]$ (Sum $= 0.8439 \le 1.0$)
- **Independent Verification:** 20 seeded random trials (4D gradients) all match grid-search minimizer within tolerance $10^{-2}$.
- **Delta-b Zero Case:** When $\delta_b = 0$ (orthogonal gradients, symmetric anchor contribution), the corrected quadratic gives $\alpha = 0.8$ (interior minimizer of $\|mix\|$), not a boundary solution.
- **Singular Cases Handled:** Exact zero-anchor, identical gradients, zero radius, and colinear gradients minimize $h(\alpha)$ at both endpoints deterministically. $c \ge 1$ is rejected.
- **Claim 7 Local Outcome:** `supported` — derived from corrected solver audit, not hard-coded.

## Verification Commands & Source Pins

- Implementation: `src/reward_free_alignment/cagrad_clip.py`
- Test suite: `tests/test_cagrad_clip.py`
- Command: `uv run --extra dev pytest tests/test_cagrad_clip.py`
- Source Pin: Paper Section 3.2, Algorithm 1

> **Notice:** Local outcomes (`supported`, `limited`, `not-supported`) are not an official verdict from challenge controllers.
