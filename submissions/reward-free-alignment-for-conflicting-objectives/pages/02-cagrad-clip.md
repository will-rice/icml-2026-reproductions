# Weighted CAGrad-Clip (Section 3.2)

## Theoretical Formulation

CAGrad computes an update gradient $g$ that maximizes local objective improvement while remaining within radius $c \|g_0\|$ of user-weighted anchor $g_0 = \sum_{k=1}^K w_k g_k$. The dual optimization solves for mixture coefficients $p \in \Delta$:

$$g_0 = \sum_{k=1}^K w_k g_k, \quad g_{mix} = \sum_{k=1}^K p_k g_k$$

Coordinate-wise clipping is applied to prevent over-correction against user preferences:

$$\tilde{p}_k = \min(p_k, w_k)$$

Crucially, $\tilde{p}$ is NOT renormalized back to sum to 1. The clipped update vector is formed as:

$$g = g_0 + c \|g_0\| \frac{g_{\tilde{p}}}{\|g_{\tilde{p}}\|}$$

## Solver Behavior

For the two-objective case ($K=2$), the minimization of $h(\alpha) = \langle \alpha g_1 + (1-\alpha) g_2, g_0 \rangle + c \|g_0\| \|\alpha g_1 + (1-\alpha) g_2\|$ is convex on $[0,1]$ for non-degenerate gradients. The solver enumerates endpoints, roots of $Q(\alpha) = 0$, and stationary candidates, choosing the alpha that minimizes $h$. Singular cases (zero radius, zero anchor, identical/colinear gradients) are handled by evaluating $h$ at both endpoints and selecting the minimum, rather than defaulting to $\alpha = w_1$. The constraint $0 \le c < 1$ is enforced; $c \ge 1$ is rejected.

## Recomputed Observations & Verification

- **Solver Verification:** For $g_1 = [1.0, -4.0], g_2 = [-1.0, 1.0]$, weights $w = [0.2, 0.8]$, $c = 0.5$:
  - Unclipped Dual Solution: $p = [1.0, 0.0]$ (boundary, $h$ is convex in 2D non-degenerate case)
  - Clipped Solution: $\tilde{p} = \min(p, w) = [0.2000, 0.0]$ (Sum $= 0.2 \le 1.0$)
  - Clipped Coordinates: index 0 ($p_0 = 1.0 > w_0 = 0.2$)
  - Output Gradient Norm: $\|g\| = 0.6022$
- **Singular Cases Handled:** Checked exact zero-anchor $\|g_0\| \le 10^{-12}$, identical gradients $g_1 = g_2$, zero radius $c=0$, and colinear gradients deterministically. All singular cases minimize $h(\alpha)$ at both endpoints.
- **Degenerate Regression:** $c=0$ and colinear cases no longer shortcut to $\alpha = w_1$; they evaluate $h(0)$ and $h(1)$ and select the minimum.
- **Claim 7 Local Outcome:** `supported` — derived from solver audit, not hard-coded.

## Verification Commands & Source Pins

- Implementation: `src/reward_free_alignment/cagrad_clip.py`
- Test suite: `tests/test_cagrad_clip.py`
- Command: `uv run --extra dev pytest tests/test_cagrad_clip.py`
- Source Pin: Paper Section 3.2, Algorithm 1

> **Notice:** Local outcomes (`supported`, `limited`, `not-supported`) are not an official verdict from challenge controllers.
