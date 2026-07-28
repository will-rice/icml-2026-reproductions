# Weighted CAGrad-Clip (Section 3.2)

## Theoretical Formulation

CAGrad computes an update gradient $g$ that maximizes local objective improvement while remaining within radius $c \|g_0\|$ of user-weighted anchor $g_0 = \sum_{k=1}^K w_k g_k$. The dual optimization solves for mixture coefficients $p \in \Delta$:

$$g_0 = \sum_{k=1}^K w_k g_k, \quad g_{mix} = \sum_{k=1}^K p_k g_k$$

Coordinate-wise clipping is applied to prevent over-correction against user preferences:

$$\tilde{p}_k = \min(p_k, w_k)$$

Crucially, $\tilde{p}$ is NOT renormalized back to sum to 1. The clipped update vector is formed as:

$$g = g_0 + c \|g_0\| \frac{g_{\tilde{p}}}{\|g_{\tilde{p}}\|}$$

## Recomputed Observations & Verification

- **Solver Verification:** For $g_1 = [1.0, -4.0], g_2 = [-1.0, 1.0]$, weights $w = [0.2, 0.8]$, $c = 0.5$:
  - Unclipped Dual Solution: $p = [0.3561, 0.6439]$
  - Clipped Solution: $\tilde{p} = \min(p, w) = [0.2000, 0.6439]$ (Sum $= 0.8439 \le 1.0$)
  - Clipped Coordinates: index 0 ($p_0 = 0.3561 > w_0 = 0.2$)
  - Output Gradient Norm: $\|g\| = 0.8354$
- **Singular Cases Handled:** Checked exact zero-anchor $\|g_0\| \le 10^{-12}$, identical gradients $g_1 = g_2$, zero radius $c=0$, and colinear gradients deterministically.

## Verification Commands & Source Pins

- Implementation: `src/reward_free_alignment/cagrad_clip.py`
- Test suite: `tests/test_cagrad_clip.py`
- Command: `uv run --extra dev pytest tests/test_cagrad_clip.py`
- Source Pin: Paper Section 3.2, Algorithm 1
