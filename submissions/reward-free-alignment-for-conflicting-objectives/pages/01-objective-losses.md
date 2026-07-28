# Objective-Specific Pairwise Losses (Section 3)

## Theoretical Formulation

RACO applies preference alignment directly to pairwise preference datasets for each objective $k \in \{1, \dots, K\}$ without training explicit scalar reward models. For objective $k$, the pairwise logistic loss is:

$$\mathcal{L}_k(\theta) = -\mathbb{E}_{(y_w, y_l) \sim \mathcal{D}_k} \left[ \log \sigma \left( \beta \left( (\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)) - (\log \pi_{ref}(y_w|x) - \log \pi_{ref}(y_l|x)) \right) \right) \right]$$

Gradients $g_k = \nabla_\theta \mathcal{L}_k(\theta)$ are computed separately for each objective and are NOT scalarized prior to multi-objective optimization.

## Recomputed Observations & Verification

- **Closed-Form Match:** Evaluated policy gap $\Delta = -0.2 - (-0.8) = 0.6$, reference gap $\Delta_{ref} = -0.4 - (-0.6) = 0.2$, $\beta = 0.5$.
  - Recomputed Loss: $0.598139$
  - Closed-Form Expected: $-\log \sigma(0.5 \times (0.6 - 0.2)) = -\log \sigma(0.2) = 0.598139$
  - Residual: $< 10^{-12}$
- **Separation Check:** Verified that objective losses return 1D tensor of shape $(K,)$ and independent gradient tuples.
- **Claim 6 Local Outcome:** `supported` — derived from closed-form match verification, not hard-coded.

## Verification Commands & Source Pins

- Implementation: `src/reward_free_alignment/pairwise.py`
- Test suite: `tests/test_pairwise.py`
- Command: `uv run --extra dev pytest tests/test_pairwise.py`
- Source Pin: Paper Section 3; Repository `PeterLauLukChen/RACO@84a943c34f38520c7e0c9dd3066517c111b3c8fa`

> **Notice:** Local outcomes (`supported`, `limited`, `not-supported`) are not an official verdict from challenge controllers.
