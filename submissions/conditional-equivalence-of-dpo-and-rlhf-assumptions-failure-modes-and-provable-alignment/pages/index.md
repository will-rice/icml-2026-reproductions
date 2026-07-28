# Conditional DPO and RLHF Reproduction Evidence Summary

- **Paper ID**: `7UEBX1KU1y`
- **Attempt ID**: `933665ed-b7ed-4d73-9b07-35704660a184`
- **Admitted Snapshot**: `09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199`
- **Source Revision**: `arxiv:2605.20834v1`

## Evaluated Mathematical Claims

1. **Conditional Equivalence**: Audited 112 finite preference configurations ($p_{ref} \in \{0.05, \ldots, 0.90\}$, $r \in \{0.1, \ldots, 2.0\}$, $\beta \in \{0.25, \ldots, 2.0\}$). Verified that the population cross-entropy stationary point $\Delta_{rlhf} = \Delta_{ref} + r/\beta$ matches the finite RLHF optimum. However, the sample-level one-sided DPO loss $L_{DPO}(\Delta) = \text{softplus}(-\beta(\Delta - \Delta_{ref}))$ has a strictly negative derivative everywhere and no finite minimizer. Local outcome: `mixed`.
2. **Relative Advantage Optimization**: Audited 75 preference shift cases. Confirmed that DPO loss reduction monotonically improves relative log-likelihood advantage $\Delta > \Delta_{ref}$ without guaranteeing absolute preference alignment $\Delta > 0$ when $\Delta_{ref} < 0$. Local outcome: `consistent`.
3. **Undesirable Solution Spaces**: Emitted concrete witness pairs ($\Delta_{ref} < \Delta < 0$) where DPO loss decreases while the model policy continues to prefer the dispreferred response $y_l$ ($p(y_w|x) < 0.5$). Local outcome: `consistent`.
4. **Constrained Preference Optimization (CPO)**: Audited 180 grid configurations ($\gamma \in \{0, 0.01, 0.05, 0.1\}$). Certified that for $\gamma = 0$ (45 cases), exact constrained RLHF recovers the finite RLHF optimum. However, for all 135 positive-$\gamma$ cases, the exact constrained RLHF objective is unbounded as $\Delta \to +\infty$ because preferred response probability approaches 1. Separately, substituting the reference policy margin $\Omega_{ref}$ (Equation 17) yields a stationary DPO-like loss with adaptive shift $\Delta_{ref} + (r + \Omega_{ref})/\beta$. Because the exact objective is unbounded while the reference approximation is stationary, this claim contains separable supporting and contradicting observations. Local outcome: `mixed`.
5. **Soft-Margin Ranking Interpretation**: Verified across 150 cases that scaled DPO loss $\beta^{-1} \text{softplus}(-\beta(\Delta - \Delta_{ref}))$ converges monotonically to the hinge loss $\max(0, \Delta_{ref} - \Delta)$ as $\beta \to 256$, with negative target margins $\Delta_{ref} < 0$. Local outcome: `consistent`.
6. **Benchmark SOTA Claim**: Unreproduced due to unavailable author repository `visitworld123/CPO` and GPU training constraints. Local outcome: `not_reproduced`.

## Honest Limitations

- No language model was trained or evaluated.
- The benchmark SOTA claim was not reproduced.
- The advertised author repository was unavailable during assessment.
- Only the official challenge harness can issue official verdict labels.
