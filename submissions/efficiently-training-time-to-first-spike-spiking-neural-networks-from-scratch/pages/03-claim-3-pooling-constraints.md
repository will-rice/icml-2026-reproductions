# Claim 3 — Pooling and the single-spike constraint

> The paper argues max-pooling violates TTFS single-spike constraints, while average-pooling preserves them (Abstract).

**Status: `reproduced` (exact numerical property check on simulated post-synaptic currents).**

A TTFS layer accumulates post-synaptic current over time, so a pooling
operator is compatible with single-spike timing only if it commutes with
temporal summation. That commutation is measured directly on simulated PSC
tensors (16 time-steps, 8x16x14x14):

| Operator | max &#124;pool(sum_t PSC) - sum_t pool(PSC)&#124; |
| --- | --- |
| average pooling | **1.91e-06** |
| max pooling | **6.3324** |

Average pooling is linear and commutes to floating-point precision. Max
pooling does not: the discrepancy is
**6.3324**, i.e. the pooled value depends on
*when* current arrived, which is exactly the timing distortion the paper
describes. Additionally, in
**100%**
of pooling windows the averaged response differs from the earliest spike time
in that window, so the two operators do not agree on the single-spike code.

- `avg_pooling_preserves_single_spike`: **True**
- `max_pooling_preserves_single_spike`: **False**
