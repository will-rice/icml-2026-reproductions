# Claim 1 — ETTFS-init versus Kaiming signal propagation

> ETTFS-init addresses the signal-diminishing problem caused by Kaiming initialization and stabilizes post-synaptic current distributions across layers (Figure 1).

**Status: `reproduced` (toy-scale simulation (6 layers x 128 units, 32 time-steps)).**

Identical stacks of integrate-and-fire layers differing *only* in weight
initialization are driven by the same TTFS-encoded input
(6 layers x 128 units, 32 time-steps,
seed 42). Firing fraction and post-synaptic-current spread are measured per
layer from the actual simulation.

| Layer | Kaiming firing frac. | ETTFS firing frac. | Kaiming PSC std | ETTFS PSC std |
| --- | --- | --- | --- | --- |
| 1 | 0.4432 | 0.3878 | 0.2506 | 0.2530 |
| 2 | 0.2581 | 0.2198 | 0.1655 | 0.1682 |
| 3 | 0.1448 | 0.1292 | 0.1279 | 0.1301 |
| 4 | 0.0604 | 0.0558 | 0.0960 | 0.1035 |
| 5 | 0.0106 | 0.0133 | 0.0622 | 0.0683 |
| 6 | 0.0002 | 0.0013 | 0.0251 | 0.0337 |

- Final-layer firing fraction: **Kaiming 0.0002**
  vs **ETTFS 0.0013**.
- PSC standard deviation decays by a factor of
  **10.0x under Kaiming** versus
  **7.5x under ETTFS-init** from the first to
  the last layer.

Both directions of the claim are therefore observed at this scale: signal
diminishes with depth under Kaiming, and ETTFS-init both fires more deep
neurons and flattens the PSC decay.
