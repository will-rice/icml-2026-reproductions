# Claim 2 — Temporal weighting decoder inference steps

> The temporal weighting decoder reduces average inference time-steps compared with the prior TQ-TTFS decoder across four datasets (Figure 1d).

**Status: `partially_reproduced` (toy-scale: four synthetic input regimes, not the paper's four datasets).**

Both decoders read the *same* output spike trains produced by a simulated
two-layer IF network over four input regimes. The TQ-TTFS readout must wait
for its quantization window to close; the temporal weighting decoder
accumulates `exp(-alpha t)` evidence and stops as soon as the top-1/top-2
margin is reached.

| Input regime | TQ-TTFS steps | TWD steps | Reduction |
| --- | --- | --- | --- |
| dense bright | 32.0 | 19.59 | 38.79% |
| dense dark | 32.0 | 32.0 | 0.0% |
| sparse bright | 32.0 | 21.54 | 32.69% |
| sparse dark | 32.0 | 32.0 | 0.0% |

- Mean over regimes: **TQ-TTFS 32.0 steps** vs
  **TWD 26.28 steps**, an overall reduction of
  **17.87%**.

The direction of the claim reproduces, but only partially: two of the four
regimes (`dense dark`, `sparse dark`) never reach the confidence margin and
consume the full window, so the reduction is concentrated in the
bright-input regimes. This is a synthetic four-regime stand-in, not the
paper's four datasets.
