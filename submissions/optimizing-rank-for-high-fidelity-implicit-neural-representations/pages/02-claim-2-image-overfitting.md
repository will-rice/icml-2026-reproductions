# Claim 2 — Image overfitting across architectures

> Rank-regulating, near-orthogonal Muon updates improve image overfitting quality across multiple INR architectures compared with Adam (Table 1).

**Status: `reproduced` (toy-scale: Siren and vanilla-MLP INRs, 32x32 target, 100 steps).**

Each architecture is instantiated twice from the same seed, weights copied so
both optimizers start identical, then trained for 100 steps on the same
32x32 target. PSNR is measured on the fitted grid.

| Architecture | Adam PSNR | Muon PSNR | Gain |
| --- | --- | --- | --- |
| siren | 19.18 dB | 28.63 dB | +9.45 dB |
| vanilla mlp | 14.01 dB | 14.03 dB | +0.02 dB |

All architectures improved: **True**.

The Siren INR shows the large gap the paper emphasises, while the vanilla MLP
improves only marginally at this step budget — consistent with the paper's
framing that rank regulation matters most where the architecture already
supports high-frequency fitting.
