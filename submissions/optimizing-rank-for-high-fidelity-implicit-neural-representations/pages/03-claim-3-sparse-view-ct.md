# Claim 3 — Sparse-view CT reconstruction

> Muon improves sparse-view CT reconstruction quality across multiple INR architectures compared with Adam (Table 4).

**Status: `not_reproduced` (toy-scale: 8-view discrete Radon operator on a 32x32 ellipse phantom).**

This is a genuine inverse problem, not a direct image fit. A deterministic
ellipse phantom is measured through a discrete Radon operator with
**8 views x 32 detector bins =
256 measurements** for **1024 pixels** — a
4x under-determined system. Each INR
is trained **only on the sinogram**; reconstruction PSNR is then evaluated
against the unseen phantom on the full grid.

| Architecture | Adam recon. PSNR | Muon recon. PSNR | Gain |
| --- | --- | --- | --- |
| siren | 14.44 dB | 13.95 dB | -0.49 dB |
| vanilla mlp | 11.54 dB | 10.54 dB | -1.0 dB |

All architectures improved: **False**.

**This claim does not reproduce at this scale — the measured effect runs in
the opposite direction.** Adam achieves higher reconstruction PSNR than Muon
on both architectures, by
0.49 dB (Siren) and
1.0 dB (vanilla MLP).

Two readings are possible and this reproduction does not claim to
distinguish them: either the rank-regulation benefit does not transfer to
under-determined inverse problems at this size and step budget, or the
paper's CT result depends on scale, tuning, or a reconstruction setup this
toy operator does not capture. The learning rates here were not tuned per
task. The number is reported as measured rather than adjusted until it
agrees with the paper.
