# RACO Score Reproduction Summary

This directory contains deterministic CPU reproduction evidence for the paper **Reward-Free Alignment for Conflicting Objectives** (ICML 2026, Paper ID `vSzRJyg6k0`, arXiv `2602.02495v3`, GitHub `PeterLauLukChen/RACO@84a943c34f38520c7e0c9dd3066517c111b3c8fa`).

> **Notice:** The local outcomes reported here (`supported`, `limited`) reflect local algorithmic and theoretical reproduction audits. They are **not an official verdict** from the challenge controllers or program chairs. All outcomes are derived from audit computations, never hard-coded.

## Live Claim Hashes & Local Outcomes

1. `e9a35e34b57a7273bf84d3d5981ab19f8ff1088adef8363f4640dcf436183944`: **supported** - RACO is an offline reward-free preference alignment method.
2. `7c0aa54e034d03f2d0905417a024af4db41338cd9a803a0b42e441945c307cf9`: **supported** - CAGrad-Clip limits correction gradients to respect trade-offs.
3. `85abbc8a21d5c4537409f6e9f2af6bffc7e4c15e2311dfa078bf816ea0cffc9e`: **limited** - TL;DR summarization Pareto trade-offs (unreplicated empirical training).
4. `dac93f364ac0469302894920781b034bfcd205816fbe16537c2f8e7c10d8995d`: **limited** - BeaverTails safety alignment trade-offs (unreplicated empirical training).
5. `269d8a5053e224206036399bccb2435455565149086de6439a9046d89682772b`: **limited** - Ablations on correction-radius constant (unreplicated empirical training).
6. `0d457572ea8a502fa8a489fef3e15da21b13cc39dd3a3730843d1cbe833059b0`: **supported** (Targeted) - Direct conflict-averse gradient descent on objective-specific pairwise losses. End-to-end pipeline from losses to gradients to CAGrad.
7. `50719d645042a500f9c4d53fbdfedf719ee56429ddb73a231912f1eaeadb1b31`: **supported** (Targeted) - Clipped CAGrad update with corrected stationary quadratic and interior alpha.
8. `5ec835ce150ff60d1e2bbd4fbdf7d1ebacf91bb2b6b2d65f72c44c3b3ed65229`: **supported** (Targeted) - Theorem 3.1 convergence audit with executed deterministic trajectory.
9. `b74a0ea75967144b210934fd40fd23449d3ef985df878d5a9e14c4b04025ba4b`: **supported** (Targeted) - Theorem 3.2 strict interior witness with $\Gamma(\tilde{\rho}) - \Gamma(\rho) = 0.2091$.
10. `58b31f527bb5e1bccb05c0dab775a74c2f2bdcd8e92ef2c0dd578733b5fb058e`: **limited** - Empirical Pareto trade-offs across Qwen 3, Llama 3, Gemma 3.

## Reproduction Highlights

- **Loss Formulation:** Verified objective-specific pairwise logistic loss against closed-form formula $\mathcal{L}(\theta) = -\log \sigma (\beta (\Delta \log \pi - \Delta \log \pi_{ref}))$. Recomputed loss $= 0.598139$.
- **CAGrad-Clip Solver:** Corrected stationary quadratic B coefficient ($-s^2 q_1 q_2$ not $-2s^2 q_2 q_1$). Solver now finds interior $\alpha \approx 0.356145$ for the plan witness, confirmed against independent grid-search minimizer (100k points) and 20 seeded random trials.
- **Theorem 3.1:** Executed deterministic trajectory on nonneg quadratic ($f_1 = x^2, f_2 = (x-1)^2$, $x_0 = 1.0 \to x_1 = 0.88$). Descent bound recomputed from $\Gamma(\rho)$, not vacuously from $f_{final} < f_{init}$.
- **Theorem 3.2:** Interior strict witness with all 8 paper conditions. $\Gamma(\tilde{\rho}) - \Gamma(\rho) = 0.2091 > 0$ (genuine positive difference, not a near-zero boundary artifact). Identity residual $= 0$.
- **Claim 6 Pipeline:** End-to-end from two objective-specific pairwise losses through gradient extraction to CAGrad-Clip application. Not disconnected fixtures.
- **Provenance:** Duplicate JSON keys rejected. Extra manifest keys rejected. Git blob IDs verified. Schema validation mandatory. All outcomes derived from audit results.

## Controller Correction Gate Addressed

1. ✅ Corrected stationary quadratic ($B = \delta^2 q_1 - s^2 q_1 q_2$). Regressed plan witness ($\alpha \approx 0.356145$, $h \approx 0.4222$) against independent minimizer and seeded property test.
2. ✅ Theorem 3.1 uses executed deterministic trajectory ($x_0 = 1.0 \to x_1 = 0.88$, $L_w(x_0) = 0.6$, $L_w(x_1) = 0.4704$), not hand-entered losses.
3. ✅ Theorem 3.2 with interior strict witness: all 8 conditions hold, $\Gamma$ difference $= 0.2091 > 0$, identity residual $= 0$.
4. ✅ Provenance fully fail-closed: reject duplicate keys, extra manifest keys, duplicate artifacts, empty entries, Git blob drift.
5. ✅ Claim 6 derives CAGrad from gradients computed from objective-specific pairwise losses (end-to-end pipeline).
6. ✅ Pages regenerated from canonical evidence. `uv.lock` committed.
7. ✅ Full suite, two evidence generations, adversarial probes, `git diff --check`.
