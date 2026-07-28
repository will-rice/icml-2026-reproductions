# RACO Score Reproduction Summary

This directory contains deterministic CPU reproduction evidence for the paper **Reward-Free Alignment for Conflicting Objectives** (ICML 2026, Paper ID `vSzRJyg6k0`, arXiv `2602.02495v3`, GitHub `PeterLauLukChen/RACO@84a943c34f38520c7e0c9dd3066517c111b3c8fa`).

> **Notice:** The local outcomes reported here (`supported`, `limited`) reflect local algorithmic and theoretical reproduction audits. They are **not an official verdict** from the challenge controllers or program chairs. All outcomes are derived from audit computations, never hard-coded.

## Live Claim Hashes & Local Outcomes

1. `e9a35e34b57a7273bf84d3d5981ab19f8ff1088adef8363f4640dcf436183944`: **supported** - RACO is an offline reward-free preference alignment method.
2. `7c0aa54e034d03f2d0905417a024af4db41338cd9a803a0b42e441945c307cf9`: **supported** - CAGrad-Clip limits correction gradients to respect trade-offs.
3. `85abbc8a21d5c4537409f6e9f2af6bffc7e4c15e2311dfa078bf816ea0cffc9e`: **limited** - TL;DR summarization Pareto trade-offs (unreplicated empirical training).
4. `dac93f364ac0469302894920781b034bfcd205816fbe16537c2f8e7c10d8995d`: **limited** - BeaverTails safety alignment trade-offs (unreplicated empirical training).
5. `269d8a5053e224206036399bccb2435455565149086de6439a9046d89682772b`: **limited** - Ablations on correction-radius constant (unreplicated empirical training).
6. `0d457572ea8a502fa8a489fef3e15da21b13cc39dd3a3730843d1cbe833059b0`: **supported** (Targeted) - Direct conflict-averse gradient descent on objective-specific pairwise losses.
7. `50719d645042a500f9c4d53fbdfedf719ee56429ddb73a231912f1eaeadb1b31`: **supported** (Targeted) - Clipped CAGrad update with user-specified objective weights.
8. `5ec835ce150ff60d1e2bbd4fbdf7d1ebacf91bb2b6b2d65f72c44c3b3ed65229`: **supported** (Targeted) - Theorem 3.1 convergence audit to Pareto-critical points.
9. `b74a0ea75967144b210934fd40fd23449d3ef985df878d5a9e14c4b04025ba4b`: **supported** (Targeted) - Theorem 3.2 exact per-step descent certificate identity.
10. `58b31f527bb5e1bccb05c0dab775a74c2f2bdcd8e92ef2c0dd578733b5fb058e`: **limited** - Empirical Pareto trade-offs across Qwen 3, Llama 3, Gemma 3.

## Reproduction Highlights

- **Loss Formulation:** Verified objective-specific pairwise logistic loss against closed-form formula $\mathcal{L}(\theta) = -\log \sigma (\beta (\Delta \log \pi - \Delta \log \pi_{ref}))$. Recomputed loss $= 0.598139$.
- **CAGrad-Clip Solver:** Implemented exact two-objective quadratic dual solver with user weights $g_0 = w_1 g_1 + w_2 g_2$ and coordinate-wise clipping $\tilde{p}_i = \min(p_i, w_i)$. Singular cases (zero radius, colinear, zero anchor) minimize $h(\alpha)$ at both endpoints, not shortcutting to $\alpha = w_1$. Constraint $0 \le c < 1$ enforced.
- **Theorem Audits:** Verified exact per-step identity $\Gamma(\tilde{\rho}) - \Gamma(\rho) = c(1 - \ell_w \eta)(\tilde{\rho} - \rho)$ for Theorem 3.2 with residual $\le 10^{-17}$. Theorem 3.1 descent bound recomputed from $\Gamma(\rho)$, not vacuously from $f_{final} < f_{init}$.
- **Provenance:** Duplicate JSON keys rejected. Schema validation mandatory. All outcomes derived from audit results.

## Controller Correction Gate Addressed

1. ✅ Singular cases minimize $h(\alpha)$ at endpoints; $c=0$ and colinear no longer shortcut to $p=w$.
2. ✅ Theorem 3.1 descent bound recomputed, not vacuous.
3. ✅ Theorem 3.2 requires nonneg improvement for `supported`; negative difference yields `not-supported`.
4. ✅ Duplicate JSON keys rejected; schema validation mandatory.
5. ✅ All statuses derived from audit results, never hard-coded.
6. ✅ Loss value corrected to $0.598139$.
7. ✅ `git diff --check` clean.
