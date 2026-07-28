# RACO Score Reproduction Summary

This directory contains deterministic CPU reproduction evidence for the paper **Reward-Free Alignment for Conflicting Objectives** (ICML 2026, Paper ID `vSzRJyg6k0`, arXiv `2602.02495v3`, GitHub `PeterLauLukChen/RACO@84a943c34f38520c7e0c9dd3066517c111b3c8fa`).

> **Notice:** The local outcomes reported here (`supported`, `limited`) reflect local algorithmic and theoretical reproduction audits. They are **not an official verdict** from the challenge controllers or program chairs. All outcomes are derived from audit computations, never hard-coded.

## Live Claim Hashes & Local Outcomes

1. `e9a35e34b57a7273bf84d3d5981ab19f8ff1088adef8363f4640dcf436183944`: **supported** - RACO is an offline reward-free preference alignment method.
2. `7c0aa54e034d03f2d0905417a024af4db41338cd9a803a0b42e441945c307cf9`: **supported** - CAGrad-Clip limits correction gradients to respect trade-offs.
3. `85abbc8a21d5c4537409f6e9f2af6bffc7e4c15e2311dfa078bf816ea0cffc9e`: **limited** - TL;DR summarization Pareto trade-offs (unreplicated empirical training).
4. `dac93f364ac0469302894920781b034bfcd205816fbe16537c2f8e7c10d8995d`: **limited** - BeaverTails safety alignment trade-offs (unreplicated empirical training).
5. `269d8a5053e224206036399bccb2435455565149086de6439a9046d89682772b`: **limited** - Ablations on correction-radius constant (unreplicated empirical training).
6. `0d457572ea8a502fa8a489fef3e15da21b13cc39dd3a3730843d1cbe833059b0`: **supported** (Targeted) - Direct conflict-averse gradient descent on objective-specific pairwise losses. End-to-end pipeline from 3-parameter model losses to gradients to CAGrad-Clip.
7. `50719d645042a500f9c4d53fbdfedf719ee56429ddb73a231912f1eaeadb1b31`: **supported** (Targeted) - Clipped CAGrad update with scale-invariant corrected stationary quadratic and interior alpha.
8. `5ec835ce150ff60d1e2bbd4fbdf7d1ebacf91bb2b6b2d65f72c44c3b3ed65229`: **supported** (Targeted) - Theorem 3.1 convergence audit with executed deterministic T=10 step trajectory and finite-horizon bound.
9. `b74a0ea75967144b210934fd40fd23449d3ef985df878d5a9e14c4b04025ba4b`: **supported** (Targeted) - Theorem 3.2 strict interior witness with $\Gamma(\tilde{\rho}) - \Gamma(\rho) = 0.2091$.
10. `58b31f527bb5e1bccb05c0dab775a74c2f2bdcd8e92ef2c0dd578733b5fb058e`: **limited** - Empirical Pareto trade-offs across Qwen 3, Llama 3, Gemma 3.

## Reproduction Highlights

- **Loss Formulation:** Verified objective-specific pairwise logistic loss against closed-form formula $\mathcal{L}(\theta) = -\log \sigma (\beta (\Delta \log \pi - \Delta \log \pi_{ref}))$. Recomputed loss $= 0.598139$.
- **CAGrad-Clip Solver:** Scale-invariant corrected stationary quadratic (all thresholds relative to data magnitude). Solver now finds interior $\alpha \approx 0.356145$ for the plan witness at all scales from $10^{-8}$ to $10^{8}$, confirmed against independent grid-search minimizer (50k points) and 30 wide-log-scale random trials.
- **Theorem 3.1:** Executed deterministic T=10 step trajectory on nonneg quadratic ($f_1 = x^2, f_2 = (x-1)^2$, $x_0 = 1.0 \to x_{10} = 0.244$). Finite-horizon bound: $\min_{t} \|\nabla L_w\|^2 = 0.026 \le 2 L_w(\theta_0) / (\eta(1-c^2)T) = 1.429$. Every $M(\theta_t)$ and $\|\nabla L_w(\theta_t)\|$ persisted.
- **Theorem 3.2:** Interior strict witness with all 8 paper conditions. $\Gamma(\tilde{\rho}) - \Gamma(\rho) = 0.2091 > 0$ (genuine positive difference, not a near-zero boundary artifact). Identity residual $= 0$.
- **Claim 6 Pipeline:** End-to-end from 3-parameter model through two objective-specific pairwise losses ($L_1 = 0.554$, $L_2 = 0.493$) through gradient extraction (non-colinear) to CAGrad-Clip application. Audit persisted under `audits.claim6_pipeline`.
- **Provenance:** Duplicate JSON keys rejected. Extra manifest keys rejected. Git blob IDs verified. Schema validation mandatory. All outcomes derived from audit results.

## Controller Correction Gate Addressed (Round 5)

1. ✅ **Exact Live Claim Registry Binding.** `load_live_claims` binds to the exact 10 admitted claim ordinals, texts, hashes, and targeted flags. Caller-edited claim text with recomputed matching SHA raises `IntegrityError`.
2. ✅ **Complete Upstream Artifact Lineage.** `VerifiedArtifact` and `evidence-v1.schema.json` include all 8 fields (`artifact_id`, `relative_path`, `sha256`, `git_blob`, `size_bytes`, `source_url`, `acquisition_command`, `license`). Exposed in `evidence/results.json` under `artifacts` and in `pages/05-limitations-and-provenance.md`. Rejects generic repo URLs and clone-only commands without `git checkout 84a943c34f38520c7e0c9dd3066517c111b3c8fa`.
3. ✅ **Executed CAGrad-Clip Trajectory.** `execute_raco_trajectory` executes audited CAGrad-Clip update at every step, persisting per-step weighted anchors, CAGrad directions, next iterates, loss before/after, $M(\theta_t)$, weighted gradient norms, and per-step descent and $M \le \|\nabla L_w\|$ booleans for $t=0,\dots,T-1$. Claim 8 supported only when every step and both finite-horizon bounds pass.
4. ✅ **Scale-aware $M(\theta_t)$ Simplex Solver.** `compute_m_simplex` evaluates gradient difference relative to data scale (`diff_norm_sq <= 1e-12 * scale`). Opposing small gradients $g_1=[10^{-8}], g_2=[-10^{-8}]$ return $0.0$.
5. ✅ **Diagnostic Corrections for Singular Cases.** Zero weighted anchor persists effective correction radius $c=0.0$ in `CAGradResult`. Identical gradients report exact subproblem objective $g_1^T g_0 + c\|g_0\|\|g_1\|$. Literal regressions added.
6. ✅ **Canonical Evidence & Page Determinism.** Evidence byte-for-byte identical across independent generations. Full test suite passes.
