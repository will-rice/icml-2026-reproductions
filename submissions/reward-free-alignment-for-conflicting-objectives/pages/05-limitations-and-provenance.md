# Limitations and Provenance

## Scope Boundaries & Limitations

1. **Deterministic CPU Audits:** This reproduction focuses on the exact mathematical, algorithmic, and theoretical foundations of RACO (objective-specific pairwise losses, weighted CAGrad-Clip dual optimization, and Theorems 3.1 & 3.2). All outcomes are derived from audit computations, never hard-coded.
2. **Empirical Benchmarks:** Full LLM fine-tuning on multi-billion parameter models (Qwen 3, Llama 3, Gemma 3) on summarization (TL;DR) and safety datasets (BeaverTails) was not re-executed due to hardware constraints. Corresponding empirical claims (Claims 3, 4, 5, 10) are marked `limited` locally. No paper-reported empirical values are entered as reproduced measurements.
3. **No LLM Inference:** All tests and evidence generation runs strictly offline on CPU without LLM training or API calls.
4. **Scale-Invariant Solver:** The solver uses scale-relative thresholds ($\max(|A|, |B|, |C|)$ for polynomial degeneracy, relative stationarity verification) so it finds correct interior solutions at all gradient magnitudes from $10^{-8}$ to $10^{8}$. Previous absolute thresholds ($10^{-14}$, $10^{-5}$) caused failures at scales $\neq 1$.
5. **Provenance Hardening:** Fully fail-closed: duplicate JSON keys rejected at every JSON load, extra manifest keys rejected, duplicate artifact IDs/paths rejected, empty artifact entries rejected, Git blob IDs recomputed and verified. Schema validation is mandatory (not conditional). Evidence generation invokes `load_verified_artifacts` during generation. Non-finite gradients and weights rejected.
6. **Claim 6 Consistency:** Uses a 3-parameter model to ensure non-colinear gradients. Audit persisted under `audits.claim6_pipeline`. Page and summary outcome derived from the exact audit, not hard-coded.
7. **Theorem 3.2 Precondition Enforcement:** Every precondition (finite simplex weights, positive finite step size, admissible $c < 1$, finite gradients, interior coefficients, positive $\Gamma$ improvement) is checked before declaring support. Adversarial regressions for each violation.

## Provenance & Pinned Identity

- **Paper ID:** `vSzRJyg6k0`
- **Attempt ID:** `97e213a5-7ca3-4a1b-a500-1ec52d94d87a`
- **Admitted Snapshot ID:** `09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199`
- **ArXiv Pin:** `arxiv:2602.02495v3`
- **GitHub Repository Pin:** `github:PeterLauLukChen/RACO@84a943c34f38520c7e0c9dd3066517c111b3c8fa`
- **API Cost:** USD 0.00

> **Notice:** Local outcomes (`supported`, `limited`, `not-supported`) are not an official verdict from challenge controllers or program chairs.
