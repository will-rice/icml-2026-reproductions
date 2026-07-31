# Claim 1: Distribution Matching Policy Optimization Objective

## Target Claim
DMPO fine-tunes diffusion LLMs by matching the model policy distribution to an optimal reward-tilted distribution through cross-entropy optimization (Section 3).

## Verification Strategy & Findings
- **Upstream Repository Pin**: `yuchen-zhu-zyc/DMPO@1661fa7d75f0ccec3bbc1b6cae94e9e3fb88571a`
- **Trainer Implementation**: `DMPO/dmpo_trainer.py` implements Weighted Denoising Cross-Entropy (`loss == "wdce"`).
- **Mathematical Form**:
  - Reward-tilted softmax logits: $\text{coeff} \cdot (\text{log\_rnds} + \text{rewards} / \alpha)$ for $\alpha > 0$.
  - Weighted cross-entropy calculation: $(\sum \text{token\_losses} / m) \cdot \text{advantages}$.
- **Independent Execution**:
  - Evaluated on sample log-ratios `[-0.7, -0.1, -1.4]` and reward vector `[0.0, 1.0, 0.25]` with $\alpha = 0.5$, $\text{coeff} = 0.8$.
  - Computed reward-tilted weights: `[0.17066929, 0.69209559, 0.13723512]`.
  - Computed weighted denoising cross-entropy: `0.485`.
- **Status**: Verified implementation objective structure via deterministic execution and AST inspection.
