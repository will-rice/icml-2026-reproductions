# ProcMEM Claims & Evidence Summary

This document summarizes the claim-level verification results and mechanism checks for Attempt `69599dee-e0f4-4f62-b6cf-2f4c6d35493d` (Paper `9kJQjx2B80`).

## Verified / Toy Claims

1. **Non-parametric Skill Learning (Section 3)**
   - Status: `toy`
   - Details: Synthetic Skill-MDP update path proposes executable procedural skills with zero model parameter updates.

2. **Skill-MDP Formalism (Section 3)**
   - Status: `toy`
   - Details: Local `Skill` object enforces activation, execution, and termination conditions on controlled synthetic state.

3. **Non-Parametric PPO & Gate (Section 4)**
   - Status: `toy`
   - Details: Semantic-gradient proposal and clipped-surrogate decision logic are tested on deterministic numeric fixtures.

## Inconclusive / Unreproduced Claims

1. **Cross-Domain Reuse Rates (Table 1)**
   - Status: `inconclusive`
   - Details: Full agent evaluation harness not executed locally.

2. **ALFWorld Memory Tokens & Success Rate (Table 2)**
   - Status: `inconclusive`
   - Details: Token accounting verified in `SkillPool`, but full ALFWorld benchmark not run.

3. **Ablation Studies (Table 3)**
   - Status: `inconclusive`
   - Details: Ablation benchmarks not reproduced locally.
