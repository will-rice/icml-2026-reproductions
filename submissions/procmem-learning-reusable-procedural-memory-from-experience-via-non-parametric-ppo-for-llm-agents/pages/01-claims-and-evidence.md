# ProcMEM Detailed Empirical Claims & Audit Table

This page documents the claim-by-claim evaluation metrics, target assertions, and fixture probe outcomes for ProcMEM (Attempt `69599dee-e0f4-4f62-b6cf-2f4c6d35493d`, Paper `9kJQjx2B80`).

## Quantitative Metric Summary

| Claim ID | Target Section / Table | Mechanism Check | Reported Metric | Reproduced Status | Numeric Lines |
|---|---|---|---|---|---|
| Claim 1 | Section 3 | Skill-MDP parameter update | 0 parameter updates | `toy` | Verified line 1 |
| Claim 2 | Section 3 | Skill object field enforcement | 3 fields (activation, execution, termination) | `toy` | Verified line 2 |
| Claim 3 | Section 4 | Non-Parametric PPO gate logic | 1 clipped surrogate threshold | `toy` | Verified line 3 |
| Claim 4 | Table 1 | In-domain ALFWorld reuse rate | 88.5% in-domain reuse | `inconclusive` | Paper metric 4 |
| Claim 5 | Table 1 | Cross-task ALFWorld reuse rate | 76.2% cross-task reuse | `inconclusive` | Paper metric 5 |
| Claim 6 | Table 1 | Cross-agent ALFWorld reuse rate | 81.4% cross-agent reuse | `inconclusive` | Paper metric 6 |
| Claim 7 | Table 2 | Memory token compression limit | 816 memory tokens | `inconclusive` | Paper metric 7 |
| Claim 8 | Table 2 | ALFWorld success rate under compression | 0.90 success rate | `inconclusive` | Paper metric 8 |
| Claim 9 | Table 3 | Full ProcMEM ablation score | 0.92 overall score | `inconclusive` | Paper metric 9 |
| Claim 10 | Table 3 | w/o Skill Use ablation score | 0.45 overall score | `inconclusive` | Paper metric 10 |
| Claim 11 | Table 3 | w/o Online Score ablation score | 0.68 overall score | `inconclusive` | Paper metric 11 |
| Claim 12 | Table 3 | w/o PPO Gate ablation score | 0.74 overall score | `inconclusive` | Paper metric 12 |

## Fixture Execution Log & Token Accounting

1. ProcMEM Skill Pool Size: 10 candidate procedural skills.
2. Max Memory Token Budget: 816 tokens maximum.
3. Observed Parameter Delta: 0.0 MB parameter changes during execution.
4. Non-Parametric PPO Gate Threshold: 0.20 clipping margin.
5. Activation Condition Evaluation Count: 100 synthetic iterations.
6. Execution Success Count: 100/100 synthetic state updates.
7. Termination Condition Pass Rate: 1.00 ratio.
8. Baseline Comparison Count: 3 baselines (ReAct, Reflexion, ExAct).
