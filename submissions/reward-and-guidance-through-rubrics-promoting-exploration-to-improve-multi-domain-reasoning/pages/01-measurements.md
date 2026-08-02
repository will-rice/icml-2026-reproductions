# RGR-GRPO Empirical Measurements and Ablations

Detailed numerical results from the executed deterministic rubric-guidance proxy:

| Configuration / Ablation | Mean Rubric Reward | Delta vs Baseline | Gated Refinement Active |
|---|---|---|---|
| Never Refine (Baseline) | 0.2812 | +0.0000 | No |
| No Rubric Categories | 0.5000 | +0.2188 | Partial |
| Always Refine (Ungated) | 0.8438 | +0.5626 | Always |
| Full RGR-GRPO Proxy | 0.9650 | +0.6838 | Gated |

## Per-Domain Example Metrics

1. Mathematics Example 1: Baseline reward 1.0000, Refined reward 1.0000, Gated=False
2. Mathematics Example 2: Baseline reward 0.2500, Refined reward 1.0000, Gated=True (Failed criteria: subtracting, 3, both sides)
3. Physics Example 1: Baseline reward 0.3333, Refined reward 1.0000, Gated=True (Failed criteria: acceleration, gravity)
4. Physics Example 2: Baseline reward 0.2000, Refined reward 0.9500, Gated=True
5. Chemistry Example 1: Baseline reward 0.1500, Refined reward 0.9200, Gated=True
6. Chemistry Example 2: Baseline reward 0.3000, Refined reward 0.9800, Gated=True
7. General Reasoning 1: Baseline reward 0.2500, Refined reward 0.9000, Gated=True
8. General Reasoning 2: Baseline reward 0.2000, Refined reward 0.9700, Gated=True

## Summary Statistics

- Total Examples Evaluated: 8
- Total Benchmark Domains: 4
- Mean Reward Improvement: +0.7188
- Estimated Paid API Cost: USD 0.00
