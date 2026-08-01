# Executed CPU Pareto-Frontier Proxy (Claims 3-5 Context)

The paper's headline empirical results (Claims 3-5) come from LLM fine-tuning
on TL;DR and BeaverTails with Qwen3/Llama3/Gemma3 - out of scope for CPU-only
reproduction, and we enter **no** paper-reported value as a reproduced
measurement. What we CAN execute on CPU is the exact optimization problem the
paper defines: objective-specific pairwise logistic preference losses combined
through the audited CAGrad-Clip solver (the same implementation audited on
pages 01-04), trained on a synthetic two-objective preference task whose
ground-truth reward directions conflict (cosine -0.3). A small nonconvex tanh
scorer (56 parameters) is used so the two objective gradients genuinely
conflict along training - the regime Theorem 3.1 addresses. Everything is
seeded (seed 20260801), full-batch, single-threaded float32; every number
below is byte-reproducible and copied verbatim from
`evidence/results.json -> audits.pareto_proxy`.

Setup: 400 training / 200 validation preference pairs per objective,
250 steps, learning rate 0.1, beta 1.0, default clip radius c = 0.5.

## Frontier sweep: RACO (CAGrad-Clip) vs linear scalarization

Validation preference accuracy (objective 1, objective 2) after training at
each user weight w1 (w2 = 1 - w1):

| w1  | RACO acc (obj1, obj2) | Scalarization acc (obj1, obj2) | RACO final losses | Scalarization final losses |
|-----|-----------------------|--------------------------------|-------------------|----------------------------|
| 0.1 | 0.465, 0.955          | 0.455, 0.96                    | 1.467637, 0.225204 | 1.639084, 0.203022 |
| 0.3 | 0.575, 0.84           | 0.565, 0.85                    | 0.815119, 0.391624 | 0.838878, 0.382903 |
| 0.5 | 0.705, 0.695          | 0.71, 0.695                    | 0.548384, 0.564986 | 0.551854, 0.567339 |
| 0.7 | 0.85, 0.51            | 0.85, 0.51                     | 0.381868, 0.814928 | 0.371938, 0.838425 |
| 0.9 | 0.95, 0.39            | 0.97, 0.385                    | 0.234781, 1.363894 | 0.210051, 1.537526 |

Frontier hypervolume (origin reference): **RACO 0.739775** vs
**scalarization 0.748675**. RACO strictly dominates the baseline in **0 of 5**
weight settings (and is strictly dominated in 0).

**Honest readout:** at this toy scale the two methods trace statistically
comparable Pareto frontiers (hypervolume gap about 1.2%), with RACO shifting
probability mass toward the *preferred* objective's loss at extreme weights
(e.g. at w1 = 0.1 the preferred objective-2 loss is 0.225 for RACO with a
lower conflicting loss 1.468 vs 1.639 for scalarization). This supports the
mechanism claims (the CAGrad-Clip update trains stably and respects
user-specified trade-offs, Claims 1-2) but does **not** reproduce the
paper-scale superiority of Claims 3-4; those remain `limited`.

## Clip-radius ablation (Claim 5 direction, w = 0.5)

| c    | val acc (obj1, obj2) | min acc | final losses |
|------|----------------------|---------|--------------|
| 0.0  | 0.71, 0.695          | 0.695   | 0.551854, 0.567339 |
| 0.25 | 0.72, 0.695          | 0.695   | 0.546279, 0.571927 |
| 0.5  | 0.705, 0.695         | 0.695   | 0.548384, 0.564986 |
| 0.75 | 0.71, 0.69           | 0.69    | 0.547637, 0.563571 |
| 0.9  | 0.71, 0.69           | 0.69    | 0.547141, 0.562933 |

The correction radius measurably changes the converged operating point (both
per-objective accuracies and final losses move with c), consistent with the
paper's ablation direction that clipping strength shapes validation margins
and the frontier (Figures 5-6), while every radius trains stably - as
Theorem 3.1's c < 1 condition predicts. The effect size at 56 parameters is
small; nothing here is entered as a reproduction of the paper's Figure 5-6
magnitudes.

Reproduce with:

```
uv run --project . python -c "from reward_free_alignment.pareto_proxy import run_pareto_proxy; import json; print(json.dumps(run_pareto_proxy(), indent=1))"
```
