# Maximum Likelihood Reinforcement Learning Reproduction

Paper: `EeuLO2BjFN`
Code revision: `7197bbb46a2ecd866da52f6b401ff20a34fe9390`

## Claim Status

- `objective_family`: **toy** - Toy Bernoulli checks verify finite-N MaxRL gradients interpolate from RL toward ML.
- `unbiased_estimator`: **toy** - Exact enumeration expectation matches the truncated MaxRL gradient for small Bernoulli tasks.
- `infinite_compute_limit`: **toy** - The truncation gap decreases monotonically with rollout count in deterministic checks.
- `pareto_dominance`: **unavailable** - Full multi-task Pareto claim requires released large-scale evaluation logs or rerunning GPU training.
- `twenty_x_scaling`: **unavailable** - The 20x scaling claim is not recomputed in this CPU-only evidence bundle.

## CPU Checks

The Bernoulli checks exactly compare the conditional-success estimator expectation with the truncated MaxRL gradient.
Large-scale Qwen3, ImageNet, and full Pareto claims are not reported as reproduced measurements.
