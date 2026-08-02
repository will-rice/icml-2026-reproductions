# GUDA Computed CPU Observations

Paper: `5f0gw9YpZC`

Attempt: `3ffbc4da-8f54-4a81-b70e-8103fe8eda1d`

This page separates CPU-computed observations from paper-reported benchmark
values. The run pins `sony/guda` at commit
`9fcf10cc4362199efc4f975e4a950df826fada07` and does not train diffusion
checkpoints or copy paper wall-clock ratios.

## Synthetic Attribution Checks

| observation | value |
| --- | ---: |
| CIFAR-10 proxy group count | `10` |
| GUDA top-1 accuracy | `1.0` |
| GUDA mean reciprocal rank | `1.0` |
| GUDA nDCG@3 | `1.0` |
| semantic baseline top-1 accuracy | `0.0` |
| gradient baseline top-1 accuracy | `0.0` |

## UnlearnCanvas Metadata

| observation | value |
| --- | ---: |
| train prompt rows | `24000` |
| eval prompt rows | `1200` |
| style count | `60` |
| object count | `20` |
| paper-faithful style count | `16` |

## Anchor And Cost Checks

| observation | value |
| --- | ---: |
| weighted-anchor top-1 accuracy | `1.0` |
| weighted-anchor nDCG@3 | `1.0` |
| uniform-anchor top-1 accuracy | `0.0` |
| uniform-anchor nDCG@3 | `0.9464556027366066` |
| relative training runs versus LOGO proxy | `0.55` |
| paper wall-clock values used | `false` |

## Noisy Partition Check

| observation | value |
| --- | ---: |
| clean partition top-1 accuracy | `1.0` |
| 5% noisy partition top-1 accuracy | `1.0` |
| clean partition nDCG@3 | `1.0` |
| 5% noisy partition nDCG@3 | `1.0` |
