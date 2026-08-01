# Neural Thickets Reproduction Logbook

Attempt: `228a446e-f3c6-4ee1-8d80-28b6d1226520`
Paper: `92oF5bU4cU`
Snapshot: `cd566b1fc072468cea13824a2382d9be6916bd5ffb684b5affcbfa814f753528`
Upstream commit: `536df0a308f3990b6270c991fbb96bd0b779a58e`

## Claim Outcomes

| Claim | Local status | Evidence note |
| --- | --- | --- |
| 1 | `toy` | Pinned source contains the 1D RandOpt toy experiment and deterministic simulation shows denser successful perturbations for the large-model proxy. |
| 2 | `unavailable` | No raw Figure 3 Qwen2.5 sweep outputs were released in the audited artifact set. |
| 3 | `toy` | Deterministic simulation produces multiple task-specialty maxima among successful perturbations. |
| 4 | `unavailable` | Pinned source contains RandOpt, majority voting, PPO/GRPO/ES baseline scaffolding, but no released full-scale benchmark outputs. |
| 5 | `toy` | Population-size mechanism is reproduced in toy simulation; scale-dependent full benchmark results are unavailable without GPU-scale artifacts. |

## Deterministic Toy Observation

- Small-model proxy density: `0.03125`
- Large-model proxy density: `0.609375`
- Ensemble accuracy proxy: `0.894334`
- Best single perturbation proxy: `0.894334`

## Limitations

- No paper-reported benchmark metric is used as reproduced evidence.
- GPU-scale Qwen/Llama/OLMo inference, PPO, GRPO, and ES runs are marked unavailable without released raw outputs.
- Toy results test qualitative RandOpt mechanisms only.

Machine-readable evidence is in `evidence/bundle.json`.
