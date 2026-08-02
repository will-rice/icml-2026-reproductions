# d2: Improved Techniques for Training Reasoning Diffusion Language Models

Attempt: `0f0cfb98-1f1a-4c31-8881-32cb6e02abde`
Official code: `github:kuleshov-group/d2@381b9f14f4afd0719297ac852e4015c74e0ed235`

## Claim Results

| Claim | Status | Observation | Limitation |
| --- | --- | --- | --- |
| 1 | verified | Official source contains RL trainer scripts and masked trajectory log-probability code; local finite-state checks recompute trajectory likelihood behavior. | This validates mechanism/source wiring, not full RL training. |
| 2 | verified | Toy any-order log likelihood matches enumeration with delta 0; source constructs the doubled sequence attention path. | The check uses a finite order-invariant toy DLM rather than an 8B checkpoint. |
| 3 | toy | The repository separates any-order causal code from standard LLaDA StepMerge code paths. | The broad empirical claim about any-order support across widely used DLMs was not rerun. |
| 4 | toy | Toy StepMerge error improves from 1.350 with 2 groups to 0.190 with 4 groups and reaches exact at 8 groups. | The computation is a finite likelihood approximation test, not a trained masked DLM run. |
| 5 | inconclusive | Training and evaluation scripts exist, but no raw benchmark result artifacts are released in the pinned repository. | Performance over RL baselines requires large-model runs or raw released outputs. |
| 6 | inconclusive | Released dataset files: 5; machine-readable result files: []. | SOTA reasoning benchmark claims were not reproduced on Countdown, Sudoku, GSM8K, or MATH500. |

## Limits

No paper-reported benchmark numbers are treated as reproduced measurements. Large DLM training and inference were not run.
