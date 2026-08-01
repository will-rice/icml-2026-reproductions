# UDM-GRPO Evidence Report

Attempt `a477051b-45d5-4442-839b-1001759853cd` audits paper `WJcFtJriqv` from snapshot `6c17591130f02ebbe0b47d265ef3c6026182ce0aaf51b004bb753ec9336bc335`.

## Result

- Final clean sample plus forward-process trajectory reconstruction: `verified` by pinned source/config inspection.
- Reduced-Step training: `verified` by configs using `train_steps: 3` and `train_start: [0, 3]`.
- CFG-Free training: `verified` by RL rollout configs using `guidance_scale: 1`.
- GenEval/PickScore/OCR metric and ablation claims: `inconclusive`, because no independent GPU training/image generation/evaluation was run.

No paper-reported table values are counted as reproduced measurements.
