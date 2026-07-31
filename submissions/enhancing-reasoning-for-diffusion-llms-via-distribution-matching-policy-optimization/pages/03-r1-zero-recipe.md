# Claim 3: R1-Zero-like Reasoning Recipe

## Target Claim
DMPO is trained without supervised fine-tuning in an R1-Zero-like recipe for reasoning tasks (Section 4).

## Verification Strategy & Findings
- **Configuration & Reward Tasks**:
  - `DMPO/dmpo_train_config.yaml` sets `loss: wdce`, `alpha: 0.04`, `num_generations: 16`, base model `GSAI-ML/LLaDA-8B-Instruct`.
  - `DMPO/reward_func.py` and `DMPO/data_utils.py` implement verification for GSM8K, MATH, Countdown, and Sudoku.
- **SFT Check**:
  - Training entrypoint `DMPO/dmpo_train.py` omits `SFTTrainer` or SFT phases, launching directly with `DMPOTrainer`.
- **Status**: Audited configuration and entrypoint markers. Full scale LLM training is excluded under CPU-only scope.
