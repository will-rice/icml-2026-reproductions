# Claim 2: Weight Baseline Subtraction

## Target Claim
The method introduces weight baseline subtraction to make small-batch DMPO training effective (Section 3.4).

## Verification Strategy & Findings
- **Trainer Implementation**: `DMPO/dmpo_trainer.py` contains explicit advantage-centering logic (`advantage_centering`, `advantage_centering_neg`, `advantage_centering_unbias`).
- **Mathematical Form**:
  - Centered advantages: $A_{\text{centered}} = A - \text{strength} \cdot C$.
- **Independent Execution**:
  - Raw advantages: `[0.75, 0.25]`
  - Centering factor: `[0.2, 0.8]`
  - Strength: `0.5`
  - Computed centered advantages: `[0.65, -0.15]`.
- **Status**: Verified implementation structure and deterministic advantage-centering behavior.
