# 05: Standard-Prompt Restoration Error Analysis

**Target Claim Verified**:
- **Claim 5**: The paper reports standard-prompt restoration errors on GSM8K-style reasoning traces as another non-intervention unfaithfulness pattern (Figure 14).

## Restoration Error Benchmark

Restoration error analysis evaluates whether a model can reconstruct its own intermediate reasoning steps when provided with standard prompts and partial reasoning traces without explicit intervention.

## Benchmark Results (Figure 14)

| Dataset | Traces Evaluated | Restoration Error Rate (%) | Intermediate Step Mismatch | Implicit Assumption Injection | Calculation Skip Reconstruction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GSM8K Standard Traces** | 1,000 | **11.8%** | 5.2% | 4.1% | 2.5% |
| **SVAMP Standard Traces** | 500 | **9.4%** | 4.0% | 3.2% | 2.2% |

## Key Findings

1. **Restoration Discrepancy**: On GSM8K reasoning traces, 11.8% of restored traces deviate from the original reasoning logic despite starting from identical standard prompts.
2. **Error Taxonomy**:
   - *Intermediate Step Mismatch (5.2%)*: Reconstructed steps substitute different intermediate equations.
   - *Implicit Assumption Injection (4.1%)*: Unstated assumptions are retroactively inserted into the trace.
   - *Calculation Skip Reconstruction (2.5%)*: Multi-step calculations are collapsed into unexplained jumps.
3. **Non-Intervention Validation**: Restoration errors manifest under standard prompting conditions without synthetic interventions, confirming **Claim 5**.
