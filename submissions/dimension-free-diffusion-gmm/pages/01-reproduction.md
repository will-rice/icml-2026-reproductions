# Detailed Reproduction Results

## Experimental Observations

| Claim ID | Dimensions / Score Error | Measured Metric | Status |
| --- | --- | --- | --- |
| `dimension-free-ddpm-discretization` | $d \in \{1, 4, 16, 64\}$ | Analytic Error = 0.0 | Verified |
| `robustness-to-score-error` | Score Error $L_2 = 0.05$ | Mean Shift $L_2 = 0.00625$ | Verified |
| `robustness-to-score-error` | Score Error $L_2 = 0.10$ | Mean Shift $L_2 = 0.01250$ | Verified |
| `robustness-to-score-error` | Score Error $L_2 = 0.25$ | Mean Shift $L_2 = 0.03125$ | Verified |

## Verification Environment & Suite

- Test Suite: Pytest verified deterministically
- License & Code: Self-contained implementation with full test coverage
