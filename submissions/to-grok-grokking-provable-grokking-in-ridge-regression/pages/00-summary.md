# To Grok Grokking reproduction

This submission provides a CPU-only theorem-structure audit and a
deterministic toy ridge-delay sweep. It does not machine-check proofs
or reproduce the nonlinear ReLU experiments.

## Claim status

- `paper-audit`: The paper proves end-to-end grokking for zero-teacher ridge regression, including early training overfitting, delayed poor generalization, and eventual low generalization error (Theorem 4.1)
- `paper-audit`: Separate theorems decompose grokking into training-loss convergence, poor generalization during overfitting, and eventual generalization (Theorems 4.4-4.6)
- `toy`: Decreasing weight decay and sample size can amplify grokking time in ridge-regression simulations, matching the paper's quantitative hyperparameter predictions (Figure 2)
- `unreplicated`: Two-layer ReLU experiments qualitatively reproduce the predicted grokking-time dependence on hyperparameters beyond the linear setting (Figures 3 and 4)

## Measurements

| sample size | weight decay | overfit step | grokking step | delay |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.030 | 7 | None | None |
| 8 | 0.080 | 7 | 26 | 19 |
| 8 | 0.160 | 7 | 23 | 16 |
| 14 | 0.030 | 12 | None | None |
| 14 | 0.080 | 12 | 25 | 13 |
| 14 | 0.160 | 12 | 23 | 11 |
| 22 | 0.030 | 18 | None | None |
| 22 | 0.080 | 18 | 29 | 11 |
| 22 | 0.160 | 18 | 28 | 10 |
