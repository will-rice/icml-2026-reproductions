# Detailed Evidence Measurements & Theorem Audit

This page details the empirical measurements and theorem structural audits for **To Grok Grokking: Provable Grokking in Ridge Regression** (Paper ID `5nNNVY8NW4`).

## Hyperparameter Sweep Measurements

Below are the exact quantitative results from the deterministic CPU hyperparameter sweep across sample sizes (\(N \in \{8, 14, 22\}\)) and weight decay values (\(\lambda \in \{0.03, 0.08, 0.16\}\)):

| Sample Size (\(N\)) | Weight Decay (\(\lambda\)) | Overfit Step | Grokking Step | Delay Steps | Final Train Loss | Final Test Loss | Condition Proxy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.030 | 7 | None | None | 4.25e-18 | 0.388712 | 3.2059 |
| 8 | 0.080 | 7 | 26 | 19 | 4.25e-18 | 0.226770 | 3.2405 |
| 8 | 0.160 | 7 | 23 | 16 | 4.25e-18 | 0.160339 | 2.5113 |
| 14 | 0.030 | 12 | None | None | 1.18e-10 | 0.389290 | 4.3002 |
| 14 | 0.080 | 12 | 25 | 13 | 1.18e-10 | 0.227197 | 4.6654 |
| 14 | 0.160 | 12 | 23 | 11 | 1.18e-10 | 0.160339 | 4.6697 |
| 22 | 0.030 | 18 | None | None | 4.82e-07 | 0.386677 | 21.6752 |
| 22 | 0.080 | 18 | 29 | 11 | 4.82e-07 | 0.225976 | 8.7951 |
| 22 | 0.160 | 18 | 28 | 10 | 4.82e-07 | 0.159282 | 9.4067 |

## Key Findings

1. **Weight Decay Effect**: Increasing weight decay from \(0.08\) to \(0.16\) shortens the grokking delay (e.g. from 19 down to 16 steps for \(N=8\), and 13 down to 11 steps for \(N=14\)).
2. **Sample Size Effect**: For low weight decay (\(\lambda = 0.03\)), training loss converges early but generalization error remains high within the step limit, confirming the predicted overfitted regime.

## Theorem Audit Summary

- **Theorem 4.1**: Stated end-to-end grokking result for zero-teacher ridge regression verified structurally in paper text (arXiv:2601.19791v4).
- **Theorems 4.4–4.6**: Three-phase decomposition (fit, delay, generalization) verified as separate analytical claims.
