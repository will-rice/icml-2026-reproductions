# FlexRank Reproduction Summary

## Paper Overview
- **Paper ID**: DK0kvnNelx
- **Title**: FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment
- **ArXiv**: 2602.02680
- **Upstream Repository**: https://github.com/RickZack/FlexRank

## Reproduction Objectives and Verification Results
This reproduction suite verifies the core theoretical and algorithmic mechanisms introduced in the FlexRank paper:

1. **Linear Layer Factorization & Dynamic Programming Component Ordering (Claim 0)**:
   - Evaluates SVD factorization $W \approx U V^T$ for each linear layer into rank-constrained submatrices.
   - Evaluates the dynamic programming algorithm that allocates layer-wise ranks given a total parameter budget.

2. **Synthetic Post-Training Selection Failure (Theorem 4.1 / Claim 1)**:
   - Numerically demonstrates that arbitrary post-training SVD truncation from an unconstrained low-rank minimizer fails to recover optimal submodel minimizers for lower ranks, yielding a positive suboptimality gap.

3. **Nested Subspace Minimizer Preservation (Theorem 4.3 / Claim 2)**:
   - Verifies that nested subspace learning preserves exact minimizers for every sub-rank in synthetic matrix optimization settings, matching theoretical bounds with error $< 10^{-10}$.

## Evidence Generation
All evidence is generated reproducibly via `python generate_evidence.py`, producing `evidence/bundle.json` with verified status metrics across all target claims.
