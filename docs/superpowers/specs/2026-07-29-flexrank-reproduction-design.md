# Reproduction Design: FlexRank (DK0kvnNelx)

## Paper Details
- **Title**: FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment
- **Paper ID**: DK0kvnNelx
- **Slug**: flexrank-nested-low-rank-knowledge-decomposition-for-adaptive-model-deployment
- **Upstream Revision Pin**: `arxiv:2602.02680+github:RickZack/FlexRank@56678319fa63e6bfeb982f432fbea12cb0fd5cd2`

## Target Claims
1. **Claim 0**: FlexRank factorizes each linear layer, obtains a global component ordering with dynamic programming, and distills nested submodels of different sizes from the base model (Figure 1).
2. **Claim 1**: The synthetic analysis shows post-training selection almost surely fails to recover optimal lower-rank submodels from arbitrary global low-rank minimizers (Theorem 4.1).
3. **Claim 2**: Nested subspace learning preserves nested minimizers for every rank in the synthetic matrix setting (Theorem 4.3).

## Verification Implementation Strategy
1. Implement static structure tests for linear layer factorization and dynamic programming component ordering algorithm matching Section 3 and Figure 1.
2. Implement numerical verification suite for Theorem 4.1 (synthetic matrix factorization failure under post-training SVD truncation).
3. Implement numerical verification suite for Theorem 4.3 (nested subspace minimizer preservation across all sub-ranks).
4. Package evidence, tests, and Hugging Face Space for automated validation and reproduction attestation.
