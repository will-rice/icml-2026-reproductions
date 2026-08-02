# Detailed Claim Evidence & Quantitative Results

This page renders the exact code-computed quantitative evidence from `evidence/bundle.json` for paper `tI5CFbRhmV`.

## Summary of Evaluated Claims

| Claim | Target Description | Evidence Status | Computed Key Metrics |
|---|---|---|---|
| Claim 2 | LM-CC Hierarchical Semantic Decomposition | `verified` | LM-CC = 4.0 across 4 synthetic blocks (depth sum = 8) |
| Claim 3 | Task Pass@1 & LM-CC Correlations | `toy` | Significant partial Spearman correlation (r = -0.6303, p = 0.0376) on Program Repair |
| Claim 4 | Semantics-Preserving Rewrite Performance Gains | `toy` | Pass@1 gains of +2.76% (Program Repair) & +4.24% (Code Translation) |

---

## Claim 2: LM-CC Hierarchical Semantic Decomposition

- **Target Claim**: LM-CC builds a hierarchical semantic decomposition from token-entropy signals and syntactic delimiters to estimate model-perceived code complexity.
- **Status**: `verified`
- **Quantitative Observations**:
  - `alpha`: 0.8
  - `synthetic_block_count`: 4
  - `synthetic_depth_sum`: 8
  - `synthetic_lmcc`: 4.0

---

## Claim 3: Spearman Correlation Analysis Across Tasks (Table 2)

- **Target Claim**: LM-CC achieves statistically significant partial Spearman correlations with pass@1 across program repair, code translation, and execution reasoning while controlling for code length.
- **Status**: `toy`

| Task Family | Records | Mean Pass@1 | Median LM-CC | Raw Spearman $r$ ($p$-value) | Partial Spearman $r$ ($p$-value) |
|---|---|---|---|---|---|
| Program Repair | 271 | 23.54% | 53.80 | -0.2484 ($p = 3.53 \times 10^{-5}$) | -0.6303 ($p = 0.0376$) |
| Code Translation | 535 | 46.13% | 18.80 | -0.1622 ($p = 1.65 \times 10^{-4}$) | N/A (0 valid groups) |
| Execution Reasoning | 486 | 94.36% | 8.70 | -0.2273 ($p = 4.09 \times 10^{-7}$) | N/A (0 valid groups) |

---

## Claim 4: Semantics-Preserving Rewrites (Table 3)

- **Target Claim**: Semantics-preserving rewrites that reduce LM-CC improve LLM task performance, with reported gains up to 20.9%.
- **Status**: `toy`

| Task Family | Paired Records | Median LM-CC $\Delta$ | Mean Pass@1 $\Delta$ |
|---|---|---|---|
| Program Repair | 58 | -3.40 | +2.76% (+0.0276) |
| Code Translation | 99 | -3.20 | +4.24% (+0.0424) |
| Execution Reasoning | 180 | -2.70 | -0.67% (-0.0067) |

---

*All metrics recomputed deterministically by `generate_evidence.py` and stored in `evidence/bundle.json`.*
