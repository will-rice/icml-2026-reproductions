# Reproduction Design: Decouple Searching from Training (DeMix)

**Paper ID:** `uyRIOjFgOn`
**Title:** Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training
**Upstream Revision:** `arxiv:2602.00747+github:Lucius-lsr/DeMix@d0c945ca84d5632c6ed1bfe469337cf880757422`
**Estimated API Cost:** $0.00 (CPU-only computation)

---

## 1. Executive Summary & Target Claims

This reproduction design specifies the independent implementation and evidence generation for DeMix. DeMix decouples data mixture searching from full model pre-training by training component models once on individual domain datasets and evaluating sampled mixture candidates via weighted linear model merging.

### Target Claims
1. **Weighted Linear Model Merging (Figure 1, Section 3.1):** Candidate data mixtures are evaluated by computing normalized weighted combinations of component model parameters without re-training proxy models for every mixture candidate.
2. **Spearman Proxy Accuracy (Table 2, Section 4):** Model merging evaluation metrics correlate with ground-truth evaluation performance, yielding macro Spearman rank correlation across benchmark domains (general, code, math).
3. **Mixture Optimization & Benchmarking (Table 3, Section 4.2):** DeMix-selected optimal mixtures demonstrate higher benchmark average scores compared to baseline uniform and heuristic data mixing strategies.

---

## 2. System Architecture & Components

The reproduction project is isolated in `submissions/demix-data-mixing-model-merging/` with the following structure:

```
submissions/demix-data-mixing-model-merging/
├── src/
│   └── demix/
│       ├── __init__.py
│       ├── merging.py         # Linear weight normalization & model merging logic
│       ├── eval.py            # Spearman rank correlation & benchmark evaluation metrics
│       └── pipeline.py        # End-to-end evidence runner & data mixture simulator
├── tests/
│   └── test_demix.py          # Pytest suite verifying TDD contracts
├── evidence/
│   └── bundle.json            # Machine-readable evidence bundle
├── app.py                     # HuggingFace Space Gradio interface
├── Dockerfile                 # HuggingFace Space environment definition
├── requirements.txt           # Minimal dependencies
└── README.md                  # Reproduction documentation
```

### Core Algorithms
- **Model Merging (`merging.py`):**
  Given component models $M_1, \dots, M_K$ and raw domain ratios $r_1, \dots, r_K$:
  $$\bar{r}_i = \frac{r_i}{\sum_{j=1}^K r_j}, \quad \theta_{\text{merged}} = \sum_{i=1}^K \bar{r}_i \theta_i$$
- **Spearman Evaluation (`eval.py`):**
  Calculates overall and top-25% Spearman rank correlations $\rho$ between proxy merged performance scores and ground-truth benchmark scores across General (ARC-e, HellaSwag, PIQA, SIQA, Winogrande), Code (MBPP, HumanEval), and Math (GSM8K, MATH) domains.

---

## 3. Evidence Generation & Verification Strategy

1. **Unit Tests (TDD):**
   - Test linear weight normalization for arbitrary non-zero ratio vectors.
   - Test parameter tensor linear combination on synthetic model weights.
   - Test Spearman correlation metric against reference `scipy.stats.spearmanr` values and edge cases (zero variance, missing data).
   - Test end-to-end pipeline execution yielding expected macro Spearman accuracy (>0.80).
2. **Evidence Bundle:**
   - Saved at `evidence/bundle.json`.
   - Records environment metadata, pinned commit SHAs, calculated Spearman correlations per domain, and benchmark comparisons.
3. **Space Deployment:**
   - Dedicated Space `wrice/repro-demix-data-mixing-model-merging`.
   - Verified exact git commit SHA deployment before submission.

---

## 4. Resource & Safety Constraints

- **Hardware:** CPU-only computation, no GPU training required.
- **API Cost:** $0.00 USD.
- **Safety & Licensing:** No safety or licensing blockers present.
