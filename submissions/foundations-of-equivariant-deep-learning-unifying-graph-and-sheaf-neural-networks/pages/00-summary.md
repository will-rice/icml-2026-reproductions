# Reproduction Summary: Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks

## Executive Summary
This project provides a complete, deterministic, verified reproduction of the core theoretical and empirical claims from the ICML 2026 submission *Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks* (arXiv:2012.06333 / OpenReview: `aIH1jyU37z`).

## Verified Claims & Results
1. **Sheaf Laplacian Diffusion Operator (Section 3)**: Implemented exact sheaf Laplacian diffusion operator $L_{\mathcal{F}}$ encoding asymmetric, signed, and varying-dimensional relations across graph nodes.
2. **Drop-in Generalization of Graph Convolutional Networks (Section 2.1)**: Verified that when restriction maps $\mathbf{F}_{v \unlhd e} = \mathbf{I}$, SheafNN collapses identically to standard Kipf-Welling GCN diffusion.
3. **Synthetic Signed Graph Node-Classification Benchmark (Figure 1)**: Demonstrated superior performance of SheafNN over standard GCN across feature noise and edge noise regimes.
4. **Statistical Evaluation & Error Bars (Figure 1)**: Evaluated 5-trial random graph benchmarks reporting mean accuracies and standard deviations.

## Implementation Structure
- `src/sheaf.py`: Core sheaf Laplacian operator construction and sheaf diffusion layer.
- `src/benchmark.py`: Synthetic signed graph generator and comparative evaluation pipeline.
- `generate_evidence.py`: Evidence extraction script producing `evidence/evidence.json`.
- `tests/test_sheaf.py`: Unit tests verifying mathematical equivalence and numerical stability.
