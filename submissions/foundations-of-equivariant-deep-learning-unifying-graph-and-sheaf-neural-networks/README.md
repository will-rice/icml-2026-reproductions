---
title: Foundations of Equivariant Deep Learning (Sheaf NNs)
emoji: 🕸️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
short_description: Reproduction of Sheaf NNs (ICML 2026)
tags:
- icml-2026
- paper-aIH1jyU37z
- repro-challenge
---

# ICML 2026 Reproduction: Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks

**Paper Title:** Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks
**Paper ID:** `aIH1jyU37z`
**Authors:** Yoshihiro Maruyama
**Upstream Revision:** `arxiv:2012.06333v3+github:twitter-research/graph-neural-sheaves@57002ef2c2c0199d7990be10f0dfc8c83a54d658`
**License:** MIT License
**Space:** `will-rice/repro-foundations-of-equivariant-deep-learning-unifying-graph-and-sheaf-neural-networks`

---

## Reproducibility Claims & Computed Evidence

1. **Sheaf-Laplacian Diffusion (Section 3):** Encodes asymmetric, signed, and varying-dimensional relations via cellular sheaf restriction maps.
2. **Drop-in Generalization (Section 2.1):** When restriction maps are identity matrices, Sheaf Laplacian diffusion matches standard Kipf-Welling GCN diffusion.
3. **Signed Graph Node Classification (Figure 1):** SheafNN outperforms GCN variants across feature/edge noise regimes.
4. **Statistical Error Bars (Figure 1):** Results are averaged over 5 random graph trials with standard deviations.

---

## Quickstart

Run evidence generation:

```bash
python generate_evidence.py
```

Run test suite:

```bash
pytest tests/
```
