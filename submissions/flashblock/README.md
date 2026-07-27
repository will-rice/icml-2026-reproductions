# FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion

This repository provides an independent, reproducible implementation and evidence bundle for ICML 2026 Paper `4jfuNNghPS`: **FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion** (arXiv:2602.05305).

## Reproduction Highlights & Target Claims

1. **`cross-step-attention-stability-discrepancy`**: Demonstrates high stability in block-external attention outputs ($\ge 0.95$ cosine similarity) across diffusion steps within a block compared to block-internal attention ($\le 0.70$).
2. **`block-external-attention-caching-speedup`**: Proves that caching block-external attention $(A_{\text{out}}, L_{\text{out}})$ reduces per-step attention FLOPs from $O(BN)$ to $O(B^2)$, achieving $>1.30\times$ theoretical speedup.
3. **`log-space-attention-composition-fidelity`**: Verifies exact numerical equivalence ($L_\infty < 10^{-5}$) between FlashBlock's log-space composition operator and full single-pass dense attention.

## Usage & Verification

Run tests:
```bash
PYTHONPATH=src uv run pytest -v
```

Generate evidence bundle:
```bash
uv run python generate_evidence.py
```
