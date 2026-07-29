# Reproduction Report: Unifying Masked Diffusion Models with Various Generation Orders and Beyond

## Summary

This interactive report presents independent CPU reproduction evidence for ICML 2026 Paper ID `ATpOQt9VVd`: "Unifying Masked Diffusion Models with Various Generation Orders and Beyond".

### Key Verification Results

1. **Proposition 3.2 (OeMDM NELBO Decomposition)**: Verified. The Order-aware Masked Diffusion Model (OeMDM) Negative Evidence Lower Bound (NELBO) decomposes exactly into a target token reconstruction term $L_{\text{recon}}$ and an order dynamics velocity mismatch term $L_{\text{vel}}$.
2. **Proposition 3.3 (Autoregressive Order Recovery)**: Verified. Setting the generation order scheduler to deterministic left-to-right sequence order $\pi(t) = t$ recovers exact left-to-right causal autoregressive factorization $P(x) = \prod_{i=1}^L P(x_i | x_{<i})$.
3. **Section 4.1 (LoMDM Single-Objective Joint Training)**: Verified. LoMDM jointly optimizes the diffusion backbone logits and learnable order scheduling logits using a single unified loss function $L_{\text{joint}} = L_{\text{token}} + L_{\text{order}}$, decreasing total joint loss monotonically over optimization steps.

All tests and empirical verifications execute deterministically on standard workstation CPU environments without external paid API or GPU dependencies.
