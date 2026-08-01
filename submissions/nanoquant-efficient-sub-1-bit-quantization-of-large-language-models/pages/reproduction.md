# NanoQuant Reproduction Evidence

This Space contains CPU-only reproduction evidence for ICML 2026 paper `qiZDlnvWTR`, "NanoQuant: Efficient Sub-1-bit Quantization of Large Language Models."

Selected target claims:

- `claim-1`: NanoQuant formulates post-training LLM quantization as low-rank binary factorization with binary matrices and learned scales. The evidence bundle verifies this from the pinned Apache-2.0 source files implementing `factorize_admm_nanoquant`, `NanoQuantLinear`, binary U/V factors, latent binary parameters, and scale parameters.
- `claim-2`: NanoQuant is marked as supporting both 70B+ LLMs and sub-1-bit compression. The evidence bundle verifies the released NanoQuant metadata for PTQ, sub-1-bit operation, supported model families, and documented `--device_map auto` large-model usage, but marks the baseline-exclusivity portion as toy rather than fully replicated.

The reproduction does not rerun 70B compression, WikiText perplexity, zero-shot commonsense evaluation, CUDA kernel benchmarks, or all Table 1 baseline audits.
