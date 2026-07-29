# ETTFS SNN Reproduction Summary

This Space provides an interactive reproduction demonstration of **"Efficiently Training Time-to-First-Spike Spiking Neural Networks from Scratch"** (ICML 2026, arXiv:2410.23619).

## Key Claims Verified

1. **Temporal Weighting Decoder (TWD) Step Reduction**: Replacing standard TQ-TTFS decoders with TWD reduces average inference time-steps across datasets by incorporating temporal decay weighting.
2. **Fashion-MNIST Ablation Accuracy Gains**: Enabling ETTFS initialization, average pooling, layer normalization, affine normalization, and TWD improves accuracy from 89.61% baseline to 92.90%.
3. **Single-Spike Pooling Constraints**: Average pooling preserves single-spike timing constraints in Time-to-First-Spike (TTFS) SNNs, whereas max pooling introduces step discontinuities.

## Reproduction Metrics & Benchmark

The interactive controls below allow running the ETTFS decoder comparison and component ablation experiments directly.
