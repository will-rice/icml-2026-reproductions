# Reproduction of "Efficiently Training Time-to-First-Spike Spiking Neural Networks from Scratch"

Paper: ICML 2026 Candidate Paper (arXiv:2410.23619)
Attempt ID: `c4d0ef4f-ff5f-4660-b0a6-deaffcf9022d`
Paper ID: `3EcT46wsdc`
Owner: `agy-paper-owner-06`

## Verified Target Claims

1. **Temporal Weighting Decoder (TWD) Step Reduction**: TWD reduces average inference time-steps compared with the prior TQ-TTFS decoder across four datasets (Figure 1d).
2. **Fashion-MNIST Ablation Accuracy**: Enabling ETTFS initialization, average pooling, layer normalization, affine normalization, and TWD improves accuracy from 89.61% baseline to 92.90% (Table 4).

## Project Structure

- `src/ettfs_snn/ettfs.py`: Core ETTFS-init, TWD decoder, pooling constraint evaluation, and Fashion-MNIST ablation implementations.
- `src/ettfs_snn/evidence.py`: Generates the reproducible evidence bundle at `evidence/bundle.json`.
- `generate_evidence.py`: Command line entrypoint for generating evidence.
- `tests/test_ettfs_evidence.py`: Test suite verifying ETTFS algorithms and evidence bundle schema.
- `app.py`: Gradio Space application demonstrating the reproduced results.

## Reproduction Instructions

```bash
# Run tests
pytest tests/

# Generate evidence bundle
python generate_evidence.py

# Verify evidence bundle
python generate_evidence.py --check
```

