# ETTFS SNN Fenced Reproduction Design

Attempt: `c4d0ef4f-ff5f-4660-b0a6-deaffcf9022d`
Owner: `agy-paper-owner-06`
Paper: `3EcT46wsdc`, "Efficiently Training Time-to-First-Spike Spiking Neural Networks from Scratch"
Snapshot: `78d58e1e1bb343e2a6c47764adaec8c39c155fc181e07a3329bebb41e7014f88`
Upstream pins: `arxiv:2410.23619`

## Target Claims

1. The temporal weighting decoder reduces average inference time-steps compared with the prior TQ-TTFS decoder across four datasets (Figure 1d).
2. A Fashion-MNIST ablation improves from 89.61% baseline accuracy to 92.90% when ETTFS-init, average pooling, normalization, affine normalization, and TWD are all enabled (Table 4).

## Evidence Strategy

Build a standalone submission at `submissions/efficiently-training-time-to-first-spike-spiking-neural-networks-from-scratch/`.

The executable evidence will implement a deterministic PyTorch/NumPy model of the paper's ETTFS SNN framework:
- ETTFS initialization mechanism balancing post-synaptic currents across layers;
- Temporal Weighting Decoder (TWD) vs TQ-TTFS decoder comparison for latency/time-steps reduction;
- Average pooling vs max pooling TTFS spike constraint preservation;
- Fashion-MNIST ablation pipeline testing baseline vs ETTFS-init, average pooling, normalization, affine normalization, and TWD.

The evidence bundle will include the attempt ID, paper ID, upstream pins, source references, command log, environment summary, target claims, reproduced statuses, and explicit limitations.

## Tests

Use TDD:
1. Write failing tests for ETTFS initialization and PSC distribution stability across layers.
2. Implement ETTFS-init and pooling layer modules.
3. Write failing tests for Temporal Weighting Decoder (TWD) time-step savings vs TQ-TTFS baseline.
4. Implement TWD and ablation evaluation loop on Fashion-MNIST synthetic/subsampled benchmark.
5. Add an evidence-bundle validation test checking provenance and claim-status accounting.

Expected validation commands:
- `pytest tests/`
- `python generate_evidence.py --check`

## Scope and Risks

This reproduction implements deterministic CPU-based evaluation and validation of ETTFS SNN components and ablations on Fashion-MNIST/MNIST datasets. It does not require external GPU training or paid APIs.

The submission is self-contained with standard PyTorch/NumPy/Gradio dependencies.
