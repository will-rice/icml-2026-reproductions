# Optimizing Rank for High-Fidelity Implicit Neural Representations Reproduction Summary

## Paper Overview
- **Paper ID**: 2azIa9tfl3
- **Title**: Optimizing Rank for High-Fidelity Implicit Neural Representations
- **ArXiv**: 2512.14366
- **Upstream Revision**: arxiv:2512.14366v1

## Reproduction Objectives and Verification Results
This reproduction package provides independent, CPU-only verification of the four core target claims in the Muon INR rank optimization paper:

1. **Stable-Rank Degradation Mechanism (Claim 1)**:
   - Verifies that Adam optimization exhibits stable-rank decay ($||W||_F^2 / ||W||_2^2$) as low-frequency components dominate during continuous coordinate fitting, whereas near-orthogonal Muon updates maintain stable rank and prevent rank collapse during training.

2. **Image Overfitting Quality Comparison (Claim 2)**:
   - Demonstrates that rank-regulating Muon updates consistently improve PSNR/MSE metrics across multiple INR architectures (Siren, Wire, vanilla MLP) on 2D image overfitting tasks compared to Adam under identical budgets.

3. **Sparse-View CT Reconstruction Quality (Claim 3)**:
   - Evaluates sparse-view tomographic reconstruction using INRs with a Radon transform operator, confirming superior PSNR and feature restoration for Muon over Adam.

4. **Multi-Domain Extension Evaluation (Claim 4)**:
   - Confirms that PSNR improvements extend across diverse data modalities including 1D audio signals, 2D medical/natural images, and 2D super-resolution tasks.

## Evidence Generation
All claims are independently verified without using reported paper values as evidence. Running `uv run python generate_evidence.py` generates `evidence/evidence.json` with all target claims verified.
