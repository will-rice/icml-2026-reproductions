# Optimizing Rank for High-Fidelity Implicit Neural Representations Reproduction Design

Attempt: `31b721c6-a250-4a46-abac-f341fb5a66a5`
Paper: `2azIa9tfl3`, "Optimizing Rank for High-Fidelity Implicit Neural Representations"
Owner: `agy-paper-owner-05`
Snapshot: `a024dfc72e1980fa8f7eefe5ff47dbce6f23aa7cf8116b0c018e0182bfcea70c`

## Target Claims

1. The paper argues that vanilla MLP INR low-frequency bias is a symptom of stable-rank degradation during training rather than an intrinsic architectural limitation (Section 3). (SHA256: `0670e0c7422f892ec13151b030fba3772815a6b8afa2911bc65bf4ebda3a2fb9`)
2. Rank-regulating, near-orthogonal Muon updates improve image overfitting quality across multiple INR architectures compared with Adam (Table 1). (SHA256: `a61547a408e0908bb8456fb9e932ef0b9b6dfcf95250b0b7ad03dd637c40ef10`)
3. Muon improves sparse-view CT reconstruction quality across multiple INR architectures compared with Adam (Table 4). (SHA256: `fcc893efa61fac59f23688ac2cddf3eddfa930ddda3ee6db84c46d414c7e5933`)
4. The reported improvements extend to natural images, medical images, audio, super-resolution, and novel-view synthesis, with up to about +9 dB PSNR over the same architecture (Tables 1-6). (SHA256: `088d6fac2fdfdbef61cefc94e0c7c7ada940c86b14ddd19114fa022aed352d6d`)

## Scope

This reproduction provides independent, CPU-only evidence for the four target claims of the Muon INR rank optimization paper. It implements an isolated, self-contained Python package under `submissions/optimizing-rank-for-high-fidelity-implicit-neural-representations`. It does not report paper values directly as evidence and runs deterministically without GPU requirements.

## Evidence Plan

1. **Stable-Rank Degradation Mechanism Test (Claim 1)**:
   - Implement stable rank calculation $\|W\|_F^2 / \|W\|_2^2$ for weight matrices in vanilla MLP / INR models during Adam vs Muon optimization on continuous coordinate fitting tasks.
   - Verify that Adam optimization exhibits stable-rank decay/degradation as low-frequency components dominate, whereas Muon updates (which enforce near-orthogonal updates via Newton-Schulz or polar decomposition) maintain higher stable rank and prevent rank collapse.

2. **Image Overfitting Quality Comparison (Claim 2)**:
   - Construct a fast, CPU-friendly synthetic 2D signal fitting benchmark (e.g., Kodak/synthetic image overfitting across Siren/Wire/MLP architectures).
   - Compare PSNR/MSE metrics between Muon optimizer and Adam optimizer under identical hyperparameter budgets, verifying improved reconstruction quality (higher PSNR) for Muon.

3. **Sparse-View CT Reconstruction Test (Claim 3)**:
   - Implement a lightweight Radon transform / 2D CT projection operator to simulate sparse-view tomographic reconstruction using INRs.
   - Evaluate Adam vs Muon INR optimization for sparse-view CT, demonstrating superior PSNR and feature restoration with Muon.

4. **Multi-Domain Extension Evaluation (Claim 4)**:
   - Run multi-domain benchmark experiments (synthetic 1D audio signal, 2D image, and 2D super-resolution tasks).
   - Evaluate relative PSNR gains across domains, verifying consistent improvements up to significant dB gains over baseline Adam.

## Tests

The suite includes unit and integration tests:
- Exact claim bindings and SHA-256 matching.
- Determinism test: repeated runs with fixed seeds yield identical metrics.
- Stable-rank metric calculation correctness test.
- Muon optimizer update orthogonalization test (verifying Newton-Schulz matrix normalization).
- Verification that Muon achieves higher PSNR than Adam on synthetic image overfitting and sparse-view CT tasks.

## Risks And Blockers

- No external GPU compute. All models must be lightweight CPU-executable INRs.
- Heavy dataset downloads are avoided by generating deterministic synthetic 2D/1D test signals matching the target representation tasks.

## Validation Commands

The reproduction package will be validated via:
`uv run pytest submissions/optimizing-rank-for-high-fidelity-implicit-neural-representations`
`uv run python skills/icml-repro-loop/scripts/state.py attest-validation state/repro-loop.json --attempt-id 31b721c6-a250-4a46-abac-f341fb5a66a5 --owner agy-paper-owner-05 --fencing-token 1`
