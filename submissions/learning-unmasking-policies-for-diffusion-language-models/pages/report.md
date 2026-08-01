# Learning Unmasking Policies Reproduction Evidence

- Attempt: `1e84c33a-e5bd-4a24-b551-de7b4d675054`
- Paper: `F9NDKf5oPy`
- Snapshot: `9e9d22e53a0f5eba83916747aebd400e61cf28e84e57cf5219a34f0c7a3b00dd`
- Repository revision: `35e4830485f1821d57f9ac3f1a303f3d4531fb82`

## Claim Results

### Claim 1: verified

The paper formalizes masked diffusion sampling as an MDP in which the diffusion language model is the environment and the policy chooses which tokens to unmask (Section 3).

- Binding: `333c510d8a8d69cc59827726bb86dd399983b01e7a253af8887d0f2251cda61b`
- Evidence: Repository mechanism audit plus deterministic masked-MDP transition checks.

### Claim 2: verified

The learned unmasking policy is a lightweight single-layer transformer mapping token confidences to unmasking decisions (Section 3.2).

- Binding: `bf6aebbeea700b651067e91333f97aef0e4fffec15565daf27c4cf0e89b06056`
- Evidence: Repository policy/config audit plus local confidence-to-action checks.

### Claim 3: inconclusive

Policy sampling matches state-of-the-art heuristic samplers in semi-autoregressive block generation settings (Figure 4).

- Binding: `28fdce6dd76e8df860c9149960ceaf5f1edf8398eb3da00be9d4844911be16f5`
- Evidence: No trained checkpoint or raw Figure 4 evaluation output is bundled for CPU recomputation.

### Claim 4: inconclusive

Learned policies outperform heuristic unmasking strategies in the full-diffusion generation setting (Figure 5).

- Binding: `d2a26b39ada20dfa224d97c827be80a8f15fe7a06bdc9811c6a53f086fc2e607`
- Evidence: No trained checkpoint or raw Figure 5 evaluation output is bundled for CPU recomputation.

### Claim 5: toy

Visualization of learned full-diffusion policies shows expert-steered policies recovering a left-to-right unmasking order on GSM8K samples (Figure 7).

- Binding: `1969db18a5c2b4252edee8f22c1e947bb7d185870e1016c0f8a491141c5415ac`
- Evidence: Source/config audit plus deterministic expert left-to-right unmasking simulation.

## Limitations

- Benchmark claims are inconclusive without released raw evaluation outputs or CPU-feasible trained policy checkpoints.
- Local simulations verify mechanism contracts and do not substitute for paper-scale model evaluation.
