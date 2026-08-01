# Learning Unmasking Policies Reproduction Design

- Attempt: `1e84c33a-e5bd-4a24-b551-de7b4d675054`
- Owner: `codex-paper-owner-05`
- Fencing token: `1`
- Paper: `F9NDKf5oPy`
- Snapshot: `9e9d22e53a0f5eba83916747aebd400e61cf28e84e57cf5219a34f0c7a3b00dd`
- Title: `Learning Unmasking Policies for Diffusion Language Models`

## Pinned Artifacts

- arXiv: `2512.09106`
- OpenReview: `https://openreview.net/forum?id=F9NDKf5oPy`
- Hugging Face paper page: `https://huggingface.co/papers/2512.09106`
- Official repository: `https://github.com/apple/ml-rl-dllm`
- Repository revision: `35e4830485f1821d57f9ac3f1a303f3d4531fb82`

## Evidence Plan

Build a CPU-only evidence package in
`submissions/learning-unmasking-policies-for-diffusion-language-models` that:

1. Pins and hashes selected repository source files and configuration files
   from `apple/ml-rl-dllm`.
2. Audits the implementation for the MDP formalization, policy inputs from
   token confidences, single-block transformer policy configuration, and
   sampling/evaluation entry points.
3. Implements deterministic toy checks for masked-state transitions,
   confidence-to-unmask decisions, semi-autoregressive block masking, and
   left-to-right expert policy visualization.
4. Marks Figure 4 and Figure 5 benchmark claims as `inconclusive` unless
   released checkpoints or raw evaluation CSV/JSON outputs are present and
   can be recomputed on CPU.

## Claim Strategy

1. MDP formalization: target `verified` if the repository implements the
   environment/policy boundary and the local toy MDP transition test passes.
2. Lightweight policy: target `verified` if the repository exposes the
   single-block confidence policy and local shape tests pass.
3. Semi-autoregressive comparison: target `inconclusive` without raw
   evaluation outputs; source/config presence alone is not a reproduced
   benchmark.
4. Full-diffusion superiority: target `inconclusive` without raw evaluation
   outputs or CPU-feasible checkpoint evaluation.
5. Left-to-right GSM8K visualization: target `toy` if source/config audit and
   deterministic expert-policy simulation recover left-to-right unmasking.

## Validation

Use the standard controller validation sequence: generate evidence, run
paper-local pytest, run root pytest, validate the skill, and run pre-commit.
No GPU or paid API use is planned.
