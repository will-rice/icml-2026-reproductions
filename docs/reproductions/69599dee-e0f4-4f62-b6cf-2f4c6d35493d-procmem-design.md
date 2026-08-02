# ProcMEM Reproduction Design

Attempt: `69599dee-e0f4-4f62-b6cf-2f4c6d35493d`
Paper: `9kJQjx2B80`, "ProcMEM: Learning Reusable Procedural Memory from Experience via Non-Parametric PPO for LLM Agents"
Owner: `codex-paper-owner-04`
Snapshot: `c797d3cfc3dccc0d6e34854ee2969147dce439e23c7dac0f6a8a57e3baeb54e9`

## Upstream Pins

- Challenge revision: `81166abbeb76e5f79ff87e51061b5a0306507203`
- Challenge file hash: `65a632313094067874c7ab2b9f62b87dfb4cf913c7a7052c1c2a29a93ca29940`
- Primary paper: `arxiv:2602.01869v1`
- OpenReview: `https://openreview.net/forum?id=9kJQjx2B80`
- No official code revision has been verified at design time. If implementation finds an author-maintained repository or dataset, it must pin an immutable commit or dataset revision before using it as evidence.

## Target Claims

1. ProcMEM/Skill-Pro learns reusable procedural skills from interaction experience without updating model parameters.
2. Skill-MDP converts passive episodic narratives into executable skills with activation, execution, and termination conditions.
3. Non-Parametric PPO uses semantic gradients for candidate skill generation and a PPO Gate for robust skill verification.
4. Skill-Pro achieves higher reuse rates than baselines in in-domain, cross-task, and cross-agent evaluations.
5. Skill-Pro maintains 816 memory tokens while reporting the highest ALFWorld success rate of 0.90 under extreme compression.
6. Ablations evaluate skill use, online score, and PPO Gate pass-rate contributions.

## Evidence Plan

Build a CPU-only, deterministic evidence bundle for mechanism and accounting claims. The implementation will encode the paper-defined data structures for skills, skill pools, semantic-gradient updates, PPO-style candidate gating, and online score maintenance. Tests will exercise those components on synthetic trajectories to verify that the claimed mechanisms are executable and that no model parameter update path is used.

Quantitative benchmark claims from Tables 1-3 will be treated conservatively. Unless a pinned primary artifact contains raw machine-readable run outputs, the evidence bundle will mark reuse-rate, ALFWorld success-rate, and ablation-performance claims as `unavailable` or `inconclusive`, with paper table values recorded only as paper-reported context.

## Tests First

Before implementation, add failing pytest coverage for:

- challenge snapshot, paper ID, target claim text, and SHA-256 bindings are present in the bundle;
- a skill contains activation, execution, and termination fields and can be applied to a synthetic state;
- semantic gradients can propose a candidate skill without mutating any model parameters;
- PPO Gate accepts only candidates with positive clipped surrogate improvement;
- online score and capacity pruning keep memory compact and deterministic;
- paper-reported benchmark/table values are never emitted as reproduced measurements.

## Expected Commands

- `uv run --project submissions/procmem-learning-reusable-procedural-memory-from-experience-via-non-parametric-ppo-for-llm-agents python submissions/procmem-learning-reusable-procedural-memory-from-experience-via-non-parametric-ppo-for-llm-agents/generate_evidence.py --output submissions/procmem-learning-reusable-procedural-memory-from-experience-via-non-parametric-ppo-for-llm-agents/evidence/bundle.json`
- `uv run --project submissions/procmem-learning-reusable-procedural-memory-from-experience-via-non-parametric-ppo-for-llm-agents python -m pytest submissions/procmem-learning-reusable-procedural-memory-from-experience-via-non-parametric-ppo-for-llm-agents/tests -q`
- controller validation through `attest-validation`, followed by publication, submission observation, bounded watching, exact verdict sync, and release.

## Exclusions

No GPU training, API judge calls, paid services, controller credentials, or unredacted environment dumps are used. The evidence must not present paper-reported values as reproduced measurements.
