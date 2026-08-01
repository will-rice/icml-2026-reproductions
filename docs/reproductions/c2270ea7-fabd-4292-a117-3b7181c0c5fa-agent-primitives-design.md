# Agent Primitives Reproduction Design

- Attempt: `c2270ea7-fabd-4292-a117-3b7181c0c5fa`
- Owner: `codex-paper-owner-05`
- Fencing token: `1`
- Paper: `CzShhpY2qU`
- Snapshot: `262de4b8f7a83b9fa6af23efd0755d5e77522b789239db644faefa5ba4cf9d30`
- Title: `Agent Primitives: Reuseable Latent Building Blocks for Multi-Agent Systems`

## Pinned Artifacts

- Paper: `arxiv:2602.03695`
- OpenReview: `https://openreview.net/forum?id=CzShhpY2qU`
- Hugging Face paper page: `https://huggingface.co/papers/2602.03695`
- Upstream executable repository: not found in the initial primary-source search.

## Evidence Plan

Build a CPU-only evidence package in
`submissions/agent-primitives-reuseable-latent-building-blocks-for-multi-agent-systems`
that:

1. Downloads and hashes the arXiv source/PDF and records exact acquisition
   commands.
2. Inspects source-package text, figures, tables, and appendix files for the
   selected challenge claims.
3. Implements deterministic toy checks for the three primitive abstractions,
   KV-cache-style latent handoff shape invariants, and Organizer configuration
   selection from a lightweight pool.
4. Keeps paper-reported accuracy, token, and latency numbers as context only;
   unless raw benchmark artifacts are found, performance claims are marked
   `inconclusive`.

## Claim Strategy

1. Primitive definitions: target `toy` if source text and deterministic local
   abstractions establish Review, Voting/Selection, and Planning/Execution.
2. KV-cache communication: target `toy` if source text and local tensor-shape
   checks establish latent handoff mechanics; do not claim benchmark evidence.
3. Organizer/knowledge pool: target `toy` with source text plus deterministic
   selection simulation.
4. Accuracy improvements: target `inconclusive` unless raw benchmark outputs
   or executable evaluation artifacts are discovered.
5. Token/latency reductions: target `inconclusive` unless raw timing/token
   artifacts are discovered.

## Validation

Use the standard controller validation sequence:

1. Generate evidence.
2. Run paper-local pytest.
3. Run root pytest.
4. Validate the skill.
5. Run pre-commit.

No GPU or paid API use is planned.
