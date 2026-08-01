# Agent Primitives Reproduction Evidence

- Attempt: `c2270ea7-fabd-4292-a117-3b7181c0c5fa`
- Paper: `CzShhpY2qU`
- Snapshot: `262de4b8f7a83b9fa6af23efd0755d5e77522b789239db644faefa5ba4cf9d30`
- Generated: `2026-08-01T00:00:00+00:00`

## Claim Results

### Claim 1: toy

Agent Primitives instantiates three reusable MAS building blocks: Review, Voting and Selection, and Planning and Execution (Section 3).

- Binding: `8115a999582028cd40604b3d6dd9ee69546547d4babd00b02e8c1abb1b719ff1`
- Evidence: Source-term audit plus deterministic Review, Voting/Selection, and Planning/Execution simulations.

### Claim 2: toy

The primitives communicate internally through KV-cache states rather than only natural-language message passing (Section 3).

- Binding: `5b9aaefaf80d29d999f70165b70dde7f2e9cb90f080dffee3c41a8f4d17228b5`
- Evidence: Source-term audit plus local KV-cache tensor shape invariant checks.

### Claim 3: toy

An Organizer agent selects and composes primitives for each query using a lightweight pool of previously successful configurations (Section 3).

- Binding: `3ff1e2007a7b9f313467db56636bd1e60598d8dfad968e2cec21aeff157168bc`
- Evidence: Source-term audit plus deterministic overlap-based primitive-pool selection.

### Claim 4: inconclusive

Primitive-based MAS improve average accuracy by 12.0-16.5% over single-agent baselines across evaluated tasks (Section 4).

- Binding: `7eadb6ed2e71e12ea036c70a7f6c4e4ef52c30861df2d99f613d30d1b0012129`
- Evidence: No released raw benchmark outputs or executable evaluation artifacts were available in this package.

### Claim 5: inconclusive

Compared with text-based MAS, Agent Primitives reduce token usage and inference latency by about 3-4x while adding only 1.3-1.6x overhead over single-agent inference (Appendix E).

- Binding: `67bd705462e640c09cc50efc1020c8eb02746643ecc5cfa913ef7f6d80da91a1`
- Evidence: No released raw token or latency traces were available in this package.

## Limitations

- Toy statuses are limited to local mechanism checks and source-term audit.
- Accuracy, token, and latency claims remain inconclusive without released raw benchmark outputs.
- Paper-reported values are recorded only as claim text, not as reproduced measurements.
