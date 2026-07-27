# Numina-Lean-Agent released-proof verification

This report provides partial support from released proofs. It is not an agent rerun and not an official verdict.

## `putnam-12-12` — partial support

**Selected claim:** Using Claude Opus 4.5, Numina-Lean-Agent solves all 12 Putnam 2025 problems, matching AXIOM's 12/12 in the comparison table (Table 1).

**Computed released-proof observation:** The 12 released companion proofs kernel-check without sorryAx.

Limitations:

- Does not rerun Numina-Lean-Agent or Claude Opus 4.5.
- Verifies the released companion proofs, not the agent-attribution or comparison-table experiment.

## `brascamp-lieb-formalization` — partial support

**Selected claim:** The paper reports successful formalization of the Brascamp-Lieb theorem through interaction with mathematicians (Abstract).

**Computed released-proof observation:** The released BrascampLieb.upperBound Gaussian supremum declaration kernel-checks without sorryAx.

Limitations:

- Checks the released Gaussian supremum bound, not the full analytic function-space Brascamp-Lieb theorem.
- Does not verify interaction with mathematicians or rerun Numina-Lean-Agent.
- The released formal statement assumes nonzero ambient dimension.

## Provenance and license boundary

`github:project-numina/numina-lean-agent@1c9af8a52e715f22fede766425ba3d3b95526132+project-numina/Numina-Putnam2025@60d33c8ba19af905bd731e938ebde1c5b8c76519+project-numina/BrascampLieb@413f2bfd31100187eb6c2d632c9cbf12e3115494`

The BrascampLieb repository has no LICENSE file. This bundle links to the pinned repository but does not redistribute its source, caches, binaries, or raw logs. The agent repository also has no root LICENSE file, so its source is not redistributed here.
