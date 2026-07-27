---
title: Numina-Lean-Agent Released-Proof Verification
emoji: 🧮
colorFrom: indigo
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
short_description: Verification of released Lean proof artifacts
tags:
  - icml2026-repro
  - paper-0bTEd4LpQr
---

# Numina-Lean-Agent released-proof verification

This static Space reports partial support for two selected claims by verifying
released Lean proofs. It is not an agent rerun and not an official verdict.
All computed statements in the report, poster, and index resolve from the five
normalized JSON files under `evidence/`.

## Reproduce the static assets

From this project directory:

```bash
python -m numina_lean.space_assets \
  --evidence-dir evidence \
  --output-dir .
pytest -q
```

The exact upstream pin is:

`github:project-numina/numina-lean-agent@1c9af8a52e715f22fede766425ba3d3b95526132+project-numina/Numina-Putnam2025@60d33c8ba19af905bd731e938ebde1c5b8c76519+project-numina/BrascampLieb@413f2bfd31100187eb6c2d632c9cbf12e3115494`

## Scope and license boundary

This is released-proof verification and partial support only. It does not
verify agent attribution, rerun Claude Opus 4.5 or Numina-Lean-Agent, reproduce
the comparison-table experiment, establish interaction with mathematicians,
or constitute an official challenge verdict.

The BrascampLieb repository has no LICENSE file. The agent repository also has
no root LICENSE file. This project links to their pinned revisions but does not
redistribute their source, caches, binaries, raw logs, or environment dumps.
