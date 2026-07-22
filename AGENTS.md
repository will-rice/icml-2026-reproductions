# ICML 2026 Reproduction Workspace

Read `docs/HANDOFF.md` before starting work and `docs/REMOTE_SETUP.md` before
running commands on a new host.

## Objective

Build independently executable evidence for papers in the ICML 2026 Agent
Repro Challenge. Recompute claims from released artifacts; never present
paper-reported values as reproduced measurements.

## Layout

- `submissions/<paper>/`: independent project, tests, evidence bundle, and
  Space source for one paper.
- `skills/icml-repro-loop/`: versioned source for the reproduction-loop skill.
- `state/repro-loop.json`: resumable machine state for the reproduction loop.
- `docs/HANDOFF.md`: current mutable state and next action.
- `docs/REMOTE_SETUP.md`: host prerequisites, authentication checks, skill
  installation, and verification commands.

## Workflow

1. When processing challenge papers, require and follow `icml-repro-loop`.
   Resume from `state/repro-loop.json` and update it after every phase change
   and external mutation.
2. Inspect the paper's live challenge status before claiming, selecting, or
   publishing it.
3. Pin every upstream repository or dataset revision used as evidence.
4. Write a failing test before evidence-generation code.
5. Run the submission's pytest suite and `uv run pre-commit run -a`.
6. Record commands, revisions, environment, and outputs in a machine-readable
   evidence bundle.
7. Deploy each paper to a separate Hugging Face Space and verify the exact
   deployed commit.
8. Update `docs/HANDOFF.md` after every material milestone.

## Constraints

- Never commit credentials or unredacted environment dumps.
- Do not modify another submission to implement a new paper.
- Do not claim unsupported results. Mark unavailable evidence as unreplicated.
- Keep the canonical NAPE repository at `will-rice/icml-2026-repro` unchanged.
- Treat `submissions/nape/` as an archival exception: do not run, modify, test,
  or format that snapshot in place during parent validation.
- Validate NAPE only from a separate canonical checkout using the pinned
  checkout and validation command sequence in `docs/REMOTE_SETUP.md`.
