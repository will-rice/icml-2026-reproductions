---
name: icml-repro-loop
description: Use when selecting, reproducing, submitting, improving, or continuously processing papers for the ICML 2026 Agent Repro Challenge.
---

# ICML Reproduction Loop

Process one paper at a time. Never label paper-reported values as reproduced; only code-computed outputs can support reproduction. Persist every phase and continue until no eligible work remains or a gate requires the user.

## Required Workflow

1. Resume from `state/repro-loop.json` and `docs/HANDOFF.md`. Refresh the live paper catalog, claims, active claims, queued submissions, and verdicts. Never start another implementation while the current paper has unresolved work or submission state.
2. Read [selection-rubric.md](references/selection-rubric.md). Exclude unavailable or duplicate work, score candidates, compare the top three, select one, and persist `selected`. Record immutable upstream revisions and the estimated API cost.
3. **REQUIRED SUB-SKILL:** Use `superpowers:brainstorming` before implementing each paper. Present the paper-specific evidence design and wait for user approval; persist `design-pending`, then `implementing` with approval.
4. **REQUIRED SUB-SKILL:** Use `superpowers:test-driven-development` for evidence code. Build an independent project under `submissions/<paper-slug>/`; distinguish downloaded inputs, computed outputs, and paper-reported context.
5. Follow [submission-checklist.md](references/submission-checklist.md). Validate evidence and persist `validated` before deployment.
6. **REQUIRED SUB-SKILL:** Use `superpowers:verification-before-completion` before any deployment or success claim. Deploy a separate Space, verify its exact SHA, persist `deployed`, submit only after refreshing live challenge state, and persist `submitted` then `judging`.
7. Poll boundedly and persist each poll. Record every claim-level verdict. Deployment, submission, and pending judging are not completion. Only a received verdict permits `complete`. For the current paper only, a concrete verdict defect may trigger one `judging` -> `improving` attempt before completion; never reselect judged or historical papers.
8. After `complete`, archive to `idle` and repeat. On `blocked`, persist the blocker, return control, and stop. Never autonomously transition `blocked` -> `idle`; only explicit user resolution may reset it.

## Compute And Pause Gates

- Any candidate requiring GPU training is ineligible. An explicitly user-requested GPU project is outside this skill.
- Estimated API cost above USD 10 per paper is ineligible. Track actual cost and pause before exceeding USD 10 or provisioning paid infrastructure.
- Known unsafe execution is ineligible. Unresolved safety ambiguity pauses the loop; work proven safe inside an approved isolation boundary may remain eligible.
- Design approval is the routine per-paper pause. Other pauses are missing credentials, an external outage, paid infrastructure, destructive or ambiguous operations, and an exhausted eligible pool.
- Before any pause, leave a clean repository, persist the current phase and known identifiers, and write the blocker plus exact next action in `docs/HANDOFF.md`.

Update state after each phase change and external mutation. Use same-phase persistence for actual cost, poll time/status, and external IDs. Inspect the authoritative CLI rather than guessing its JSON or transition syntax:

```bash
uv run python skills/icml-repro-loop/scripts/state.py --help
```
