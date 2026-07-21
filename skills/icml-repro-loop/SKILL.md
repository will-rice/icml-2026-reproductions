---
name: icml-repro-loop
description: Use when selecting, reproducing, submitting, improving, or continuously processing papers for the ICML 2026 Agent Repro Challenge.
---

# ICML Reproduction Loop

Process one paper at a time. Only code-computed outputs support claims. Persist state until no eligible work remains or a gate requires the user.

## Mandatory Response/Action Contract

Every response must name its phase and applicable items. Omission bars a success claim.

- **Selection or design:** name live claim-state refresh, next `state/repro-loop.json` write (phase and fields), design presentation and explicit-approval wait before code, and exact continue/stop action.
- **Implementing or later:** name live refresh when relevant, next state write, confirm design approval is already recorded (do not request it again), and exact continue/stop action.
- **Tools unavailable:** propose and label these writes unperformed: `state/repro-loop.json: <phase and fields>` and `docs/HANDOFF.md: blocker <reason>; next action <action>`. Never claim either occurred.

## Required Workflow

1. Resume from `state/repro-loop.json` and `docs/HANDOFF.md`; refresh live catalog, claims, queues, and verdicts. Do not start another implementation with unresolved work or submission state.
2. Read [selection-rubric.md](references/selection-rubric.md). Check live claim state and primary artifacts. While idle, persist each ineligible candidate and reason with the `reject` command; do not transition phase. Continue ranking. Score, compare the top three eligible, select, and persist `selected` with revisions and API estimate. Stop only after selection or a persisted exhausted pool.
3. **REQUIRED SUB-SKILL:** Use `superpowers:brainstorming`. Persist `design-pending`, present the paper design, and wait for explicit approval. Then persist `implementing` and use `superpowers:test-driven-development` under `submissions/<paper-slug>/`.
4. Follow [submission-checklist.md](references/submission-checklist.md): distinguish inputs, computed outputs, and paper context; validate and persist `validated`.
5. **REQUIRED SUB-SKILL:** Use `superpowers:verification-before-completion` before deployment or success claims. Deploy a separate Space, verify exact SHA, persist `deployed`, refresh live state before submitting, then persist `submitted` and `judging`.
6. For judging, define finite maximum poll count and deadline; persist every observation. A pending verdict is not success. On either limit without a verdict, persist pending/`blocked`, write blocker and next check in `docs/HANDOFF.md`, return control, and do not claim completion. Only a claim-level verdict permits `complete`; one evidence-focused `judging` -> `improving` attempt is allowed.
7. After `complete`, archive to `idle` and repeat. Never reselect judged or historical papers or autonomously transition `blocked` -> `idle`.

## Compute And Pause Gates

- Autonomous GPU work of any kind is ineligible. Missing credentials and provisioning paid infrastructure pause the loop.
- Estimated or actual paid-API cost above USD 10 per paper is ineligible; stop before actual cost can exceed USD 10. Known unsafe execution is ineligible; unresolved safety ambiguity pauses selection.
- Before every pause, persist state and identifiers, then write `docs/HANDOFF.md` with blocker and exact next action; with unavailable tools, use the unperformed-write contract above.

Update state after each phase change and external mutation. Use same-phase updates for actual cost, polls, and external IDs. Inspect the authoritative CLI:

```bash
uv run python skills/icml-repro-loop/scripts/state.py --help
```
