---
name: icml-repro-loop
description: Use when selecting, reproducing, submitting, improving, or continuously processing papers for the ICML 2026 Agent Repro Challenge.
---

# ICML Reproduction Loop

Process one paper at a time. Only computed outputs support claims. Persist state until work ends or a user gate.

## Mandatory Response/Action Contract

Every response must name its phase and applicable items. Omission bars success.

- **Selection or design:** name live claim refresh, next state phase/fields, design presentation and approval wait, and continue/stop action.
- **Implementing or later:** name relevant live refresh, next state write, recorded design approval, and continue/stop action.
- **Tools unavailable:** propose and label these writes unperformed: `state/repro-loop.json: <phase and fields>` and `docs/HANDOFF.md: blocker <reason>; next action <action>`. Never claim either occurred.

## Required Workflow

1. Resume from state and `docs/HANDOFF.md`; refresh live catalog, claims, queues, and verdicts. Do not start with unresolved work or submission state.
2. Read [selection-rubric.md](references/selection-rubric.md). Check live state and primary artifacts. While idle, persist each ineligible candidate with `reject`; continue ranking. Compare the top three eligible, then select with nonempty `upstream_revision`, finite explicit `estimated_api_cost_usd`, and at least two unique `target_claims`. Stop after selection or a persisted exhausted pool.
3. **REQUIRED SUB-SKILL:** Use `superpowers:brainstorming`. Persist `design-pending`, present the paper design, and wait for explicit approval. Then persist `implementing` and use `superpowers:test-driven-development` under `submissions/<paper-slug>/`.
4. Follow [submission-checklist.md](references/submission-checklist.md): distinguish inputs, computed outputs, and paper context; validate and persist `validated`.
5. **REQUIRED SUB-SKILL:** Use `superpowers:verification-before-completion` before deployment or success claims. Deploy a separate Space, verify exact SHA, persist `deployed`, refresh live state before submitting, then persist `submitted` and `judging`.
6. Each fresh `judging` transition starts a round with positive integer `poll_limit` and aware ISO `poll_deadline`. Persist observations within that round's count/deadline. At either limit without a verdict, enter `blocked` with a blocker, update HANDOFF, and return control. Resume only to `blocked_from`; pending is not success.
7. Verdict claims must exactly match `target_claims`, using `verified`, `partial`, `inconclusive`, `contradicted`, or `unavailable`. The one `judging` -> `improving` attempt requires a verdict and `improvement_reason`; append both verdicts to history, and append/retain the final verdict on `complete`. Then archive to `idle`; never reselect judged/history papers.

## Compute And Pause Gates

- Autonomous GPU work of any kind is ineligible. Missing credentials and provisioning paid infrastructure pause the loop.
- Estimated or actual paid-API cost above USD 10 per paper is ineligible; stop before actual cost can exceed USD 10. Known unsafe execution is ineligible; unresolved safety ambiguity pauses selection.
- Before pausing, persist state/IDs and HANDOFF blocker/next action; if tools are unavailable, use the contract above. Never autonomously abandon. `blocked` -> `idle` requires user `abandon=true` and archives/cost-accounts.

Update state after each phase change and external mutation. Use same-phase updates for actual cost, polls, and external IDs. Inspect the authoritative CLI:

```bash
uv run python skills/icml-repro-loop/scripts/state.py --help
```
