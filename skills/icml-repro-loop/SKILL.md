---
name: icml-repro-loop
description: Use when selecting, reproducing, submitting, improving, or continuously processing papers for the ICML 2026 Agent Repro Challenge.
---

# ICML Reproduction Loop

Maintain up to 20 independent runnable paper attempts. Only computed outputs support claims. Persist every attempt until completion, an explicit user abandonment, or a durable blocker.

## Mandatory Response/Action Contract

Every response must name every materially affected attempt ID, paper, phase, owner, next action, and blocker. Omission bars success.

- **Selection or design:** name immutable snapshot ID, scheduler result, attempt IDs, design authors and different reviewers, next writes, and continue/stop actions.
- **Implementing or later:** name relevant snapshot ID, each explicit attempt write, recorded independent design approval, and continue/stop action.
- **Tools unavailable:** name every affected attempt and label its shard/index
  writes unperformed. Never claim they occurred.

## Required Workflow

1. Resume the schema-v6 index and every named attempt shard. Run raw `refresh-live`, then use read-only `show-snapshot` to inspect its exact challenge revision, current `challenge.json` papers, and extracted claims. Bare challenge metadata never supplies feasibility, score, cost, target selection, or an upstream pin.
2. In parallel, inspect primary artifacts and write explicit assessment JSON described by [selection-rubric.md](references/selection-rubric.md), bound to the raw snapshot's challenge revision. Run `refresh-live --assessments-json PATH`. A newly fetched revision mismatch fails explicitly; restart from raw refresh rather than reusing stale assessments.
3. `refresh-live` is the sole network-aware state command. Raw and assessed forms record exact challenge/verdict dataset SHAs, both challenge files, authoritative verdict Space keys, and all current `paper-*` Space identities in immutable content-addressed snapshots. Assessed form also verifies and records the canonical assessment hash. Only tagged Spaces without a verdict key are pending.
4. Run `scheduler-pass` with the assessed snapshot ID. Its JSON assignments expose each `attempt_id`, `paper_id`, writer `owner`, and `fencing_token`. Use those values to transition each new attempt from `selected` to `design-pending`; it admits only matching assessed candidates and refills toward 20 runnable paper attempts.
5. Give one authoritative implementation agent the current writer identity. It must call `renew-attempt` before the fixed two-hour lease expires. If the lease expired or was released, a successor calls `claim-attempt` with the exact predecessor token and its new owner identity; the incremented token permanently fences the stale owner. Never reclaim a live lease or guess its predecessor token.
6. **REQUIRED SUB-SKILL:** Use `superpowers:brainstorming` for each admitted paper. Call fenced `record-design` with its committed design and author, then have a different agent call `review-design`. Approval advances only that attempt to `implementing`; rejection revises only that design. Use `superpowers:test-driven-development` under its isolated `submissions/<paper-slug>/` worktree.
7. Follow [submission-checklist.md](references/submission-checklist.md): distinguish inputs, computed outputs, and paper context; validate and persist `validated`.
8. **REQUIRED SUB-SKILL:** Use `superpowers:verification-before-completion` before deployment or success claims. Deploy a separate Space, verify exact SHA, persist `deployed`, refresh live state with the explicit assessment file before submitting, then persist `submitted` and `judging`.
9. `watch-attempt` starts an independently fenced judgment with positive `poll_limit` and aware `poll_deadline`. Persist observations through `record-poll` and exact-source verdicts through `record-verdict`, preserving the Task 5 signatures. At either bound without a verdict, persist the blocker and next action in that attempt shard and refill its capacity. Resume only to `blocked_from`; pending is not success.
10. Verdict claims must exactly match `target_claims`, using `verified`, `partial`, `inconclusive`, `contradicted`, or `unavailable`. The one `judging` -> `improving` attempt requires a verdict and `improvement_reason`; append both verdicts to history, and append/retain the final verdict on `complete`. Then archive to `idle`; never reselect judged/history papers.

## Compute And Pause Gates

- Autonomous GPU work of any kind is ineligible. Missing credentials and provisioning paid infrastructure pause the loop.
- Estimated or actual paid-API cost above USD 10 per paper is ineligible; stop before actual cost can exceed USD 10. Codex and Antigravity subscriptions reserve and record USD 0.00; only genuinely metered external services consume reservations. Known unsafe execution is ineligible; unresolved safety ambiguity pauses selection.
- Before pausing, persist IDs, blocker, and next action in the affected attempt
  shard; if tools are unavailable, use the contract above. Never autonomously
  abandon. `blocked` -> `idle` requires user `abandon=true` and
  archives/cost-accounts.

Every attempt-mutating command requires `--attempt-id`, `--owner`, and `--fencing-token`; the CLI reconstructs and validates that persisted lease. For `claim-attempt`, the token is the expected predecessor (zero only when no lease exists) and owner names the successor. `renew-attempt` requires the exact current live identity. Both return the authoritative owner, token, acquisition, and expiry fields; their work TTL is always two hours and is not caller-configurable. Never infer a current attempt. Legacy schema-v3 mutation commands are not public. `refresh-live` and `scheduler-pass` are coordinator operations rather than attempt mutations. Inspect the authoritative CLI:

```bash
uv run python skills/icml-repro-loop/scripts/state.py --help
```
