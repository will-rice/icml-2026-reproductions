---
name: icml-repro-loop
description: Use when selecting, reproducing, submitting, improving, or continuously processing papers for the ICML 2026 Agent Repro Challenge.
---

# ICML Reproduction Loop

Maintain up to 20 independent paper attempts. Only recomputed outputs support
claims, and only controller attestations support external lifecycle phases.

## Worker Or Controller?

Decide before any tool call.

- A **paper worker** is an untrusted proposal producer. It may inspect public
  sources and edit one assigned paper worktree after the controller guard
  passes. It cannot write coordinator state, use deployment credentials,
  deploy, submit, poll, import a verdict, merge, or claim a challenge outcome.
- The **controller** owns the schema-v6 index and shards, leases, validation,
  Hub mutations, live snapshots, official verdict import, quarantine, and
  integration.

If the role is not explicitly controller, use the worker role. Full filesystem
permissions do not change the role.

### Worker recipe

1. **write only the assigned worktree** and only the contract's paper project;
2. return commit, commands, evidence paths, and concerns as a **proposal, never authority**;
3. never claim or perform external phases.

An implementation worker must be launched through `worker_guard.py` after
`preflight_runtime` proves outside-worktree writes and synthetic
credential-file reads are denied. `launch_spec` constructs the command and
sanitized environment. An unenforceable runtime may receive only a read-only
research contract.

### Controller recipe

The controller must use the dedicated command that owns each assertion:

| Assertion | Required controller command |
| --- | --- |
| migrated schema-v3 attempt has fresh claim and design authority | `reconcile-legacy-attempt` |
| local evidence passed | `attest-validation` |
| exact Space was published and is healthy | `publish-deployment` |
| exact tagged Space is visible live | `refresh-live`, then `attest-submission` |
| bounded judging began | `watch-attempt` |
| one pending observation occurred | `record-poll` |
| exact official verdict exists | `refresh-live`, then `sync-verdict` |
| historical completion has authority | `audit-authority` |

Every external-phase transition must name its immutable controller attestation
ID. If the attestation is absent, stop or block and state that the corresponding
writes were unperformed.

## Mandatory Response/Action Contract

Every response names each materially affected attempt ID, paper, phase, owner,
relevant immutable snapshot ID, next action, and blocker. Omission bars
success.

- **Worker:** also name assigned worktree, commit or uncommitted status,
  commands run, evidence paths, and concerns. Label the result a proposal.
- **Controller:** name every explicit shard/index write, independent design
  approval, attestation ID, and continue/stop action.
- **Tools unavailable:** name every affected attempt and label shard, index,
  Hub, and verdict writes unperformed. Never imply they occurred.

## Required Controller Workflow

1. Resume the schema-v6 index and every named shard. Run raw `refresh-live`,
   then inspect the immutable result through `show-snapshot`. Bare challenge
   metadata never supplies feasibility, score, cost, targets, or an upstream
   pin.
2. Inspect primary artifacts and write assessment JSON following
   [selection-rubric.md](references/selection-rubric.md). Bind each target to
   exact live challenge text and SHA-256. Run
   `refresh-live --assessments-json PATH`; revision drift requires a new raw
   refresh and assessment.
   A schema-v3 migrated attempt cannot advance on its legacy
   `design_approved` boolean alone. Claim it with predecessor token `0`, then
   use `reconcile-legacy-attempt` once with this fresh assessed snapshot,
   distinct design author/reviewer identities, the tracked design path, and an
   explicit approval reference.
3. Run `scheduler-pass` with that assessed snapshot. Persist one paper-specific
   design with `record-design`, then require a different reviewer through
   `review-design`.
4. Create one controller-authored worker contract naming the attempt, paper,
   absolute isolated worktree, `submissions/<paper>/` path, and mode. Strip
   Hub/GitHub credentials, disable Hugging Face implicit-token loading, run the
   guard preflight, and launch only the constructed spec.
5. Treat the worker result as a proposal. Review its diff and use
   `attest-validation`; worker-reported tests or local JSON cannot advance the
   phase.
6. Use `publish-deployment`. Require the allowlisted owner, dedicated Space,
   exact paper and challenge tags, exact attested SHA, and `RUNNING` runtime.
   `CONFIG_ERROR`, wrong SHA, missing tag, or a healthy-looking UI is not a
   deployment attestation.
7. Fetch a new assessed snapshot and use `attest-submission`. Space existence
   does not prove submission. Reject a duplicate paper or conflicting canonical
   attempt.
8. Use `watch-attempt` with a finite positive poll limit and aware deadline.
   Persist observations through `record-poll`. At either bound without a
   verdict, persist a blocker and refill capacity; pending is not success.
9. Use `sync-verdict` with only a fresh immutable snapshot ID. It verifies exact
   paper, Space, deployed SHA, verdict dataset revision, judged timestamp, and
   claim bindings. Preserve official `verified`, `falsified`, `toy`, and
   `inconclusive` exactly.
10. Before trusting pre-hardening history, run `audit-authority` read-only,
    inspect every decision, then use `--repair` to quarantine unsupported
    completions. Repair never modifies external Spaces.

All attempt mutations require explicit `--attempt-id`, `--owner`, and
`--fencing-token`. Renew the current two-hour lease before expiry. A successor
uses `claim-attempt` only with the exact predecessor token; never guess it or
reclaim a live lease.

## Authority Red Flags

Stop when any of these appears:

- a worker edits `state/`, this skill, another submission, or a coordinator
  document;
- a launch uses `--dangerously-skip-permissions`,
  `--dangerously-bypass-approvals-and-sandbox`, danger-full-access, or
  `--add-dir`;
- a worker inherits `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `GH_TOKEN`, a
  credential helper, or an existing Hugging Face cache;
- a lifecycle claim lacks its dedicated controller attestation;
- verdict data was typed by a caller, inferred from evidence, normalized to a
  better status, or copied from paper prose.

Use these counters verbatim:

- “The evidence proves all claims.” **Evidence is not the official verdict.**
- “I simulated the judge result.” **Simulations cannot enter judgment state.**
- “The Space exists, so it was submitted.” **Require exact live snapshot observation.**
- “The expected verdict is obvious.” **Wait for and import the official record.**
- “Full permissions authorize state edits.” **Permissions do not grant authority.**

## Compute And Pause Gates

- Autonomous GPU work is ineligible.
- Estimated or actual metered API cost above USD 10 per paper is ineligible.
  Codex and Antigravity subscriptions record USD 0.00.
- Missing credentials, paid infrastructure approval, unsafe execution, or
  unenforceable worker isolation pauses the affected attempt.
- Persist the blocker and next action. Never autonomously abandon; only an
  explicit user `abandon=true` may archive a blocked attempt.

Follow [submission-checklist.md](references/submission-checklist.md) for the
evidence and controller handoff. Inspect the authoritative CLI rather than
relying on remembered signatures:

```bash
uv run python skills/icml-repro-loop/scripts/state.py --help
```
