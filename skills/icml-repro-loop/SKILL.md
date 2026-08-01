---
name: icml-repro-loop
description: Use when selecting, reproducing, submitting, improving, or continuously processing papers for the ICML 2026 Agent Repro Challenge.
---

# ICML Reproduction Loop

Maintain independently executable evidence for challenge papers. Only recomputed outputs support claims, and only immutable controller attestations support external lifecycle phases.

## Persistent Direct Dispatch Contract

A directly dispatched worker is a **persistent paper-owner worker** and a trusted controller. It runs this skill in a loop, owns one current paper at a time, and may use controller credentials for the exact lifecycle operations required by that paper.

The paper-owner worker must:

1. use `claim-next` to select or reclaim exactly one paper;
2. remain dedicated through implementation, publication, submission, and judging;
3. import the exact official verdict, call `release-paper --outcome scored`, and select the next paper; or
4. persist a genuine external blocker by fenced `transition-attempt` to
   `blocked` with nonempty `blocker` and `next_action`, call
   `release-paper --outcome blocked`, notify the root coordinator, and select
   the next paper.

An optional subordinate implementation subprocess is not the dispatched worker. It remains credential-free, worktree-scoped, and proposal-only.

Direct dispatch continues until explicitly stopped: score or genuine blocked release, then repeat. Submission, a subprocess exit, a healthy Space, and judging are events within the current iteration; none authorizes the persistent paper-owner worker to select another paper. A submitted or judging paper remains dedicated. Blocked release preserves the attempt as active and reclaimable by the same or another paper-owner worker. Reclamation retains its ID and history but requires a fresh assessed immutable snapshot and fresh fencing token.

Follow [paper-owner-loop.md](references/paper-owner-loop.md) for mandatory event reactions. Do not wait for a user status request between phases.

## Roles And Authority

The persistent paper-owner worker is the trusted controller and paper-owner controller for its current fenced attempt. It may use controller credentials to claim, validate, publish, submit, watch, correct, import the official verdict, and release that attempt. The paper-owner worker may publish only the exact validated source for its current fenced attempt. The non-negotiable rule is: controller credentials never enter Git, evidence, logs, or subordinate environments.

An optional **subordinate implementation subprocess** is an untrusted proposal producer, not the dispatched worker. It may inspect public sources and edit only its assigned paper worktree/project after the guard preflight passes. It is credential-free and cannot write coordinator state, use deployment credentials, deploy, submit, poll, import a verdict, merge, release an attempt, or claim any external phase. Full filesystem permissions do not transfer this authority.

### Subordinate implementation subprocess recipe

1. Write only the assigned worktree and the contract's paper project.
2. Return a commit, commands, evidence paths, and concerns as a proposal, never authority.
3. Never claim or perform external lifecycle phases.

Launch a subordinate implementation subprocess only through `state.py run-worker` after `preflight_runtime` proves that outside-worktree writes and synthetic credential-file reads are denied. The guarded launch constructs a sanitized environment and records the actual child-process boundary. An unenforceable runtime may receive only a read-only research contract.

### Persistent paper-owner worker recipe

The persistent paper-owner worker stays active after `run-worker`. It inspects the proposal, validates or issues a concrete correction, publishes, submits, watches, improves, imports the exact verdict, releases the scored or blocked attempt, and begins the next iteration with `claim-next`.

Use the dedicated command that owns each assertion:

| Assertion | Required persistent paper-owner worker command |
| --- | --- |
| migrated schema-v3 attempt has fresh claim and design authority | `reconcile-legacy-attempt` |
| local evidence passed | `attest-validation` |
| exact Space was published and is healthy | `publish-deployment` |
| exact tagged Space is visible live | `refresh-live`, then `attest-submission` |
| bounded judging began | `watch-attempt` |
| one pending observation occurred | `record-poll` |
| exact official verdict exists | `refresh-live`, then `sync-verdict` |
| terminal scored or blocked iteration released | `release-paper` |
| historical completion has authority | `audit-authority` |

Every external-phase transition must name its immutable controller attestation ID. If the attestation is absent, stop or block and state that the corresponding writes were unperformed.

`validation-rejected` is an event, not a phase: before an official verdict, the attempt stays `implementing`, receives exact correction-contract defects, and relaunches through normal fenced `run-worker`. An official correctable deficiency is different: import it with `sync-verdict --improvement-reason REASON`, which preserves that exact verdict and transitions the attempt to `improving`; then `run-worker` derives correction telemetry from the phase.

## Mandatory Response/Action Contract

Every response names each materially affected attempt ID, paper, phase, owner, relevant immutable snapshot ID, next action, and blocker. Omission bars success.

- A subordinate implementation subprocess result also names its assigned worktree, commit or uncommitted status, commands run, evidence paths, and concerns, and labels itself a proposal.
- The persistent paper-owner worker names every explicit shard/index write, independent design approval, attestation ID, and continue/stop action.
- If tools are unavailable, name every affected attempt and label shard, index, Hub, and verdict writes unperformed. Never imply they occurred.

## Required Persistent Paper-Owner Loop

For every iteration:

1. Inspect `state/repro-loop.json`. For schema-v3, run only `migrate-v6 --dry-run` and stop until the controller explicitly authorizes the real migration; no schema-v6 lifecycle command is valid before it. After migration, resume the schema-v6 index and every named shard.
2. Run raw `refresh-live`, inspect its immutable result through `show-snapshot`, and assess pinned `challenge.json` candidates from primary artifacts. Bare challenge metadata never supplies feasibility, score, cost, targets, or an upstream pin. Write assessment JSON following [selection-rubric.md](references/selection-rubric.md), bind each target to exact challenge text and SHA-256, then run `refresh-live --assessments-json PATH`. Revision drift requires a new raw refresh and assessment.
3. Endgame saturation rule: count publish-ready lanes (phase `validated`
   plus `blocked` with `blocked_from: validated`). While that count is at
   or above the daily Hugging Face Space-creation quota (20), selecting a
   NEW paper is forbidden — the backlog already saturates publishing
   capacity. Spend the iteration instead on, in priority order: (a)
   reclaiming and publishing a publish-ready lane if quota remains today;
   (b) an improvement cycle on a judged attempt with a correctable
   zero/toy/inconclusive verdict (real computed evidence, rendered into
   the served pages); (c) preparing improvements in an isolated clone for
   a lane whose submission is pending judgment, without touching its live
   Space. Before selecting new work, inspect every active released blocked attempt.
   If its recorded blocker is resolved or its `next_action` is actionable,
   explicitly reclaim the highest-priority eligible attempt with
   `claim-next --reclaim-attempt-id ATTEMPT`, the fresh assessed immutable
   snapshot, and a fresh fencing token. If none is ready, leave unresolved
   blockers reclaimable and use ordinary `claim-next` to select new work. This
   routing is mandatory: ordinary `claim-next` must not auto-reclaim unresolved
   blocked attempts. A
   blocked reclamation or migrated legacy attempt retains its history; a
   migrated schema-v3 attempt also requires `reconcile-legacy-attempt` with
   the fresh snapshot, distinct design author/reviewer identities, tracked
   design, and approval reference.
4. Persist the paper-specific design through `record-design` and require a different reviewer through `review-design`. Pin every upstream revision. Create the guarded subordinate contract only if implementation assistance is needed; otherwise implement directly as the trusted paper owner.
5. Treat a subordinate result as a proposal. Inspect its diff, use `attest-validation`, and issue exact correction findings on rejection. Validation or local JSON alone never advances the lifecycle.
6. Work in worktrees under the workspace root (`.worktrees/<name>`) and put staging clones and scratch files under `scratch/<name>` (a workspace symlink onto the large `/data` drive), never in `/tmp` — `/tmp` is wiped on boot and invisible to `candidate-census`. Before publishing, commit the exact attested source to the shared repository on `main` under `submissions/<slug>/` with its `.gitignore` negation block (`!submissions/<slug>/` and `!submissions/<slug>/**`); staging that exists only outside the repository is one reboot from forcing reimplementation. Use `publish-deployment` for the dedicated Space. Require the allowlisted owner, exact paper and challenge tags, exact attested SHA, and `RUNNING` runtime. Verify the live Space serves every project file, especially every `pages/*.md` — judges read only the served logbook. Then take a fresh assessed snapshot and use `attest-submission`. Space existence does not prove submission.
7. Use `watch-attempt` with a finite positive poll limit and aware deadline, recording pending observations through `record-poll`. Submitted and judging states remain dedicated to this attempt. On an official correctable deficiency, preserve it with `sync-verdict --improvement-reason REASON`, correct, redeploy, resubmit, and watch the same attempt.
8. Use `sync-verdict` only with a fresh immutable snapshot. It verifies exact paper, Space, deployed SHA, verdict dataset revision, judged timestamp, and claim bindings. Preserve `verified`, `falsified`, `toy`, and `inconclusive` exactly. Then call `release-paper --outcome scored` and return to step 2.
9. If a genuine deadline, authority, feasibility, or other external blocker
   prevents progress, use fenced `transition-attempt ... blocked` with
   nonempty `blocker` and `next_action` to persist the exact phase,
   observation, next action, and unperformed writes. Then release the blocked
   attempt with `release-paper --outcome blocked`, notify the root coordinator,
   and return to step 2. Never abandon automatically.

Run read-only `score-report` after each subordinate exit, validation or deployment outcome, and official verdict refresh. Keep official verdict points, pending estimates, rank observations, runnable capacity, and actual telemetry separate. Follow [submission-checklist.md](references/submission-checklist.md) for the exact evidence and report contracts.

All attempt mutations require explicit `--attempt-id`, `--owner`, and `--fencing-token`. Renew the current two-hour lease before expiry. A successor uses `claim-attempt` only with the exact predecessor token; never guess it or reclaim a live lease.

## Authority Red Flags

Stop when any of these appears:

- a subordinate implementation subprocess edits `state/`, this skill, another submission, or a coordinator document;
- a launch uses `--dangerously-skip-permissions`, `--dangerously-bypass-approvals-and-sandbox`, danger-full-access, or `--add-dir`;
- a subordinate environment inherits `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `GH_TOKEN`, a credential helper, or an existing Hugging Face cache;
- a lifecycle claim lacks its dedicated controller attestation;
- verdict data was typed by a caller, inferred from evidence, normalized to a better status, or copied from paper prose;
- a commit stages any `state/` path: live coordinator state is never git-tracked; durability comes from the sharded store, and tracked copies get reverted by git operations, destroying claims and leases;
- a branch is moved to one of its own ancestors (`git reset --hard <older-sha>`, `git checkout -B`): in a shared checkout this discards other workers' commits.

Use these counters verbatim:

- “The evidence proves all claims.” **Evidence is not the official verdict.**
- “I simulated the judge result.” **Simulations cannot enter judgment state.**
- “The Space exists, so it was submitted.” **Require exact live snapshot observation.**
- “The expected verdict is obvious.” **Wait for and import the official record.**
- “Full permissions authorize state edits.” **Permissions do not grant authority.**

## Compute And Pause Gates

- Autonomous GPU work is ineligible.
- Estimated or actual metered API cost above USD 10 per paper is ineligible. Codex and Antigravity subscriptions record USD 0.00.
- Missing controller credentials, paid infrastructure approval, unsafe execution, or unenforceable subordinate isolation blocks the current iteration.
- Use fenced `transition-attempt` to persist nonempty `blocker` and
  `next_action`, release the blocked attempt reclaimably, and never
  autonomously abandon; only an explicit user `abandon=true` may archive it.

Inspect the authoritative CLI rather than relying on remembered signatures:

```bash
uv run python skills/icml-repro-loop/scripts/state.py --help
```

`show-*`, `list-attempts`, `candidate-census`, `score-report`, and `audit-authority` without `--repair` are read-only. `refresh-live`, `claim-next`, `scheduler-pass` (an available general scheduling command, not direct-dispatch routing), `run-worker`, lease/design/lifecycle commands, controller attestations, `sync-verdict`, `release-paper`, and `audit-authority --repair` mutate local or external authority; only the controller may run them for its current fenced attempt. Here, the controller is the persistent paper-owner worker.
