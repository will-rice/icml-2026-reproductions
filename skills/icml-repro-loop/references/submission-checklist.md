# Evidence And Controller Handoff Checklist

A checked item needs an artifact, command result, controller attestation, or
exact live observation. Intention and worker self-report are not evidence.

## Paper-Owner Completion Gate

- [ ] The directly dispatched persistent paper-owner worker owns one current
  paper at a time and repeats only after exact verdict import or a
  genuine persisted blocker.
- [ ] A subordinate implementation subprocess exit triggered immediate diff review and fresh
  controller validation without a user status prompt.
- [ ] A rejected validation produced exact correction findings and a guarded
  relaunch on the same attempt.
- [ ] The paper owner continued through `publish-deployment`,
  `attest-submission`, `watch-attempt`, and `sync-verdict`.
- [ ] Pending queue state was watched rather than treated as evidence failure.
- [ ] `submitted` and `judging` remained dedicated to their paper and did not
  free owner capacity; release occurred only after exact verdict import
  or a genuine persisted blocker.
- [ ] Each iteration began with `claim-next` using a fresh assessed immutable
  snapshot and ended with `release-paper --outcome scored` or
  `release-paper --outcome blocked`.
- [ ] A `paper-owner-released` record names the attempt, owner, fence,
  immutable snapshot, outcome, and exact verdict or persisted blocker.
- [ ] A blocked attempt remains reclaimable; its later owner uses a fresh
  assessed immutable snapshot and fresh fence without changing its history.
- [ ] Before `release-paper --outcome blocked`, the paper owner used fenced
  `transition-attempt` to enter `blocked` with nonempty `blocker` and
  `next_action`, then notified the root coordinator and released reclaimably.

## Subordinate Implementation Subprocess Boundary

- [ ] The controller contract names one attempt, paper, absolute worktree,
  `submissions/<paper>/` project path, and `implementation` or `research` mode.
- [ ] An optional subordinate implementation subprocess was launched only through fenced
  `state.py run-worker` after its runtime preflight passed.
- [ ] The subordinate environment contains no `HF_TOKEN`,
  `HUGGING_FACE_HUB_TOKEN`, `GH_TOKEN`, credential helper, inherited Hugging
  Face cache, or implicit-token loading.
- [ ] The subordinate changed only its assigned worktree/project and returned a
  commit, commands, evidence paths, and concerns as a proposal, never
  authority.
- [ ] A runtime that cannot enforce isolation received only a read-only
  research contract.
- [ ] Subprocess queue time comes from queued/launched UTC observations; subprocess
  process time comes from launched/exited monotonic counters. Git revisions
  identify inputs and outputs and are never runtime estimates.
- [ ] A subordinate launched from `implementing` records
  `work_kind="implementation"`; one launched from `improving` records
  `work_kind="correction"`. Validation and deployment are separate controller
  stage intervals.

## Upstream And Claims

- [ ] Record the exact paper identifier and challenge snapshot.
- [ ] Pin every upstream repository, dataset, checkpoint, prediction, and
  release to an immutable revision and record downloaded-file hashes.
- [ ] Record source URLs, acquisition commands, licenses, and which artifact
  supports each target.
- [ ] Bind each target slug to exact challenge claim text and its SHA-256.
- [ ] Define expected observations and failing tests before implementation.

## Evidence Proposal

- [ ] Persist a paper-specific design through `record-design`; approval comes
  from a different reviewer through `review-design`.
- [ ] Observe each evidence test fail before implementation.
- [ ] Keep inputs, computed outputs, and paper-reported context distinct. Only
  code-computed output may be called reproduced evidence.
- [ ] Emit deterministic machine-readable claim results with provenance,
  observations, tolerances, and status.
- [ ] Mark inaccessible artifacts or untestable claims `unavailable`; never
  substitute README values, screenshots, or assertions.
- [ ] Keep the evidence pipeline independently executable from a clean
  environment.

## Controller Validation

- [ ] Review the subordinate implementation subprocess diff for cross-paper edits, coordinator paths,
  credentials, mutable references, generated caches, and unrelated changes.
- [ ] Use `attest-validation` with the approved manifest. The controller—not a
  worker string—checks clean Git identity/path scope, runs the evidence
  command, submission pytest, root pytest, skill validation, and pre-commit,
  and rechecks the live fence.
- [ ] Record the immutable validation attestation ID. Without it, validation
  and all later writes are unperformed.

## Durable Source In The Shared Repository

- [ ] After validation and before publication, commit the exact validated
  source to the shared repository on `main` under `submissions/<slug>/`.
  Ephemeral staging worktrees under `/tmp` do not survive a reboot; a
  validated source that exists only in `/tmp` is one restart away from
  forcing full reimplementation.
- [ ] `.gitignore` ignores `submissions/*` by default. Every new submission
  commit must include its negation block — `!submissions/<slug>/` and
  `!submissions/<slug>/**` plus per-submission cache re-ignores — or the
  source is invisible to git and lost on resets.
- [ ] When the shared checkout is not on `main`, land the commit through a
  linked worktree under `.worktrees/` instead of switching the shared HEAD.
  Never rewind or force-move shared branches.
- [ ] Record repo-relative paths (never `/tmp` paths) in `next_action` and
  blocker text so a reclaiming owner can resume from the repository.
- [ ] Create every validation worktree, staging clone, and scratch checkout
  under the workspace root (`.worktrees/` for linked worktrees), never in
  `/tmp`: systemd wipes `/tmp` on boot, `candidate-census` polices only
  workspace worktrees, and `/tmp` staging is how sixty gigabytes of
  validated sources nearly vanished. Remove your worktree when the
  iteration releases.

## Deployment And Submission

- [ ] Use `publish-deployment` for one dedicated Space. Require the allowlisted
  owner, exact `paper-<paper_id>` and `icml2026-repro` tags, exact validation
  source, exact remote SHA, and `RUNNING` runtime.
- [ ] Treat a missing tag, wrong SHA/owner, `CONFIG_ERROR`, build in progress,
  or healthy UI without exact identity as unproven.
- [ ] Immediately before submission observation, run
  `refresh-live --assessments-json PATH`; revision drift requires a new raw
  refresh and assessment.
- [ ] Use `attest-submission` with that snapshot. Stop for a duplicate paper,
  conflicting attempt, missing tag, missing Space, or wrong revision.
- [ ] Record deployment and submission attestation IDs. Space existence or an
  invented submission ID is never authority.
- [ ] After publication, list the live Space files and require every project
  file — especially every `pages/*.md` — to be present. Partial uploads have
  produced summary-only logbooks that judges score zero; the judge reads only
  what the Space actually serves.

## Judging And Official Verdict

- [ ] Use `watch-attempt` with the exact attempt/owner/fence, positive
  `poll_limit`, and aware `poll_deadline`; record its authority attestation ID.
- [ ] Use `record-poll` only for bounded pending observations. At the limit or
  deadline without a verdict, block with the next action and leave completion
  writes unperformed.
- [ ] Refresh live state and use `sync-verdict --snapshot-id ID`. The command
  accepts no caller verdict, status, or source revision.
- [ ] Require exact paper ID, Space ID, attested SHA, verdict revision, judged
  timestamp, and selected claim text/hash.
- [ ] Copy official claim text, evidence, and status exactly. The only official
  statuses are `verified`, `falsified`, `toy`, and `inconclusive`; never promote
  `toy` or `inconclusive`.
- [ ] Evidence is not the official verdict. Simulations cannot enter judgment
  state. Require exact live snapshot observation. Wait for and import the
  official record. Permissions do not grant authority.
- [ ] `submitted` and `judging` remain dedicated states: watch the current
  paper and do not select another. A scored or genuinely blocked iteration
  releases only through `release-paper` before the next `claim-next`.
- [ ] A genuinely blocked iteration first uses fenced `transition-attempt` to
  record nonempty `blocker` and `next_action`, then calls
  `release-paper --outcome blocked` and remains reclaimable.

## Census And Score Report

- [ ] Run `candidate-census` only against an immutable raw snapshot. Its
  `authority="research-required"` rows and candidate-slug project matches are
  discovery inputs, not assessments.
- [ ] Run `score-report` after every worker exit, validation/deployment
  outcome, and verdict refresh. It is read-only: it must not recover, repair,
  update, or transition coordinator state.
- [ ] Keep `official` snapshot-derived verdict points separate from
  `pending_judgment` estimates and `candidate_queue` estimates. Never add
  estimated points to the official total.
- [ ] Read capacity from `capacity.max_runnable`, `runnable`, and `idle`.
  Telemetry reports actual `worker_queue_seconds`, `worker_process_seconds`,
  `validation_seconds`, `deployment_seconds`, and
  `first_launch_to_submission_seconds`; incomplete intervals are `null`, not
  phase- or Git-derived guesses. Session counts distinguish implementation
  from correction, and judged point rates use only complete measured
  intervals.
- [ ] Supply a rank observation only as JSON with exactly this schema; omit the
  option to report `rank_observation=null`:

```json
{
  "observed_at": "2026-07-27T12:00:00+00:00",
  "source_url": "https://example.org/leaderboard",
  "username": "wrice",
  "points": 12,
  "rank": 7
}
```

The timestamp must be timezone-aware, the URL HTTP(S), `rank` a positive
integer, and `username` and integer `points` must exactly match the report's
requested user and snapshot-derived official points. The report does not
scrape or infer rank.

## Historical Repair

- [ ] Run `audit-authority PATH --snapshot-id ID` before trusting legacy
  completion history.
- [ ] Inspect exact snapshot/index/shard hashes and every proposed phase.
- [ ] Use `audit-authority ... --repair` only after review. Confirm unsupported
  originals are byte-preserved under quarantine, removed from history,
  restored blocked at their last proven phase, and require a fresh lease.
- [ ] Re-run repair and confirm it is idempotent. External Spaces remain
  untouched.

## Controller CLI Examples

Every fenced attempt command also takes `--attempt-id ATTEMPT --owner OWNER
--fencing-token TOKEN`.

```bash
uv run python skills/icml-repro-loop/scripts/state.py claim-next state/repro-loop.json --snapshot-id SNAPSHOT --owner OWNER
uv run python skills/icml-repro-loop/scripts/state.py run-worker state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --runtime codex --model MODEL --worktree /ABSOLUTE/WORKTREE --contract /ABSOLUTE/WORKTREE/.superpowers/worker-contract.json
uv run python skills/icml-repro-loop/scripts/state.py attest-validation state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --manifest validation-manifest.json
uv run python skills/icml-repro-loop/scripts/state.py publish-deployment state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --space-id OWNER/SPACE --source-dir submissions/PAPER
uv run python skills/icml-repro-loop/scripts/state.py attest-submission state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --snapshot-id SNAPSHOT
uv run python skills/icml-repro-loop/scripts/state.py watch-attempt state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --poll-limit 12 --poll-deadline 2026-07-25T18:00:00+00:00
uv run python skills/icml-repro-loop/scripts/state.py sync-verdict state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --snapshot-id SNAPSHOT
uv run python skills/icml-repro-loop/scripts/state.py release-paper state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --outcome scored
uv run python skills/icml-repro-loop/scripts/state.py transition-attempt state/repro-loop.json blocked --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --updates-json '{"blocker":"EXTERNAL_BLOCKER","next_action":"NEXT_ACTION"}'
uv run python skills/icml-repro-loop/scripts/state.py release-paper state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --outcome blocked
uv run python skills/icml-repro-loop/scripts/state.py audit-authority state/repro-loop.json --snapshot-id SNAPSHOT
uv run python skills/icml-repro-loop/scripts/state.py score-report state/repro-loop.json --snapshot-id SNAPSHOT --username wrice --rank-observation-json state/wrice-rank-observation.json
```

```text
run-worker
  -> inspect worker-exited telemetry
  -> attest-validation OR correction run-worker
  -> publish-deployment
  -> refresh-live + attest-submission
  -> watch-attempt + record-poll
  -> improvement loop OR sync-verdict
```

No arrow in this handoff is driven by a user status question.

The next `claim-next` occurs only after `sync-verdict` followed by scored
release, or after fenced `transition-attempt` records nonempty `blocker` and
`next_action` for a genuine blocker followed by reclaimable blocked release.
It must not select while `submitted` or `judging`.

## Resume-First Routing Example

Inspect active attempts and each released blocked attempt before selecting new
work:

```bash
uv run python skills/icml-repro-loop/scripts/state.py list-attempts state/repro-loop.json
uv run python skills/icml-repro-loop/scripts/state.py show-attempt state/repro-loop.json --attempt-id BLOCKED_ATTEMPT
```

Run exactly one of the two `claim-next` commands. If the recorded blocker is
resolved or `next_action` is actionable, reclaim the highest-priority eligible
blocked attempt explicitly:

```bash
uv run python skills/icml-repro-loop/scripts/state.py claim-next state/repro-loop.json --snapshot-id SNAPSHOT --owner OWNER --reclaim-attempt-id BLOCKED_ATTEMPT
```

If every recorded blocker remains unresolved, leave those attempts
reclaimable and select new work without a reclaim target:

```bash
uv run python skills/icml-repro-loop/scripts/state.py claim-next state/repro-loop.json --snapshot-id SNAPSHOT --owner OWNER
```

Ordinary `claim-next` must not auto-reclaim unresolved blocked attempts.

Read-only/reporting commands are `list-attempts`, `show-attempt`,
`show-snapshot`, `candidate-census`, `score-report`, and `audit-authority`
without `--repair`. `refresh-live` writes immutable snapshots;
`claim-next`, `scheduler-pass`, `run-worker`, lease/design/transition commands,
lifecycle attestations, `sync-verdict`, `release-paper`, and
`audit-authority --repair` mutate coordinator or external authority and are
persistent-paper-owner-controller-only.
