# Evidence And Controller Handoff Checklist

A checked item needs an artifact, command result, controller attestation, or
exact live observation. Intention and worker self-report are not evidence.

## Worker Boundary

- [ ] The controller contract names one attempt, paper, absolute worktree,
  `submissions/<paper>/` project path, and `implementation` or `research` mode.
- [ ] An implementation worker was launched only through fenced
  `state.py run-worker` after its runtime preflight passed.
- [ ] The worker environment contains no `HF_TOKEN`,
  `HUGGING_FACE_HUB_TOKEN`, `GH_TOKEN`, credential helper, inherited Hugging
  Face cache, or implicit-token loading.
- [ ] The worker changed only its assigned worktree/project and returned a
  commit, commands, evidence paths, and concerns as a proposal, never
  authority.
- [ ] A runtime that cannot enforce isolation received only a read-only
  research contract.
- [ ] Worker queue time comes from queued/launched UTC observations; worker
  process time comes from launched/exited monotonic counters. Git revisions
  identify inputs and outputs and are never runtime estimates.
- [ ] A worker launched from `implementing` records
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

- [ ] Review the worker diff for cross-paper edits, coordinator paths,
  credentials, mutable references, generated caches, and unrelated changes.
- [ ] Use `attest-validation` with the approved manifest. The controller—not a
  worker string—checks clean Git identity/path scope, runs the evidence
  command, submission pytest, root pytest, skill validation, and pre-commit,
  and rechecks the live fence.
- [ ] Record the immutable validation attestation ID. Without it, validation
  and all later writes are unperformed.

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
- [ ] Judging and blocked attempts release runnable implementation capacity;
  refill through a bounded `scheduler-pass` without waiting for unrelated
  verdicts or blockers.

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
uv run python skills/icml-repro-loop/scripts/state.py run-worker state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --runtime codex --model MODEL --worktree /ABSOLUTE/WORKTREE --contract /ABSOLUTE/WORKTREE/.superpowers/worker-contract.json
uv run python skills/icml-repro-loop/scripts/state.py attest-validation state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --manifest validation-manifest.json
uv run python skills/icml-repro-loop/scripts/state.py publish-deployment state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --space-id OWNER/SPACE --source-dir submissions/PAPER
uv run python skills/icml-repro-loop/scripts/state.py attest-submission state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --snapshot-id SNAPSHOT
uv run python skills/icml-repro-loop/scripts/state.py watch-attempt state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --poll-limit 12 --poll-deadline 2026-07-25T18:00:00+00:00
uv run python skills/icml-repro-loop/scripts/state.py sync-verdict state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --snapshot-id SNAPSHOT
uv run python skills/icml-repro-loop/scripts/state.py audit-authority state/repro-loop.json --snapshot-id SNAPSHOT
uv run python skills/icml-repro-loop/scripts/state.py score-report state/repro-loop.json --snapshot-id SNAPSHOT --username wrice --rank-observation-json state/wrice-rank-observation.json
```

Read-only/reporting commands are `list-attempts`, `show-attempt`,
`show-snapshot`, `candidate-census`, `score-report`, and `audit-authority`
without `--repair`. `refresh-live` writes immutable snapshots;
`scheduler-pass`, `run-worker`, lease/design/transition commands, lifecycle
attestations, `sync-verdict`, and `audit-authority --repair` mutate coordinator
or external authority and are controller-only.
