# Five-Paper Autonomous Scheduler Design

## Goal

Maintain five independently executable ICML reproduction attempts at all
times. Each paper advances through selection, design, implementation,
validation, publication, submission, and judging without blocking unrelated
papers. When a runnable attempt completes or becomes blocked, the scheduler
admits a replacement until five runnable lanes are active again.

## Scope

This change replaces the schema-v3 single-current-paper model on `main`. It
does not merge historical submissions, maintenance overlays, or state from the
divergent multi-state branches. It selectively ports their tested sharded
state, leases, attempts, and scheduling concepts into the current repository.

The active EEG-FM-Bench attempt becomes lane 1 without changing its paper ID,
upstream revision, targets, design approval, implementation branches, or paid
cost. Four newly refreshed and eligible papers fill lanes 2 through 5.

## Alternatives Considered

### Sharded coordinator with five runnable lanes

Use one small global index and one atomic state shard per attempt, judgment,
lease, and live snapshot. This is the selected design because attempts can be
written independently while candidate claims and external resources remain
fenced against duplicates.

### Five schema-v3 lane files

Reuse the current state machine independently five times. This is smaller but
does not safely deduplicate candidate selection, Spaces, verdicts, or aggregate
cost across concurrent writers.

### Five unrelated repository clones

Run five autonomous loops without a coordinator. This starts quickly but has
the highest risk of duplicate papers, conflicting publications, incomplete
recovery, and contradictory handoff state.

## State Layout

`state/repro-loop.json` becomes a schema-v6 coordinator index containing:

- `max_runnable_attempts: 5`;
- immutable references to active and archived attempt shards;
- recorded rejections and globally claimed paper identities;
- immutable live-refresh snapshot references;
- resource limits and active lease summaries;
- aggregate metered external-API cost.

Mutable records live below `state/repro-loop/`:

- `attempts/<attempt-id>.json` stores paper identity, targets, phase,
  transition history, design/review, worktree, assigned agents, cost, external
  IDs, deployment identity, and blockers;
- `judgments/<attempt-id>.json` stores bounded poll rounds, raw verdicts,
  normalized target mappings, and verdict provenance;
- `leases/<lease-id>.json` stores owner, attempt, resource, expiry, fencing
  token, and release state;
- `snapshots/<sha256>.json` stores immutable live catalog, claims, Spaces,
  queues, and verdict observations;
- `transactions/<transaction-id>.json` supports crash-safe multi-file writes.

The index never duplicates mutable attempt content.

## Lane Semantics

A runnable attempt is in `selected`, `design-pending`, `implementing`,
`validated`, `deployed`, `submitted`, `judging`, or `improving`. The scheduler
admits candidates until five runnable attempts exist.

`complete` attempts move to history and immediately free a lane. `blocked`
attempts remain durable and visible but do not consume a runnable lane, so an
unrelated blocker cannot reduce throughput. Blocked attempts are never
autonomously abandoned; only an explicit user instruction may archive them.

Each attempt has exactly one authoritative writer. Read-only research, review,
and monitoring agents may overlap. Different attempts always use separate git
worktrees and submission directories.

## Admission

Every admission pass uses one current immutable live snapshot. A candidate is
eligible only when it:

- has no verdict, tagged reproduction Space, queued submission, candidate
  lease, active attempt, or history entry;
- has at least two independently testable claims;
- has a pinned upstream revision and compatible artifact access;
- has a CPU-only execution path;
- has no unresolved safety or licensing blocker.

The scheduler scores eligible candidates with the repository rubric, acquires
a globally fenced candidate lease, creates an attempt shard, and assigns an
isolated worktree. It fails closed if any identity changed after the snapshot.

## Automatic Design Approval

Every attempt receives a committed paper-specific design. A different agent
reviews the design for live claim alignment, truthful scope, artifact access,
licensing, safety, cost, and execution feasibility. A recorded approval
advances that attempt to `implementing`. A rejection returns only that design
for revision and does not pause other papers.

The user's 2026-07-24 instruction authorizes this independent-review gate in
place of per-paper user approval. User instructions can still pause or reject
any attempt.

## Resource Leases

Fenced leases serialize only conflicting resources:

- candidate paper identity;
- attempt writer;
- Hugging Face Space publication;
- compute job;
- metered external-API reservation;
- global migration or index maintenance.

Different Spaces may publish concurrently. The same Space, paper, or attempt
cannot have two writers. Expired owners cannot overwrite successor work because
all mutations require the current fencing token.

Subscription Codex and Antigravity usage records USD 0.00. Cost reservation
applies only to genuinely metered external APIs or paid infrastructure. The
existing per-paper USD 10 safety ceiling remains a fail-safe for such services
and does not limit subscription agents.

## Migration

Migration validates and hashes the current schema-v3 source before writing:

1. Convert EEG-FM-Bench `current` into one active attempt shard, preserving
   exact phase, targets, revision, design approval, and cost.
2. Convert the abandoned diffusion record into an archived attempt shard,
   preserving its blocker and history payload.
3. Preserve every rejection and aggregate cost.
4. Write all shards and a transaction manifest before atomically installing
   the v6 index last.
5. Verify semantic equivalence and retain a hash-addressed v3 backup.

Migration is idempotent. Recovery completes verified staged writes or removes
them before index installation.

## Scheduling and Recovery

One bounded scheduler pass:

1. validates or records a live snapshot;
2. expires stale leases;
3. reconciles durable attempts with worktree commits and external IDs;
4. assigns dependency-ready work;
5. admits candidates until five runnable lanes exist;
6. advances bounded judgment monitoring independently.

The scheduler does not hold global locks during network, agent, test, or
publication work. An ambiguous external mutation blocks only its owning
attempt until reconciled.

## Testing

Test-driven implementation must cover:

- semantic schema-v3 migration and idempotent recovery;
- simultaneous claims for the same and different papers;
- concurrent writes to independent attempts;
- same-attempt fencing and stale-writer rejection;
- exactly five runnable admissions and automatic refill;
- blocked attempts not consuming runnable capacity;
- duplicate paper, Space, submitted SHA, and verdict rejection;
- independent design approval and rejection;
- simultaneous publication leases for different Spaces;
- metered-cost reservation while subscription agents remain zero-cost;
- monitoring one judgment while other attempts advance;
- interrupted multi-file writes at every transaction boundary;
- root tests, skill validation, and pre-commit.

Concurrency tests use deterministic barriers instead of timing-only sleeps.
External services use recorded fixtures in unit tests, followed by read-only
live identity checks.

## Operational Outcome

After migration, EEG-FM-Bench continues in lane 1. A fresh live refresh selects
and starts four additional papers. The coordinator keeps five runnable paper
pipelines active, while every paper retains independent evidence, deployment,
judging, failure recovery, and truthful claim status.
