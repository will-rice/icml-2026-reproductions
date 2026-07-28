# Paper-Owner Controller Skill Design

## Objective

Make one invocation of `icml-repro-loop` own one ICML reproduction from paper
selection through an official score. A dispatched top-level agent must use the
skill directly and must not return merely because implementation finished.

## Dispatch Contract

Every lane is dispatched with a minimal instruction equivalent to:

> Use `icml-repro-loop` directly. Own one paper through selection,
> implementation, controller validation, Space publication, submission,
> official-score watching, and any evidence-driven correction required before
> the deadline.

The dispatch does not duplicate workflow details. The skill is the sole
versioned source of lifecycle behavior, command ordering, authority, retry
rules, and completion criteria.

## Roles

The dispatched top-level agent is a **paper-owner controller**, not an
untrusted paper worker. It holds one explicit attempt lease and has authority
to run controller lifecycle commands for that attempt, publish its dedicated
Space, observe the live challenge, and import its exact official verdict.

For code generation, the paper-owner controller launches a guarded
paper-scoped implementation subprocess through `state.py run-worker`. That
subprocess receives internet and write access to its assigned worktree but no
Hub credentials or coordinator authority. The paper-owner controller reviews
the proposal and retains responsibility for everything after worker exit.

This makes the externally dispatched worker autonomous end-to-end while
keeping generated paper code away from shared credentials and other attempts.

## Per-Paper State Machine

One invocation advances exactly one attempt:

1. **Select and claim**
   - Refresh and assess live challenge data when no assessed snapshot is
     available.
   - Choose the highest expected leaderboard points per remaining hour that
     satisfies the challenge constraints.
   - Claim one fenced attempt and persist its paper-specific design.
2. **Design and review**
   - Bind exact live claim text and hashes, upstream revisions, feasibility,
     cost, evidence methods, and judge surfaces.
   - Obtain an independent design review. Correct rejected designs without
     abandoning the attempt.
3. **Implement**
   - Create an isolated worktree, worker contract, runtime preflight, and
     persistent worktree permission rules.
   - Dispatch the paper-scoped implementation subprocess with the direct skill
     requirement and approved plan.
4. **Watch worker completion**
   - Treat process exit telemetry as an event.
   - On exit, immediately inspect its commit and diff; never wait for a user
     status request.
   - A no-op exit, timeout, or rejected proposal immediately creates a
     correction contract and relaunches the worker with concrete findings.
5. **Controller validation**
   - Run fresh evidence generation, project tests, root tests excluding the
     archival NAPE snapshot, skill validation, and pre-commit through an
     immutable validation manifest.
   - Reject hard-coded outcomes, paper-value relabeling, provenance gaps,
     stale pages, nondeterminism, authority claims, or cross-project edits even
     when unit tests pass.
   - Persist `attest-validation` only from fresh controller evidence.
6. **Publish and submit**
   - Publish the exact validated source to a dedicated Hugging Face Space.
   - Verify owner, paper tag, challenge tag, deployed SHA, and `RUNNING`
     status through `publish-deployment`.
   - Refresh live data and use `attest-submission` only for the exact observed
     Space and commit.
7. **Watch for score**
   - Start a bounded watch and persist every official poll observation.
   - Pending is not failure and does not justify changing scientific evidence.
     First verify that the exact submission remains visible and healthy.
   - Continue watching within the deadline rather than returning control.
8. **Correct when it does not score**
   - Deployment or visibility failure: repair publication/submission and
     re-observe.
   - Controller validation failure: return exact findings to the implementation
     subprocess and rerun validation.
   - Official `inconclusive`, rejected, or low-value claim verdict: diagnose
     the cited evidence deficiency, enter improvement, implement a distinct
     correction, redeploy, resubmit, and watch the new exact commit.
   - A queue that is merely pending is watched, not cosmetically resubmitted.
9. **Terminate**
   - Success only after importing an exact official verdict with
     `sync-verdict`.
   - Otherwise stop only for the competition deadline or a persisted genuine
     blocker that cannot be resolved within authority and constraints.

## Completion and Refill Events

Worker exit is not lane completion. The paper-owner controller stays active
through validation and judgment.

The surrounding competition coordinator receives events from paper owners:

- `scored`: record official points and immediately dispatch a new paper owner
  if eligible papers and capacity remain;
- `judging`: the implementation slot is free, so dispatch another paper owner
  without waiting for the verdict;
- `correction-needed`: reuse the same attempt and worktree with a new fenced
  owner when required;
- `blocked`: retain the blocker on that attempt and refill capacity elsewhere.

No status question from the user is required to trigger these transitions.

## Failure Handling

- Every subprocess has durable queued, launched, and exited telemetry.
- A watcher polls process state and live challenge state at bounded intervals;
  it reacts to conditions rather than elapsed-time guesses.
- Expired leases are reclaimed only with the exact predecessor token.
- Repeated no-op worker exits cause model/runtime substitution or a more
  explicit correction plan, not repeated identical dispatches.
- Concurrent paper owners mutate only their explicit attempt and dedicated
  Space; schema transactions and fencing reject stale owners.
- Credential, scope, or authority violations reject the proposal and rotate
  the worker. They never become controller attestations.

## Testing Strategy

Skill pressure tests must demonstrate these behaviors:

1. A worker reports green tests but hard-codes evidence; the paper owner rejects
   it, relaunches correction, and does not attest validation.
2. A worker exits without edits due to a permission prompt; the owner fixes
   scoped runtime permissions and relaunches automatically.
3. A Space builds but has the wrong SHA or tags; publication is not attested.
4. A submission remains pending; the owner keeps watching and does not alter
   evidence merely to refresh the queue.
5. An official inconclusive verdict cites missing provenance; the owner enters
   improvement, deploys a corrected commit, and watches again.
6. A paper reaches judging; the competition coordinator immediately fills the
   newly available implementation capacity.
7. A stale owner tries to mutate an attempt after lease reclaim; fencing
   rejects it.

The skill passes only if dispatched agents use it directly and carry the paper
through the full state machine instead of stopping at an implementation
proposal.

## Non-Goals

- Giving generated paper code unrestricted access to shared credentials or
  other submissions.
- Treating local evidence statuses as official verdicts.
- Repeatedly resubmitting unchanged evidence while the official queue is
  merely pending.
- Allowing one paper's blocker to stop independent paper-owner lanes.
