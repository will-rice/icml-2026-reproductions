# Persistent Paper-Owner Workers

## Objective

Make every dispatched competition worker a persistent, controller-capable
agent that runs `icml-repro-loop` directly. A worker repeatedly owns one paper
from selection through exact official score, then selects another paper.

The dispatch instruction is intentionally small:

```text
Use icml-repro-loop directly and keep running its paper-owner loop.
```

The skill, not an expanding dispatch prompt, defines the complete workflow.

## Roles

### Persistent paper-owner worker

This is the agent dispatched by the competition coordinator. It has authority
to:

- refresh and assess live challenge state;
- atomically select or reclaim one eligible paper attempt;
- create and write its assigned worktree;
- implement and independently validate the reproduction;
- publish the dedicated Hugging Face Space;
- submit and verify the exact deployed SHA;
- watch the official judgment;
- correct and resubmit fixable deficiencies;
- import the exact official verdict; and
- continue with another paper after releasing the completed or blocked lane.

The worker receives the credentials and network access required for these
controller operations. Credentials must never be committed or copied into
evidence bundles.

### Optional implementation subprocess

A paper-owner worker may use a subordinate coding process to generate a
proposal. That subprocess is not the dispatched competition worker. It remains
credential-free and worktree-scoped. The paper-owner worker retains lifecycle
authority and must independently validate any proposal before publication.

### Root competition coordinator

The root coordinator maintains pool capacity and receives worker events. It
does not manually shepherd normal paper phases. It:

- starts and replaces persistent workers;
- observes scored, blocked, and failed-worker events;
- notifies the user about material blockers;
- repairs shared infrastructure or authority conflicts; and
- reports aggregate score, rank, pending estimates, and capacity.

## Worker Loop

Each worker performs this loop until the competition closes or the coordinator
stops it:

```text
refresh and assess live state
→ atomically claim the best eligible paper
→ design and obtain independent review
→ implement with test-driven development
→ independently validate exact source
→ publish and verify the exact Space SHA
→ submit and observe the exact live queue record
→ watch the official verdict
→ correct, redeploy, and resubmit when the verdict is fixable
→ import the exact score
→ release the attempt
→ repeat
```

The worker owns at most one runnable paper at a time. While its paper is
submitted or judging, it remains dedicated to that paper and does not select
another.

## Atomic Selection And Reclamation

Selection must be one atomic controller operation against shared schema-v6
state. It must:

1. use a fresh assessed immutable snapshot;
2. rank eligible candidates by conservative expected points per remaining
   hour;
3. reject papers already held by a live lease or already completed by our
   owner;
4. create or reclaim exactly one attempt;
5. issue a fresh owner identity, lease, and fencing token; and
6. return the full attempt record needed by the skill.

Concurrent workers may race for candidates, but only one claim may commit.
Losing workers refresh and try the next candidate. A stale worker cannot write
after its lease is released or reclaimed.

Reclamation continues the existing attempt, evidence, deployment history, and
verdict history. It must not create a duplicate local attempt or reset prior
authority.

## Completion And Blocking

Normal paper completion requires importing the exact official verdict for the
exact submitted Space SHA. Worker exit, local validation, deployment, queue
visibility, and a pending judgment are not completion.

A genuine blocker is an observed condition that prevents useful progress now,
such as unavailable required artifacts, an external service outage, an
authority conflict, or an expired judging bound without a verdict. On a
genuine blocker, the worker must:

1. persist the phase, evidence, blocker, next action, and unperformed writes;
2. emit a structured blocker event to the root coordinator;
3. release its current lease without abandoning the attempt; and
4. continue its loop by selecting another paper.

The root coordinator notifies the user. The blocked attempt remains eligible
for the same or another worker to reclaim on a later iteration when its
observable prerequisite changes.

Scientific or implementation defects are not blockers. They trigger a
correction cycle on the same worker and paper.

## Authority And Credentials

The dispatched paper-owner worker is a trusted controller for its one current
attempt. It may execute the schema-v6 lifecycle, Hub publication, submission,
polling, and verdict commands required by the skill.

Authority remains bounded by:

- one live attempt lease and fencing token at a time;
- atomic shared-state commands;
- immutable snapshots and attestations;
- exact Space, paper, and deployed-SHA matching;
- independent validation before publication; and
- no credential persistence in Git, artifacts, logs, or subordinate
  subprocess environments.

Host permissions do not replace these application-level checks. Conversely,
the optional subordinate-process restrictions must not be applied to the
paper-owner worker itself.

## Failure Recovery

- **Worker process exits unexpectedly:** preserve its lease until expiry, then
  let a replacement reclaim the attempt with the predecessor token.
- **Lease nears expiry:** renew before the next mutation or long validation,
  deployment, or judgment wait.
- **Selection race:** refresh and retry without user interaction.
- **Validation fails:** remain on the same paper, record exact defects, and
  correct them.
- **Deployment or submission verification fails:** repair the publication and
  reverify the exact SHA; do not infer success.
- **Official fixable deficiency:** preserve the verdict, enter improvement,
  correct evidence, redeploy, resubmit, and resume watching.
- **External blocker:** notify root, release, and make the attempt reclaimable.

## Required Changes

1. Rewrite `icml-repro-loop` around the persistent worker loop and remove the
   ambiguous use of “worker” for both dispatched agents and subordinate coding
   processes.
2. Update `AGENTS.md` and handoff language so a dispatched paper-owner worker
   is explicitly controller-capable, while an optional subordinate process is
   explicitly not.
3. Add an atomic select-or-reclaim controller command suitable for concurrent
   persistent workers.
4. Add structured scored, blocked, and worker-failed events for the root
   coordinator.
5. Make the loop resume existing owned/reclaimable attempts before selecting
   new work.
6. Keep dispatch prompts minimal and require the skill to carry the complete
   workflow.

## Verification

Tests must prove:

- two concurrent workers cannot claim the same paper;
- a worker owns only one runnable paper at a time;
- a judging worker does not select another paper;
- exact official verdict import releases the attempt and starts the next
  iteration;
- a genuine blocker emits an event, releases the lease, and remains
  reclaimable;
- a later worker resumes the same blocked attempt without duplicating it;
- validation or implementation defects stay on the same attempt;
- lifecycle operations require current owner and fencing token;
- paper-owner workers can publish and watch using controller credentials;
- subordinate coding processes receive no Hub/GitHub credentials;
- the minimal direct dispatch causes a fresh agent to execute the complete
  loop rather than returning after implementation or submission; and
- skill validation, the focused reproduction-loop suite, root tests,
  pre-commit, and `git diff --check` pass.

Forward tests must dispatch fresh agents with only:

```text
Use icml-repro-loop directly and keep running its paper-owner loop.
```

The evaluation environment must use fake Hub and challenge services so the
test cannot mutate live competition infrastructure.
