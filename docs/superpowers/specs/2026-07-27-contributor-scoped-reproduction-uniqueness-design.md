# Contributor-Scoped Reproduction Uniqueness Design

## Goal

Allow our independently produced reproduction of a paper to be submitted and
scored even when another contributor already has a tagged Space, queued
submission, or official verdict for the same paper.

## Policy

Paper uniqueness is scoped to the publishing account, not to the challenge as
a whole.

- Another owner's tagged Space, queued submission, or verdict is informational
  and does not exclude a paper from our scheduler.
- Another owner's verdict does not block attestation of our exact deployed
  Space.
- We publish at most one canonical Space per paper under our allowlisted
  publishing account.
- A prior attempt in our own durable state still prevents the scheduler from
  creating a second local attempt for the same paper.
- An existing tagged Space under our publishing account for the same paper is
  a conflict unless it is the exact Space and revision bound to the active
  attempt.
- Official points enter our score only through a verdict for our exact Space
  and deployed revision.

This policy does not merge, copy, or claim evidence from another contributor.
Every submitted Space must contain our independently executable evidence and
must pass the existing validation and deployment attestations.

## Controller Changes

### Candidate scheduling

The scheduler will continue to exclude papers represented by our active
attempts, history, rejections, and leases. Public queue, tag, and verdict
records will exclude a candidate only when they belong to the configured
publishing owner. Records owned by other contributors will not make the paper
ineligible.

Owner identity is derived from the Space ID prefix before `/`. Public records
without a valid Space ID cannot establish contributor ownership and therefore
cannot create an external-owner exclusion.

### Submission attestation

`attest_submission` will ignore verdicts for the same paper when their
`space_id` differs from our deployed Space. It will still require:

1. one exact live Space record for our deployed Space ID;
2. the exact controller-attested deployed revision;
3. exactly one paper association on that Space;
4. one exact tagged record for the paper, Space, and revision;
5. one exact pending queue record for the paper, Space, and revision; and
6. no second canonical Space for that paper under our publishing owner.

A verdict already attached to our exact Space means the Space is no longer a
pending submission and must be handled through verdict synchronization, not a
new submission attestation.

### Existing Graph attempt

Attempt `64bfe193-333b-4b37-9683-9ac25ca5ac27` for Graph Pruning is currently
blocked from `deployed` solely because another contributor has an official
verdict for the paper. After the policy implementation passes tests, the
controller will renew or reclaim its lease through the normal fenced command,
resume it to `deployed`, fetch a fresh assessed snapshot, and attest its exact
Space only if all exact-Space requirements pass.

## Testing

Tests will prove:

- another owner's verdict does not block our exact pending Space;
- another owner's tagged or queued Space does not exclude a scheduler
  candidate;
- our own second Space for the same paper remains blocked;
- an exact-Space revision, tag, paper association, and queue mismatch remains
  blocked;
- our durable attempt/history/rejection/lease exclusions remain unchanged;
- official score reporting still counts only exact official verdicts and does
  not transfer another contributor's points.

Each behavior change will follow red-green-refactor. The focused controller and
scheduler suites run first, followed by the complete root test suite and
pre-commit.

## Rollout

1. Implement and verify the contributor-scoped guard.
2. Integrate the guard into the controller branch.
3. Refresh live public state.
4. Resume Graph from its recorded blocked phase and submit its exact deployed
   Space if the fresh snapshot observes it as pending.
5. Independently validate AGoQ, TimeRewarder, mHC, and RBench.
6. Deploy each validated paper to a distinct canonical Space under our account.
7. Refresh after each deployment and attest only exact pending observations.
8. Begin bounded verdict watching; never infer a score from queue state.

No existing external Space is deleted or overwritten, and no other
contributor's submission or verdict is mutated.
