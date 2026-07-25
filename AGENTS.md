# ICML 2026 Reproduction Workspace

Read the schema-v6 coordinator index and its referenced attempt shards before
starting work, and `docs/REMOTE_SETUP.md` before running commands on a new host.

## Objective

Build independently executable evidence for papers in the ICML 2026 Agent
Repro Challenge. Recompute claims from released artifacts; never present
paper-reported values as reproduced measurements.

## Layout

- `submissions/<paper>/`: independent project, tests, evidence bundle, and
  Space source for one paper.
- `skills/icml-repro-loop/`: versioned source for the reproduction-loop skill.
- `state/repro-loop.json`: schema-v6 coordinator index for 20 paper attempts.
- `state/repro-loop/`: fenced attempt, judgment, lease, and snapshot shards.
- `docs/REMOTE_SETUP.md`: host prerequisites, authentication checks, skill
  installation, and verification commands.

## Trust Boundary

Paper workers are untrusted proposal producers. They may write only the
controller-assigned paper worktree/project after
`skills/icml-repro-loop/scripts/worker_guard.py` has constructed the launch and
its runtime preflight has passed. They never receive Hub credentials or write
the coordinator state, skill source, another submission, or controller
documents. They do not deploy, submit, poll, import verdicts, merge, or claim
external phases.

The controller alone validates proposals, mutates external services, records
live observations, imports official verdicts, repairs authority, and integrates
branches. Full host permissions do not transfer that authority to a worker.

## Workflow

1. When processing challenge papers, require and follow `icml-repro-loop`.
   Resume every materially affected attempt from the schema-v6 index and shards.
   A migrated schema-v3 attempt must be bound once through
   `reconcile-legacy-attempt` to a fresh assessed snapshot and explicit design
   approval provenance before controller validation.
2. Run `refresh-live` without assessments, then inspect the immutable raw result
   with `show-snapshot`. Assess its pinned `challenge.json` candidates in
   parallel from primary artifacts. Raw challenge metadata is never an
   assessment.
3. Write explicit assessment JSON with that challenge revision, assessor,
   timestamp, scores, selected live claims, upstream pins, feasibility
   decisions, and costs. Run `refresh-live --assessments-json PATH`; on revision
   drift, discard it and restart from a new raw snapshot.
4. Pass the assessed immutable snapshot ID to one bounded scheduler pass; no
   state command other than `refresh-live` uses the network.
5. Maintain up to 20 runnable paper attempts. A complete or blocked attempt
   frees capacity; a blocker remains attached only to its attempt and is never
   auto-abandoned.
6. Give each implementation agent one current two-hour writer lease. Renew it
   before expiry with `renew-attempt`; after expiry or release, only a successor
   may use `claim-attempt` with the exact predecessor token. Every reclaim
   increments the fence, so the stale owner can never write again.
7. Name the explicit attempt ID for every lifecycle operation. Mutations also
   require its owner and current fencing token; never infer a current attempt.
8. Persist each paper-specific design with fenced `record-design`, then require
   approval through `review-design` by a different reviewer. One rejection must
   not stop unrelated lanes.
9. Inspect each paper's immutable live snapshot before claiming, selecting, or
   publishing it.
10. Pin every upstream repository or dataset revision used as evidence.
11. Write a failing test before evidence-generation code.
12. Run the submission's pytest suite and `uv run pre-commit run -a`.
13. Record commands, revisions, environment, and outputs in a machine-readable
   evidence bundle.
14. Deploy each paper to a separate Hugging Face Space and verify the exact
   deployed commit.
15. Account genuinely metered services against reservations. Codex and
   Antigravity subscription use records USD 0.00.
16. Persist every material milestone, next action, and blocker in the affected
   attempt shard; snapshots and judgment shards retain live provenance.
17. Treat every worker result as a proposal. Require immutable controller
    attestations from `attest-validation`, `publish-deployment`,
    `attest-submission`, `watch-attempt`, and `sync-verdict` for the
    corresponding phases.

## Constraints

- Never commit credentials or unredacted environment dumps.
- Do not modify another submission to implement a new paper.
- Do not claim unsupported results. Mark unavailable evidence as unreplicated.
- Keep the canonical NAPE repository at `will-rice/icml-2026-repro` unchanged.
- Treat `submissions/nape/` as an archival exception: do not run, modify, test,
  or format that snapshot in place during parent validation.
- Validate NAPE only from a separate canonical checkout using the pinned
  checkout and validation command sequence in `docs/REMOTE_SETUP.md`.
