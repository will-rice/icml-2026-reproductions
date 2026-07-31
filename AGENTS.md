# ICML 2026 Reproduction Workspace

Read `docs/HANDOFF.md` and inspect the version in `state/repro-loop.json`
before starting work. The authoritative coordinator state is schema-v6. Read
`docs/REMOTE_SETUP.md` before running commands on a new host.

## Objective

Build independently executable evidence for papers in the ICML 2026 Agent
Repro Challenge. Recompute claims from released artifacts; never present
paper-reported values as reproduced measurements.

## Layout

- `submissions/<paper>/`: independent project, tests, evidence bundle, and
  Space source for one paper.
- `skills/icml-repro-loop/`: versioned source for the reproduction-loop skill.
- `state/repro-loop.json`: authoritative schema-v6 coordinator index.
- `state/repro-loop/`: fenced attempt, judgment, lease, and snapshot shards.
- `docs/REMOTE_SETUP.md`: host prerequisites, authentication checks, skill
  installation, and verification commands.

## Persistent Paper-Owner Authority

Directly dispatched persistent paper-owner workers are trusted controllers.
Each owns one current fenced paper at a time, uses `claim-next` for each
iteration, and may use controller credentials only for the exact lifecycle of
its current fenced attempt. That includes validation, external service
mutations, live observations, Hub publication, submission, bounded verdict
watching, correction, exact official-verdict import, authority repair, and
`release-paper`. Submitted and judging attempts remain dedicated; only an
exactly scored or genuinely blocked release allows the next `claim-next`.

## Subordinate Implementation Subprocess Boundary

An optional subordinate implementation subprocess is not the dispatched
worker. It is an untrusted proposal producer that may write only the
paper-owner-assigned worktree/project after
`skills/icml-repro-loop/scripts/worker_guard.py` constructs the launch and its
runtime preflight passes. It never receives Hub or controller credentials and
cannot write coordinator state, skill source, another submission, or controller
documents. It does not deploy, submit, poll, import verdicts, merge, release,
or claim external phases. Full host permissions do not transfer this authority.

## Workflow

1. When processing challenge papers, require and follow `icml-repro-loop`.
   Use the schema-v6 index and shards. A migrated historical attempt must be
   bound once through
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
4. Each persistent paper-owner worker passes the fresh assessed immutable
   snapshot ID to `claim-next` for one current paper or reclaim. Networked live
   lifecycle operations require current controller authority for that fenced
   attempt; read-only local state operations do not create that authority.
5. A persistent paper-owner worker owns one current paper at a time. It does
   not select another paper while submitted or judging; scored or blocked
   release ends its iteration. Before blocked release, use fenced
   `transition-attempt` to record nonempty `blocker` and `next_action`; the
   blocker remains attached to a reclaimable attempt and is never
   auto-abandoned.
6. Give each persistent paper-owner worker one current two-hour writer lease.
   Any optional subordinate implementation subprocess works only under the
   guarded contract and never owns lifecycle authority. Renew the lease
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
17. Treat every subordinate implementation subprocess result as a proposal.
    Require immutable persistent-paper-owner-controller attestations from
    `attest-validation`, `publish-deployment`,
    `attest-submission`, `watch-attempt`, and `sync-verdict` for the
    corresponding phases.

## Constraints

- Never commit credentials or unredacted environment dumps.
- Never track `state/` in Git (no `git add -f` of coordinator state); the
  sharded store is the durability layer. Never rewind a shared branch to an
  ancestor commit.
- Do not modify another submission to implement a new paper.
- Do not claim unsupported results. Mark unavailable evidence as unreplicated.
- Keep the canonical NAPE repository at `will-rice/icml-2026-repro` unchanged.
- Treat `submissions/nape/` as an archival exception: do not run, modify, test,
  or format that snapshot in place during parent validation.
- Validate NAPE only from a separate canonical checkout using the pinned
  checkout and validation command sequence in `docs/REMOTE_SETUP.md`.
