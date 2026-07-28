# Current Handoff

## Operating State

The authoritative coordinator state is schema-v6 in
`state/repro-loop.json` and its `state/repro-loop/` shards. Inspect those
records at runtime; do not copy attempt phase, score, rank, queue, lease, or
blocker values from this document.

Use the repository source at `skills/icml-repro-loop/` and install it according
to `docs/REMOTE_SETUP.md` on a new host. Historical attempts, verdicts,
attestations, and evidence remain in the coordinator shards and Git history,
not as executable instructions here.

All lifecycle operations use the schema-v6 persistent paper-owner commands
defined by that skill.

## Persistent Worker Model

Direct dispatch creates a trusted persistent paper-owner worker. Its complete
loop is:

```text
refresh and assess
→ inspect ready released blockers
→ atomically reclaim or claim one paper
→ design and independently review
→ implement and validate
→ publish and verify the exact Space SHA
→ submit and observe the exact live record
→ watch, correct, and resubmit as required
→ import the exact official verdict
→ release
→ repeat
```

The owner holds one current fenced paper and a two-hour writer lease. It
remains dedicated while the paper is submitted or judging. It uses controller
credentials only for the current attempt and never writes credentials to Git,
evidence, logs, or subordinate environments.

An optional subordinate implementation subprocess is not the dispatched
worker. It is credential-free, proposal-only, and limited to the assigned
paper worktree/project. The persistent owner independently validates its
proposal and retains publication, submission, watching, verdict, state, and
release authority.

## Iteration Boundaries

After exact `sync-verdict`, use `release-paper --outcome scored`. For a genuine
external blocker, first persist `blocked` with nonempty `blocker` and
`next_action`, then use `release-paper --outcome blocked`, notify the root
coordinator, and continue the loop.

A blocked release stays active and reclaimable. Before new selection, inspect
released blocked attempts against a fresh assessed immutable snapshot. Reclaim
the highest-priority resolved or actionable blocker explicitly with
`claim-next --reclaim-attempt-id`; leave unresolved blockers reclaimable and
use ordinary `claim-next` for new work.

Worker exit, local validation, Space health, submission visibility, and a
pending judgment are not completion. Never abandon an attempt automatically.

## Validation

Exclude the archival NAPE snapshot from parent validation:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q --ignore=submissions/nape
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run python \
  "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" \
  skills/icml-repro-loop
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
PRE_COMMIT_HOME=/tmp/icml-repro-pre-commit \
uv run pre-commit run -a
git diff --check
```

Validate NAPE only from the separate pinned canonical checkout described in
`docs/REMOTE_SETUP.md`.
