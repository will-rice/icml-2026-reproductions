# Contributor-Scoped Reproduction Uniqueness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permit our distinct reproduction to be scheduled and submitted when another contributor has already reproduced the same paper, while retaining one canonical Space per paper under our publishing account.

**Architecture:** Put publishing-owner identity in one small policy module shared by scheduler and controller. Scope external queue/tag/verdict exclusions to allowlisted owners, and make submission authority depend only on our exact deployed Space plus same-owner canonical uniqueness.

**Tech Stack:** Python 3.10+, pytest, schema-v6 reproduction-loop controller, Hugging Face immutable snapshots.

## Global Constraints

- Another contributor's Space, queue record, or verdict never transfers points to us and never blocks our distinct reproduction.
- One durable local attempt and one canonical Space per paper under `wrice` remain the maximum.
- Existing exact-Space revision, tag, paper-association, pending-queue, validation, deployment, and verdict-sync checks remain unchanged.
- Every production change follows a witnessed red-green-refactor cycle.
- Do not mutate Hub state until all focused and root verification passes.
- Never modify or validate `submissions/nape/` in place.

---

### Task 1: Centralize Publishing-Owner Policy

**Files:**
- Create: `skills/icml-repro-loop/scripts/publication_policy.py`
- Modify: `skills/icml-repro-loop/scripts/controller.py`
- Test: `tests/test_repro_loop_controller_hub.py`

**Interfaces:**
- Produces: `ALLOWED_SPACE_OWNERS: frozenset[str]`
- Produces: `space_owner(space_id: object) -> str`
- Consumes: Space IDs in exact `owner/name` form.

- [ ] **Step 1: Write the failing policy test**

Add a focused test that imports `publication_policy`, asserts `wrice` is
allowlisted, accepts `wrice/repro-paper`, and rejects missing owner/name or
additional `/` components with `ValueError("space_id")`.

- [ ] **Step 2: Run the test and witness RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q tests/test_repro_loop_controller_hub.py -k publication_policy
```

Expected: FAIL because `publication_policy` does not exist.

- [ ] **Step 3: Add the minimal shared policy**

Create:

```python
"""Publishing-account policy shared by scheduling and Hub authority."""

ALLOWED_SPACE_OWNERS = frozenset({"wrice"})


def space_owner(space_id: object) -> str:
    if type(space_id) is not str:
        raise ValueError("space_id")
    owner, separator, name = space_id.partition("/")
    if not separator or not owner or not name or "/" in name:
        raise ValueError("space_id")
    return owner
```

Replace controller's private constant and `_space_owner` calls with imports
from this module. Delete the redundant private helper only after all references
are migrated.

- [ ] **Step 4: Run focused controller tests and witness GREEN**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q tests/test_repro_loop_controller_hub.py
```

Expected: all controller Hub tests pass.

- [ ] **Step 5: Commit**

```bash
git add skills/icml-repro-loop/scripts/publication_policy.py skills/icml-repro-loop/scripts/controller.py tests/test_repro_loop_controller_hub.py
git commit -m "refactor: centralize reproduction publishing owner"
```

### Task 2: Scope Scheduler Exclusions to Our Account

**Files:**
- Modify: `skills/icml-repro-loop/scripts/scheduler.py`
- Modify: `tests/test_repro_loop_scheduler.py`

**Interfaces:**
- Consumes: `publication_policy.ALLOWED_SPACE_OWNERS`
- Consumes: `publication_policy.space_owner(space_id)`
- Produces: `_external_claimed_paper_ids(snapshot: dict) -> set[str]`

- [ ] **Step 1: Write failing scheduler tests**

Add one test whose snapshot contains candidate `paper-a` plus queue, tag, and
verdict records for `other/repro-paper-a`; assert `scheduler_pass` admits
`paper-a`.

Add a parameterized test for each of `queued_submissions`, `tagged_spaces`, and
`verdicts` using `wrice/repro-paper-a`; assert `paper-a` remains excluded.
Each test record must include a valid `space_id`.

- [ ] **Step 2: Run the tests and witness RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q tests/test_repro_loop_scheduler.py -k "external_contributor or publishing_owner"
```

Expected: the external-contributor case fails because all public records are
currently treated as globally claimed.

- [ ] **Step 3: Implement contributor-scoped external exclusions**

Import `publication_policy`. Add:

```python
def _external_claimed_paper_ids(snapshot: dict) -> set[str]:
    claimed = set()
    for field in ("queued_submissions", "tagged_spaces", "verdicts"):
        for record in snapshot[field]:
            space_id = record.get("space_id")
            try:
                owner = publication_policy.space_owner(space_id)
            except ValueError:
                continue
            if owner in publication_policy.ALLOWED_SPACE_OWNERS:
                claimed.add(record["paper_id"])
    return claimed
```

Use this helper in `_claimed_paper_ids` while leaving attempts, history,
rejections, and active candidate leases unchanged.

- [ ] **Step 4: Run focused scheduler tests and witness GREEN**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q tests/test_repro_loop_scheduler.py
```

Expected: all scheduler tests pass, including durable local uniqueness.

- [ ] **Step 5: Commit**

```bash
git add skills/icml-repro-loop/scripts/scheduler.py tests/test_repro_loop_scheduler.py
git commit -m "fix: scope paper claims to publishing owner"
```

### Task 3: Permit Another Owner's Verdict at Submission

**Files:**
- Modify: `skills/icml-repro-loop/scripts/controller.py`
- Modify: `tests/test_repro_loop_controller_hub.py`

**Interfaces:**
- Consumes: the exact deployment attestation's `space_id` and `space_sha`
- Preserves: `attest_submission(...) -> dict`

- [ ] **Step 1: Replace the old failing expectation**

Create a test that deploys the fixture Space, adds a same-paper verdict for
`other/repro-paper-a`, preserves the exact `wrice` Space/tag/pending queue
records, and asserts `attest_submission` transitions to `submitted`.

Retain a separate test proving a second same-paper Space owned by `wrice`
raises `ValueError("duplicate")`.

- [ ] **Step 2: Run the tests and witness RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q tests/test_repro_loop_controller_hub.py -k "other_owner_verdict or same_owner_duplicate"
```

Expected: the other-owner verdict case fails with `ValueError("verdict")`.

- [ ] **Step 3: Remove only the global verdict veto**

Delete the loop in `attest_submission` that rejects every same-paper verdict
on a different Space. Preserve exact live Space lookup, exact revision, exact
paper association, same-owner canonical-Space count, exact tag, and exact
pending queue validation.

- [ ] **Step 4: Run focused controller tests and witness GREEN**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q tests/test_repro_loop_controller_hub.py
```

Expected: all controller Hub tests pass.

- [ ] **Step 5: Commit**

```bash
git add skills/icml-repro-loop/scripts/controller.py tests/test_repro_loop_controller_hub.py
git commit -m "fix: allow distinct paper submission by owner"
```

### Task 4: Align Workflow Documentation and Verify

**Files:**
- Modify: `skills/icml-repro-loop/SKILL.md`
- Modify: `skills/icml-repro-loop/references/selection-rubric.md`
- Modify: `docs/HANDOFF.md` only by adding a concise current milestone without overwriting existing user edits.

**Interfaces:**
- Documents the contributor-scoped scheduler and submission policy.
- Does not alter lifecycle command signatures.

- [ ] **Step 1: Update policy wording**

Change global phrases such as "Reject a duplicate paper" to state that another
owner's reproduction is allowed, while a duplicate local attempt or second
same-owner canonical Space remains rejected.

- [ ] **Step 2: Run complete verification**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run python /home/will/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/icml-repro-loop
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache PRE_COMMIT_HOME=/tmp/icml-repro-pre-commit uv run pre-commit run -a
git diff --check
```

Expected: all tests, skill validation, hooks, and whitespace checks pass.
Parent validation must continue to exclude archival `submissions/nape/` as
specified by `AGENTS.md`.

- [ ] **Step 3: Commit code-owned documentation**

Stage the skill documentation and only the exact new HANDOFF hunk. Do not stage
unrelated existing HANDOFF changes or telemetry.

```bash
git add skills/icml-repro-loop/SKILL.md skills/icml-repro-loop/references/selection-rubric.md
git commit -m "docs: permit contributor-scoped reproductions"
```

### Task 5: Resume Graph and Submit the Exact Distinct Space

**Files:**
- Mutate through controller commands: `state/repro-loop.json` and its Graph attempt, lease, snapshot, transaction, and attestation shards.
- Do not edit Graph evidence source unless fresh validation identifies a defect.

**Interfaces:**
- Attempt: `64bfe193-333b-4b37-9683-9ac25ca5ac27`
- Paper: `a3GdvuPItd`
- Space: `wrice/repro-graph-pruning`
- Current phase: `blocked`, `blocked_from: deployed`

- [ ] **Step 1: Inspect exact CLI signatures**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run python skills/icml-repro-loop/scripts/state.py --help
```

Inspect the help for `claim-attempt`, the blocked-phase resume command,
`refresh-live`, `attest-submission`, and `watch-attempt`. Never reconstruct a
lease token by guesswork.

- [ ] **Step 2: Renew or reclaim the Graph controller lease**

Use the exact predecessor fencing token recorded in the lease/attempt
transaction. Resume only from `blocked` to its recorded `blocked_from:
deployed`.

- [ ] **Step 3: Refresh and inspect live state**

Run raw `refresh-live`, inspect it with `show-snapshot`, and generate a fresh
assessment bound to the unchanged challenge revision if `attest-submission`
requires an assessed snapshot.

- [ ] **Step 4: Attest the exact pending Graph Space**

Proceed only if the fresh snapshot contains:

- `wrice/repro-graph-pruning`;
- deployed SHA `3c483996fffc32b05074d909330df05cfb4e6b80`;
- paper tag `a3GdvuPItd`; and
- one exact pending queue record.

Another owner's Graph verdict is permitted but never imported into our
attempt.

- [ ] **Step 5: Start bounded judging**

Use `watch-attempt` with a positive poll limit and aware deadline. Record its
attestation ID and run `score-report`; do not infer points before an exact
official verdict exists for our Space and SHA.

- [ ] **Step 6: Commit controller state and HANDOFF milestone**

Commit the exact Graph state/snapshot/attestation/transaction shards and the
new HANDOFF hunk. Exclude telemetry unless separately reviewed.

### Task 6: Advance the Four Completed Proposals

**Files:**
- AGoQ worktree: `.worktrees/points-agoq-clean`
- TimeRewarder worktree: `.worktrees/points-timerewarder-clean`
- mHC worktree: `.worktrees/points-mhc-clean`
- RBench worktree: `.worktrees/points-rbench-clean`
- Mutate controller state only through dedicated validation, deployment, snapshot, submission, and watch commands.

**Interfaces:**
- AGoQ attempt: `2fc3b006-3307-4fc3-8df6-c000379298c4`
- TimeRewarder attempt: `bf0d2300-4479-4e3c-ba99-bb023ee6751e`
- mHC attempt: `3d164e18-39ef-416e-b986-96b5a5d4e12d`
- RBench attempt: `8c21f2dc-a357-422e-9c1b-79a4d417e3dc`

- [ ] **Step 1: Independently inspect each clean proposal**

Confirm its approved plan, commit lineage, clean scoped diff, deterministic
evidence hashes, claim statuses, and absence of state/Hub/NAPE changes.

- [ ] **Step 2: Run each authoritative validation manifest**

Use `attest-validation` separately for each attempt and preserve its immutable
attestation ID. A worker test report alone cannot advance phase.

- [ ] **Step 3: Publish one canonical Space per paper**

Use `publish-deployment` from each exact validated source tree. Verify the
Space is `RUNNING`, carries the exact paper and challenge tags, and resolves to
the attested SHA.

- [ ] **Step 4: Refresh and attest each distinct submission**

After every deployment, obtain a fresh snapshot and call
`attest-submission`. Other contributors' verdicts do not block these exact
Spaces; same-owner duplicate Spaces still do.

- [ ] **Step 5: Begin bounded judging and report**

Run `watch-attempt` separately with finite limits/deadlines, persist every
attestation, update HANDOFF, and run `score-report`. Official points remain
unchanged until `sync-verdict` imports a matching official verdict.
