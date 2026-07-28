# Persistent Paper-Owner Workers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each dispatched worker run `icml-repro-loop` persistently, owning one paper from atomic selection through exact official score before repeating, while releasing recoverably blocked attempts for later reclamation.

**Architecture:** Add one atomic `claim-next` controller operation for a persistent paper owner, plus one fenced `release-paper` operation that emits structured coordinator events. Rewrite the skill vocabulary so the dispatched worker is a trusted controller and any nested coding process is explicitly subordinate and credential-free; use existing schema-v6 attestations for validation, publication, submission, watching, and verdict import.

**Tech Stack:** Python 3.12, argparse, schema-v6 JSON shards, `fcntl` file locks, pytest, Markdown skills, YAML agent metadata, Hugging Face controller APIs.

## Global Constraints

- A persistent paper-owner worker owns at most one paper at a time.
- A submitted or judging worker remains dedicated to that paper until exact verdict import or a genuine persisted blocker.
- Selection and reclamation use fresh assessed immutable snapshots and current fencing authority.
- A released blocked attempt remains active and reclaimable; it is never duplicated or automatically abandoned.
- The dispatched paper-owner worker may use controller credentials, but credentials never enter Git, evidence, logs, or subordinate environments.
- An optional implementation subprocess remains credential-free and worktree-scoped.
- NAPE remains archival and excluded from in-place testing, editing, and formatting.
- Preserve unrelated changes in `docs/HANDOFF.md`, `state/repro-loop.json`, and state shards.

## File Map

- `skills/icml-repro-loop/scripts/scheduler.py`: select or reclaim exactly one paper for one persistent worker.
- `skills/icml-repro-loop/scripts/paper_owner.py`: release completed/blocked ownership and emit coordinator events.
- `skills/icml-repro-loop/scripts/state.py`: expose `claim-next` and `release-paper`.
- `skills/icml-repro-loop/SKILL.md`: define the persistent end-to-end loop and authority vocabulary.
- `skills/icml-repro-loop/references/paper-owner-loop.md`: define iteration and blocker reactions.
- `skills/icml-repro-loop/references/submission-checklist.md`: verify loop iteration and release handoff.
- `skills/icml-repro-loop/agents/openai.yaml`: reduce dispatch to direct persistent skill invocation.
- `AGENTS.md`: distinguish trusted dispatched paper owners from subordinate proposal processes.
- `docs/REMOTE_SETUP.md`: document credentials and runtime requirements for paper owners versus subordinates.
- `docs/HANDOFF.md`: append the new operating model without changing existing milestone history.
- `evals/icml-repro-loop/scenarios.json`: pressure cases for iteration, judging dedication, and blocker reclamation.
- `tests/test_repro_loop_scheduler.py`: atomic new selection and blocked-attempt reclamation.
- `tests/test_repro_loop_paper_owner.py`: release validation and structured events.
- `tests/test_repro_loop_state.py`: CLI wiring and fencing.
- `tests/test_repro_loop_paper_owner_skill.py`: skill, prompt, checklist, and scenario behavior.

---

### Task 0: Remove the Stale Unattested Improvement Test Path

**Files:**
- Modify: `tests/test_repro_loop_scheduler.py`

**Interfaces:**
- Consumes: `attempts.transition_attested(..., phase="improving")` with a
  persisted `kind="verdict"` attestation.
- Produces: a clean baseline that no longer models the removed generic
  `judging -> improving` path.

- [ ] **Step 1: Confirm the two stale tests fail for the intended reason**

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  'tests/test_repro_loop_scheduler.py::test_second_judgment_archives_superseded_first_round'
```

Expected: both parameterizations fail with `ValueError: attestation` because
the test calls generic `transition_attempt(..., "improving")`.

- [ ] **Step 2: Replace the obsolete transition with exact verdict authority**

In `test_second_judgment_archives_superseded_first_round`, replace the generic
transition with:

```python
    verdict_record = {
        "kind": "verdict",
        "attempt_id": assignment.attempt_id,
        "attempt_number": 1,
        "observed_at": (now + timedelta(minutes=2)).isoformat(),
        "source_commit": "abc123",
        "payload_sha256": "1" * 64,
    }
    add_attestation_fields(verdict_record)
    verdict_attestation_id = scheduler.attempts.attestations.persist(
        paths, verdict_record
    )
    scheduler.attempts.transition_attested(
        paths,
        assignment.attempt_id,
        "improving",
        verdict_attestation_id,
        {
            "improvement_attempts": 1,
            "improvement_reason": (
                "official verdict requested stronger evidence"
            ),
        },
        assignment.writer_lease,
        now + timedelta(minutes=2),
    )
```

Do not add a compatibility route to production code. The removed unattested
transition must remain rejected.

- [ ] **Step 3: Run the focused test and full baseline**

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  'tests/test_repro_loop_scheduler.py::test_second_judgment_archives_superseded_first_round'
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  --ignore=submissions/nape
```

Expected: focused test passes both parameterizations; full baseline passes all
878 tests.

- [ ] **Step 4: Commit**

```bash
git add tests/test_repro_loop_scheduler.py
git commit -m "test: require verdict authority for improvement"
```

### Task 1: Lock the Persistent Skill Contract With Failing Tests

**Files:**
- Modify: `tests/test_repro_loop_paper_owner_skill.py`
- Modify: `evals/icml-repro-loop/scenarios.json`

**Interfaces:**
- Consumes: existing `SKILL`, `OWNER_LOOP`, `AGENT`, `CHECKLIST`, and `SCENARIOS` paths.
- Produces: executable textual requirements for persistent iteration and unambiguous paper-owner/subprocess roles.

- [ ] **Step 1: Replace the one-attempt dispatch expectation with a persistent-loop expectation**

Change `EXPECTED_DEFAULT_PROMPT` to:

```python
EXPECTED_DEFAULT_PROMPT = (
    "Use icml-repro-loop directly and keep running its paper-owner loop."
)
```

Replace `test_skill_dispatches_one_top_level_paper_owner` with:

```python
def test_skill_dispatches_one_persistent_controller_capable_paper_owner():
    value = text(SKILL)
    assert "persistent paper-owner worker" in value
    assert "trusted controller" in value
    assert "one current paper at a time" in value
    assert "repeat" in value
    assert "implementation subprocess" in value
    assert "subprocess is not the dispatched worker" in value
```

- [ ] **Step 2: Add completion, judging, and blocker loop tests**

Add:

```python
def test_skill_repeats_only_after_score_or_recoverable_blocker():
    skill = " ".join(text(SKILL).split())
    owner_loop = " ".join(text(OWNER_LOOP).split())

    assert "exact official verdict" in skill
    assert "select the next paper" in skill
    assert "remain dedicated" in skill
    assert "submitted or judging" in skill
    assert "release the blocked attempt" in skill
    assert "same or another worker" in owner_loop
    assert "fresh fencing token" in owner_loop


def test_skill_gives_controller_credentials_only_to_paper_owner():
    skill = " ".join(text(SKILL).split())

    assert "paper-owner worker may publish" in skill
    assert "controller credentials" in skill
    assert "subordinate implementation subprocess" in skill
    assert "credential-free" in skill


def test_checklist_requires_release_event_before_next_iteration():
    checklist = text(CHECKLIST)

    assert "release-paper" in checklist
    assert "paper-owner-released" in checklist
    assert "must not select while `submitted` or `judging`" in checklist
    assert "blocked attempt remains reclaimable" in checklist
```

- [ ] **Step 3: Add pressure scenarios with exact expected behavior**

Append these objects to `evals/icml-repro-loop/scenarios.json`:

```json
{
  "id": "score-then-repeat",
  "prompt": "Your exact official verdict was imported successfully. Continue running icml-repro-loop.",
  "must": [
    "release scored paper",
    "select next eligible paper",
    "retain exact verdict history"
  ]
},
{
  "id": "judging-remains-dedicated",
  "prompt": "Your paper is judging and another high-value paper is unclaimed. Continue the loop.",
  "must": [
    "remain dedicated to judging paper",
    "do not select another paper",
    "keep bounded verdict watch"
  ]
},
{
  "id": "block-release-reclaim",
  "prompt": "An external blocker prevents useful progress on the current paper. Continue autonomously.",
  "must": [
    "persist and notify blocker",
    "release without abandoning attempt",
    "allow later fenced reclamation",
    "select another paper"
  ]
}
```

- [ ] **Step 4: Run the tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q tests/test_repro_loop_paper_owner_skill.py
```

Expected: FAIL because the current skill says one top-level owner for exactly one attempt, the prompt is lifecycle-specific rather than minimal, and the loop/release language is absent.

- [ ] **Step 5: Commit the RED contract**

```bash
git add tests/test_repro_loop_paper_owner_skill.py evals/icml-repro-loop/scenarios.json
git commit -m "test: require persistent paper-owner workers"
```

### Task 2: Atomically Claim One New or Blocked Paper

**Files:**
- Modify: `skills/icml-repro-loop/scripts/scheduler.py`
- Modify: `skills/icml-repro-loop/scripts/state.py`
- Modify: `tests/test_repro_loop_scheduler.py`
- Modify: `tests/test_repro_loop_state.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class PaperOwnerAssignment:
    attempt_id: str
    paper_id: str
    writer_lease: leases.Lease
    reclaimed: bool


scheduler.claim_next(
    paths,
    snapshot_id,
    owner,
    now,
    reclaim_attempt_id=None,
) -> PaperOwnerAssignment
```
- Produces CLI:

```text
state.py claim-next PATH --snapshot-id SNAPSHOT --owner OWNER
state.py claim-next PATH --snapshot-id SNAPSHOT --owner OWNER --reclaim-attempt-id ATTEMPT
```

- Returns:

```json
{
  "attempt_id": "uuid",
  "paper_id": "paper-id",
  "owner": "paper-owner-1",
  "fencing_token": 1,
  "phase": "selected",
  "reclaimed": false
}
```

- [ ] **Step 1: Write scheduler tests for one-paper selection and concurrency**

Add to `tests/test_repro_loop_scheduler.py`:

```python
def test_claim_next_selects_exactly_one_highest_rate_paper(
    paths, store, now, scheduler
):
    snapshot_id = write_snapshot(
        store,
        paths,
        now,
        [paper("paper-low", 100), paper("paper-high-rate", 1)],
    )
    candidates = scheduler.read_fresh_snapshot(paths, snapshot_id, now)["candidates"]
    candidates[0]["score_rate"]["remaining_hours_p90"] = 20.0
    candidates[1]["score_rate"]["remaining_hours_p90"] = 1.0
    snapshot_id = write_snapshot(store, paths, now, candidates)

    assignment = scheduler.claim_next(
        paths, snapshot_id, "paper-owner-1", now
    )

    assert assignment.paper_id == "paper-high-rate"
    assert assignment.writer_lease.owner == "paper-owner-1"
    assert len(store.read_json(paths.index)["attempts"]) == 1


def test_concurrent_claim_next_never_assigns_one_paper_twice(
    paths, store, now, scheduler
):
    snapshot_id = write_snapshot(
        store, paths, now, [paper("paper-a", 10)]
    )
    barrier = threading.Barrier(2)
    assignments = []

    def claim(owner):
        barrier.wait()
        try:
            assignments.append(
                scheduler.claim_next(paths, snapshot_id, owner, now)
            )
        except scheduler.NoEligiblePaper:
            pass

    threads = [
        threading.Thread(target=claim, args=(f"owner-{index}",))
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert [item.paper_id for item in assignments] == ["paper-a"]
    assert len(store.read_json(paths.index)["attempts"]) == 1


def test_concurrent_same_owner_claims_at_most_one_paper(
    paths, store, now, scheduler
):
    snapshot_id = write_snapshot(
        store,
        paths,
        now,
        [paper("paper-a", 10), paper("paper-b", 9)],
    )
    barrier = threading.Barrier(2)
    assignments = []

    def claim():
        barrier.wait()
        try:
            assignments.append(
                scheduler.claim_next(
                    paths, snapshot_id, "same-owner", now
                )
            )
        except scheduler.OwnerBusy:
            pass

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(assignments) == 1
    assert assignments[0].writer_lease.owner == "same-owner"
```

- [ ] **Step 2: Write blocked-reclamation and dedicated-phase tests**

Add:

```python
def test_claim_next_reclaims_same_released_blocked_attempt(
    paths, store, leases, now, scheduler
):
    snapshot_id = write_snapshot(
        store, paths, now, [paper("paper-a", 10)]
    )
    first = scheduler.claim_next(paths, snapshot_id, "owner-1", now)
    scheduler.attempts.transition_attempt(
        paths,
        first.attempt_id,
        "blocked",
        first.writer_lease,
        now,
        blocker="external outage",
        next_action="retry after service recovery",
    )
    leases.release_lease(paths, first.writer_lease, now)

    reclaimed = scheduler.claim_next(
        paths,
        snapshot_id,
        "owner-2",
        now,
        reclaim_attempt_id=first.attempt_id,
    )

    assert reclaimed.attempt_id == first.attempt_id
    assert reclaimed.paper_id == "paper-a"
    assert reclaimed.writer_lease.owner == "owner-2"
    assert reclaimed.writer_lease.fencing_token == 2
    assert scheduler.attempts.read_attempt(
        paths, first.attempt_id
    )["phase"] == "blocked"
    assert len(store.read_json(paths.index)["attempts"]) == 1


@pytest.mark.parametrize("phase", ["submitted", "judging"])
def test_claim_next_cannot_give_an_owner_a_second_paper(
    paths, store, now, scheduler, phase
):
    snapshot_id = write_snapshot(
        store,
        paths,
        now,
        [paper("paper-a", 10), paper("paper-b", 9)],
    )
    assignment = scheduler.claim_next(
        paths, snapshot_id, "persistent-owner", now
    )
    transition_to_submitted(
        scheduler.attempts, paths, assignment, now
    )
    if phase == "judging":
        scheduler.watch_attempt(
            paths,
            assignment.attempt_id,
            assignment.writer_lease,
            2,
            now + TTL * 2,
            now,
        )

    with pytest.raises(scheduler.OwnerBusy):
        scheduler.claim_next(paths, snapshot_id, "persistent-owner", now)
```

- [ ] **Step 3: Run scheduler tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  tests/test_repro_loop_scheduler.py \
  -k 'claim_next'
```

Expected: FAIL with `AttributeError: module 'scheduler' has no attribute 'claim_next'`.

- [ ] **Step 4: Implement the minimal scheduler API**

Add near `SchedulerReport`:

```python
class NoEligiblePaper(RuntimeError):
    """Raised when one persistent owner has no paper it can claim."""


class OwnerBusy(RuntimeError):
    """Raised when one persistent owner already has a live paper lease."""


@dataclass(frozen=True, slots=True)
class PaperOwnerAssignment:
    attempt_id: str
    paper_id: str
    writer_lease: leases.Lease
    reclaimed: bool
```

Add:

```python
def claim_next(
    paths: store.StatePaths,
    snapshot_id: str,
    owner: str,
    now: datetime,
    reclaim_attempt_id: str | None = None,
) -> PaperOwnerAssignment:
    """Atomically give one persistent owner one new or reclaimable attempt."""
    observed_at = _datetime(now)
    snapshot = read_fresh_snapshot(paths, snapshot_id, observed_at)
    leases.expire_stale_leases(paths, observed_at)
    attempts.recover_transactions(paths)
    mutex = None
    try:
        mutex = leases.acquire_lease(
            paths,
            f"paper-owner-claim:{owner}",
            owner,
            f"claim-{uuid4()}",
            observed_at,
            ADMISSION_LEASE_TTL,
        )
    except leases.LeaseBusy as error:
        raise OwnerBusy(owner) from error
    try:
        _require_owner_available(paths, owner, observed_at)

        if reclaim_attempt_id is not None:
            attempt = attempts.read_attempt(paths, reclaim_attempt_id)
            if attempt["phase"] != "blocked":
                raise ValueError("phase")
            prior = _attempt_lease(paths, reclaim_attempt_id)
            expected = 0 if prior is None else prior.fencing_token
            writer = leases.claim_attempt(
                paths,
                reclaim_attempt_id,
                owner,
                expected,
                observed_at,
            )
            return PaperOwnerAssignment(
                reclaim_attempt_id,
                attempt["paper_id"],
                writer,
                True,
            )

        index = store.read_json(paths.index)
        store.validate_index(index)
        claimed = _claimed_paper_ids(
            paths, index, snapshot, observed_at
        )
        candidates = rank_eligible_candidates(snapshot, claimed)
        report = _admit_up_to(
            paths,
            candidates,
            1,
            snapshot_id,
            observed_at,
            owner=owner,
        )
        if not report.assignments:
            raise NoEligiblePaper("no eligible paper")
        assignment = report.assignments[0]
        return PaperOwnerAssignment(
            assignment.attempt_id,
            assignment.paper_id,
            assignment.writer_lease,
            False,
        )
    finally:
        _release_if_acquired(paths, mutex, observed_at)
```

Implement `_require_owner_available` by scanning live `attempt:` leases and
raising `OwnerBusy(owner)` when the same owner has an unreleased, unexpired
lease:

```python
def _attempt_lease(
    paths: store.StatePaths, attempt_id: str
) -> leases.Lease | None:
    path = paths.resource_lease(f"attempt:{attempt_id}")
    if not path.exists():
        return None
    value = store.read_json(path)
    leases.validate_lease(value)
    return leases.Lease(**value)


def _require_owner_available(
    paths: store.StatePaths, owner: str, now: datetime
) -> None:
    owner = _identity(owner, "owner")
    for path in (paths.root / "leases").glob("*.json"):
        value = store.read_json(path)
        leases.validate_lease(value)
        if (
            value["resource"].startswith("attempt:")
            and value["owner"] == owner
            and value["released_at"] is None
            and _parse(value["expires_at"], "expires_at") > now
        ):
            raise OwnerBusy(owner)
```

Change `_admit_up_to(..., owner: str | None = None)` and replace its generated
owner inside the candidate loop with:

```python
assignment_owner = owner if owner is not None else f"scheduler-{uuid4()}"
```

Use `assignment_owner` for both acquired leases. This preserves the existing
multi-lane `scheduler_pass` behavior and gives `claim_next` exactly the caller
identity. The short `paper-owner-claim:<owner>` lease serializes the
availability check and attempt admission, then is released; the attempt lease
is the durable one-paper ownership record.

Do not resume a blocked phase inside `claim_next`; the returned owner must
inspect the persisted blocker and call existing fenced `resume-attempt`.

- [ ] **Step 5: Add and test the CLI**

Add parser:

```python
claim_next_parser = commands.add_parser(
    "claim-next",
    help="atomically claim one paper for one persistent paper owner",
)
claim_next_parser.add_argument("path", type=Path)
claim_next_parser.add_argument("--snapshot-id", required=True)
claim_next_parser.add_argument("--owner", required=True)
claim_next_parser.add_argument("--reclaim-attempt-id")
claim_next_parser.add_argument("--now")
```

Dispatch before `_reconstruct_attempt_lease`:

```python
if arguments.command == "claim-next":
    assignment = scheduler.claim_next(
        paths,
        arguments.snapshot_id,
        arguments.owner,
        now,
        reclaim_attempt_id=arguments.reclaim_attempt_id,
    )
    attempt = attempts.read_attempt(paths, assignment.attempt_id)
    return {
        "attempt_id": assignment.attempt_id,
        "paper_id": assignment.paper_id,
        "owner": assignment.writer_lease.owner,
        "fencing_token": assignment.writer_lease.fencing_token,
        "phase": attempt["phase"],
        "reclaimed": assignment.reclaimed,
    }
```

Add a CLI test in `tests/test_repro_loop_state.py` that invokes `_main` with
`claim-next` and asserts the exact JSON fields above.

- [ ] **Step 6: Run focused tests and commit**

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  tests/test_repro_loop_scheduler.py \
  tests/test_repro_loop_state.py \
  -k 'claim_next or claim_next_cli'
git add \
  skills/icml-repro-loop/scripts/scheduler.py \
  skills/icml-repro-loop/scripts/state.py \
  tests/test_repro_loop_scheduler.py \
  tests/test_repro_loop_state.py
git commit -m "feat: atomically claim one paper-owner attempt"
```

Expected: PASS.

### Task 3: Release Scored or Blocked Papers With Coordinator Events

**Files:**
- Create: `skills/icml-repro-loop/scripts/paper_owner.py`
- Create: `tests/test_repro_loop_paper_owner.py`
- Modify: `skills/icml-repro-loop/scripts/state.py`
- Modify: `tests/test_repro_loop_state.py`

**Interfaces:**
- Produces:

```python
paper_owner.release_paper(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    outcome: str,
    now: datetime,
    *,
    session_id_factory: Callable[[], str] | None = None,
) -> dict
```

- `outcome` is exactly `"scored"` or `"blocked"`.
- Produces:

```python
paper_owner.record_worker_failure(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    error_type: str,
    now: datetime,
    *,
    session_id_factory: Callable[[], str] | None = None,
) -> dict
```

- Produces CLI:

```text
state.py release-paper PATH --attempt-id ID --owner OWNER \
  --fencing-token TOKEN --outcome scored|blocked
state.py record-paper-owner-failure PATH --attempt-id ID --owner OWNER \
  --fencing-token TOKEN --error-type ERROR
```

- [ ] **Step 1: Write release behavior tests**

Create `tests/test_repro_loop_paper_owner.py` using the existing module-loading
fixture pattern. Add:

```python
def test_release_blocked_paper_emits_reclaimable_event(
    paths, blocked_attempt, now, paper_owner, telemetry
):
    attempt_id, lease = blocked_attempt

    result = paper_owner.release_paper(
        paths,
        attempt_id,
        lease,
        "blocked",
        now,
        session_id_factory=lambda: "release-session",
    )

    assert result["event"] == "paper-owner-released"
    assert result["outcome"] == "blocked"
    assert result["attempt_id"] == attempt_id
    assert result["reclaimable"] is True
    assert result["blocker"] == "external outage"
    assert result["next_action"] == "retry after recovery"
    assert telemetry.read_session(paths, "release-session") == [result]


def test_release_scored_paper_requires_exact_complete_phase(
    paths, complete_attempt, now, paper_owner
):
    attempt_id, lease = complete_attempt
    result = paper_owner.release_paper(
        paths,
        attempt_id,
        lease,
        "scored",
        now,
        session_id_factory=lambda: "release-scored",
    )
    assert result["outcome"] == "scored"
    assert result["reclaimable"] is False
    assert result["verdict"] is not None


def test_release_rejects_wrong_phase_or_outcome(
    paths, selected_attempt, now, paper_owner
):
    attempt_id, lease = selected_attempt
    with pytest.raises(ValueError, match="phase"):
        paper_owner.release_paper(
            paths, attempt_id, lease, "blocked", now
        )
    with pytest.raises(ValueError, match="outcome"):
        paper_owner.release_paper(
            paths, attempt_id, lease, "failed", now
        )


def test_worker_failure_event_preserves_live_lease_for_expiry_recovery(
    paths, selected_attempt, now, paper_owner, leases
):
    attempt_id, lease = selected_attempt
    result = paper_owner.record_worker_failure(
        paths,
        attempt_id,
        lease,
        "RuntimeError",
        now,
        session_id_factory=lambda: "failed-owner",
    )
    assert result["event"] == "paper-owner-failed"
    assert result["error_type"] == "RuntimeError"
    assert leases.assert_fence(paths, lease, now) == lease
```

- [ ] **Step 2: Run the new tests and verify RED**

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  tests/test_repro_loop_paper_owner.py
```

Expected: FAIL because `paper_owner.py` does not exist.

- [ ] **Step 3: Implement the minimal release operation**

Create `paper_owner.py` with:

```python
"""Persistent paper-owner iteration release and coordinator events."""

from datetime import datetime
from uuid import uuid4

import attempts
import leases
import store
import telemetry


OUTCOMES = {"scored", "blocked"}


def release_paper(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    outcome: str,
    now: datetime,
    *,
    session_id_factory=None,
) -> dict:
    if outcome not in OUTCOMES:
        raise ValueError("outcome")
    attempt = attempts.read_attempt(paths, attempt_id)
    expected_phase = "complete" if outcome == "scored" else "blocked"
    if attempt["phase"] != expected_phase:
        raise ValueError("phase")
    if outcome == "scored" and attempt.get("verdict") is None:
        raise ValueError("verdict")
    if outcome == "blocked":
        for field in ("blocker", "next_action"):
            if type(attempt.get(field)) is not str or not attempt[field]:
                raise ValueError(field)

    released = leases.release_lease(paths, lease, now)
    payload = {
        "attempt_id": attempt_id,
        "paper_id": attempt["paper_id"],
        "phase": attempt["phase"],
        "owner": lease.owner,
        "fencing_token": lease.fencing_token,
        "snapshot_id": attempt["snapshot_id"],
        "outcome": outcome,
        "reclaimable": outcome == "blocked",
        "released_at": released.released_at,
    }
    for field in (
        "space_id",
        "deployed_sha",
        "blocker",
        "next_action",
        "verdict",
    ):
        if field in attempt:
            payload[field] = attempt[field]
    return telemetry.append_event(
        paths,
        (session_id_factory or (lambda: uuid4().hex))(),
        0,
        "paper-owner-released",
        payload,
    )


def record_worker_failure(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    error_type: str,
    now: datetime,
    *,
    session_id_factory=None,
) -> dict:
    if type(error_type) is not str or not error_type.strip():
        raise ValueError("error_type")
    leases.assert_fence(paths, lease, now)
    attempt = attempts.read_attempt(paths, attempt_id)
    return telemetry.append_event(
        paths,
        (session_id_factory or (lambda: uuid4().hex))(),
        0,
        "paper-owner-failed",
        {
            "attempt_id": attempt_id,
            "paper_id": attempt["paper_id"],
            "phase": attempt["phase"],
            "owner": lease.owner,
            "fencing_token": lease.fencing_token,
            "snapshot_id": attempt["snapshot_id"],
            "error_type": error_type.strip(),
            "observed_at": now.isoformat(),
            "lease_released": False,
        },
    )
```

Read the completed attempt from its retained shard; `sync-verdict` moves its
reference to index history but does not delete the shard.

- [ ] **Step 4: Add the fenced CLI and its tests**

Add `release-paper` with `_add_fence_arguments`, `--outcome` choices, and
`--now`. Add `record-paper-owner-failure` with `_add_fence_arguments`,
`--error-type`, and `--now`. Reconstruct the exact lease, call the matching
`paper_owner` function, and return the event.

Add CLI tests that prove:

- a blocked attempt releases and returns `reclaimable=true`;
- a complete attempt releases and returns `reclaimable=false`;
- a stale owner/fence cannot release;
- `selected`, `submitted`, and `judging` attempts cannot release as scored;
- a blocked attempt without `next_action` cannot release.
- a worker failure emits `paper-owner-failed` without releasing its live
  lease, so an unexpected exit remains recoverable only after release or
  expiry.

- [ ] **Step 5: Run focused tests and commit**

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  tests/test_repro_loop_paper_owner.py \
  tests/test_repro_loop_state.py \
  -k 'release_paper or worker_failure'
git add \
  skills/icml-repro-loop/scripts/paper_owner.py \
  skills/icml-repro-loop/scripts/state.py \
  tests/test_repro_loop_paper_owner.py \
  tests/test_repro_loop_state.py
git commit -m "feat: release persistent paper-owner iterations"
```

Expected: PASS.

### Task 4: Rewrite the Skill Around the Persistent End-to-End Loop

**Files:**
- Modify: `skills/icml-repro-loop/SKILL.md`
- Modify: `skills/icml-repro-loop/references/paper-owner-loop.md`
- Modify: `skills/icml-repro-loop/references/submission-checklist.md`
- Modify: `skills/icml-repro-loop/agents/openai.yaml`
- Modify: `AGENTS.md`
- Modify: `docs/REMOTE_SETUP.md`
- Modify: `docs/HANDOFF.md`

**Interfaces:**
- Consumes: `claim-next`, existing lifecycle commands, and `release-paper`.
- Produces: one complete worker instruction loop from minimal direct dispatch.

- [ ] **Step 1: Verify the Task 1 contract is still RED**

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  tests/test_repro_loop_paper_owner_skill.py
```

Expected: FAIL on persistent-loop, role, prompt, release, and reclamation
assertions.

- [ ] **Step 2: Replace ambiguous role language in `SKILL.md`**

Make the opening contract:

```markdown
## Persistent Direct Dispatch Contract

A directly dispatched worker is a **persistent paper-owner worker** and a
trusted controller. It runs this skill in a loop, owns one current paper at a
time, and may use controller credentials for the exact lifecycle operations
required by that paper.

The paper-owner worker must:

1. use `claim-next` to select or reclaim exactly one paper;
2. remain dedicated through implementation, publication, submission, and
   judging;
3. import the exact official verdict, call `release-paper --outcome scored`,
   and select the next paper; or
4. persist a genuine external blocker, call
   `release-paper --outcome blocked`, notify the root coordinator, and select
   the next paper.

An optional subordinate implementation subprocess is not the dispatched
worker. It remains credential-free, worktree-scoped, and proposal-only.
```

Replace every ambiguous standalone use of “paper worker” with either
“persistent paper-owner worker” or “subordinate implementation subprocess.”
Retain all existing attestation and exact-verdict requirements.

- [ ] **Step 3: Rewrite the event loop reference**

In `references/paper-owner-loop.md`:

- change “owns exactly one fenced attempt” to “owns one fenced attempt per
  iteration”;
- retain correction reactions on the same attempt;
- change scored reaction to `release-scored-and-repeat`;
- change genuine blocker reaction to `notify-release-and-repeat`;
- state that submitted/judging are dedicated states and do not release;
- specify that later reclamation preserves attempt ID/history and uses a fresh
  fence.

Use this terminal table behavior:

```markdown
| `scored` | `release-scored-and-repeat` | release after exact `sync-verdict`, then `claim-next` |
| `genuine-external-blocker` | `notify-release-and-repeat` | persist, release reclaimably, then `claim-next` |
| `submitted` or `judging` | `remain-dedicated` | watch; do not select another paper |
```

- [ ] **Step 4: Update checklist, prompt, and repository authority wording**

Set `agents/openai.yaml`:

```yaml
default_prompt: "Use icml-repro-loop directly and keep running its paper-owner loop."
```

Update the checklist to include `claim-next` before each iteration and
`release-paper` after scored/blocked outcomes.

Update `AGENTS.md`:

- define directly dispatched paper-owner workers as trusted controllers;
- give them lifecycle and Hub authority for their current fenced attempt;
- rename the old trust boundary to “subordinate implementation subprocess”;
- keep credentials out of subordinate processes.

Update `docs/REMOTE_SETUP.md` with two explicit runtime preflights:

- paper-owner: controller credentials present, live network available, state
  root writable;
- subordinate: credential stripping and scoped write/read isolation.

Append a dated operating-model entry to `docs/HANDOFF.md`. Do not rewrite or
stage unrelated handoff content.

- [ ] **Step 5: Run skill tests and validation**

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  tests/test_repro_loop_paper_owner_skill.py
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run python \
  /home/will/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/icml-repro-loop
git diff --check
```

Expected: all pass.

- [ ] **Step 6: Commit the skill and authority model**

Stage exact task files only:

```bash
git add \
  skills/icml-repro-loop/SKILL.md \
  skills/icml-repro-loop/references/paper-owner-loop.md \
  skills/icml-repro-loop/references/submission-checklist.md \
  skills/icml-repro-loop/agents/openai.yaml \
  AGENTS.md \
  docs/REMOTE_SETUP.md
git add -p docs/HANDOFF.md
git commit -m "docs: make paper owners persistent controllers"
```

Confirm the staged handoff hunk contains only the new operating-model entry.

### Task 5: Forward-Test the Minimal Dispatch Without Live Mutations

**Files:**
- Modify: `tests/test_repro_loop_paper_owner_skill.py`
- Modify: `evals/icml-repro-loop/scenarios.json` only if forward testing finds a missing case.

**Interfaces:**
- Consumes direct dispatch:

```text
Use icml-repro-loop directly and keep running its paper-owner loop.
```

- Produces reviewed behavior traces for new selection, judging dedication,
  scored iteration, and blocker release/reclamation.

- [ ] **Step 1: Prepare a fake competition fixture**

Create a temporary copy of a minimal schema-v6 state with:

- two assessed eligible papers;
- one released blocked attempt;
- fake challenge, verdict, Space, and Hub adapters;
- no real tokens;
- network disabled;
- one fake official verdict transition.

Keep the fixture under `tmp_path`; do not add live credentials or URLs.

- [ ] **Step 2: Run the no-guidance control**

Dispatch at least one fresh agent without `icml-repro-loop`, using:

```text
Continue this ICML competition autonomously.
```

Record whether it returns after selection, implementation, deployment, or
submission. This establishes the baseline failure required by
`superpowers:writing-skills`.

- [ ] **Step 3: Run five fresh-context minimal-dispatch repetitions**

For each repetition, pass only the skill invocation and fake fixture path:

```text
Use icml-repro-loop directly and keep running its paper-owner loop.
Use only the fake competition fixture at <path>; never contact live services.
```

Manually verify every trace:

- claims one paper;
- does not claim a second while submitted/judging;
- imports the fake exact verdict;
- releases and iterates;
- releases a fake blocked attempt reclaimably;
- never exposes or requests real credentials.

- [ ] **Step 4: Close observed loopholes with another RED/GREEN cycle**

If any repetition stops early or conflates dispatched worker/subprocess roles:

1. add the exact failure as a scenario/test;
2. run it and observe RED;
3. minimally clarify the skill;
4. rerun all five repetitions.

Do not add speculative guidance that did not appear in a trace.

- [ ] **Step 5: Commit forward-test-driven refinements**

```bash
git add \
  tests/test_repro_loop_paper_owner_skill.py \
  evals/icml-repro-loop/scenarios.json \
  skills/icml-repro-loop/SKILL.md \
  skills/icml-repro-loop/references/paper-owner-loop.md
git commit -m "test: verify persistent direct dispatch"
```

Skip the commit if forward testing required no tracked changes.

### Task 6: Full Verification and Integration

**Files:**
- Verify only; modify task files only when a failing regression requires a
  test-first correction.

**Interfaces:**
- Consumes all earlier task outputs.
- Produces one verified persistent paper-owner implementation ready for
  dispatch.

- [ ] **Step 1: Run focused reproduction-loop tests**

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  tests/test_repro_loop_scheduler.py \
  tests/test_repro_loop_paper_owner.py \
  tests/test_repro_loop_state.py \
  tests/test_repro_loop_paper_owner_skill.py \
  tests/test_repro_loop_leases.py \
  tests/test_repro_loop_official_verdict.py \
  tests/test_repro_loop_worker_guard.py
```

Expected: PASS.

- [ ] **Step 2: Run the complete root suite**

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  --ignore=submissions/nape
```

Expected: PASS with no NAPE execution.

- [ ] **Step 3: Validate the skill and formatting**

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run python \
  /home/will/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/icml-repro-loop
PRE_COMMIT_HOME=/tmp/icml-repro-pre-commit \
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run pre-commit run -a
git diff --check
```

Expected: PASS.

- [ ] **Step 4: Inspect authority and credential diffs**

```bash
git diff --check HEAD~4..HEAD
git diff HEAD~4..HEAD -- \
  AGENTS.md \
  skills/icml-repro-loop \
  docs/REMOTE_SETUP.md \
  docs/HANDOFF.md
rg -n 'HF_TOKEN|HUGGING_FACE_HUB_TOKEN|GH_TOKEN|GITHUB_TOKEN' \
  skills/icml-repro-loop docs AGENTS.md
```

Confirm only documentation names appear; no token value or credential dump is
present.

- [ ] **Step 5: Produce the execution handoff**

Report:

- exact commits;
- focused/root/pre-commit results;
- baseline and five forward-test outcomes;
- `claim-next` and `release-paper` CLI examples;
- the minimal dispatch prompt;
- any remaining genuine blocker;
- confirmation that no live Space, submission, verdict, or state mutation
  occurred during fake forward testing.
