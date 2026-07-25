# Twenty-Paper Autonomous Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-current-paper state machine with a crash-safe coordinator that continuously maintains exactly 20 runnable, independently advancing reproduction attempts.

**Architecture:** `state/repro-loop.json` becomes a schema-v6 index whose mutable attempts, judgments, leases, snapshots, and transactions are stored in atomic JSON shards below `state/repro-loop/`. A bounded scheduler uses fenced leases to admit work until 20 runnable attempts exist, while each paper keeps a separate worktree, writer, evidence bundle, Space, and judgment record.

**Tech Stack:** Python 3.11+ standard library, JSON, `fcntl`, `pathlib`, `pytest`, subprocess concurrency tests, existing `uv`, Trackio, Hugging Face Hub.

## Global Constraints

- `max_runnable_attempts` is exactly `20`; completed and blocked attempts do not consume runnable capacity.
- Preserve the active EEG-FM-Bench attempt byte-for-byte at the field level during schema-v3 migration.
- Preserve the archived diffusion attempt, every rejection, and aggregate cost.
- One authoritative writer per attempt; different attempts and different Spaces may progress concurrently.
- Every mutating operation requires the current fencing token for its resource.
- Every JSON write is validated, locked, written to a sibling temporary file, fsynced, and atomically replaced.
- Multi-file operations use recoverable write-ahead transaction records.
- Subscription Codex and Antigravity usage costs USD 0.00; only metered external services consume cost reservations.
- CPU only. Never modify or validate `submissions/nape/` in place.
- Write and observe a failing test before each production change.

---

### Task 1: Sharded Store and Schema-v6 Index

**Files:**
- Create: `skills/icml-repro-loop/scripts/store.py`
- Create: `tests/test_repro_loop_store.py`
- Modify: `skills/icml-repro-loop/scripts/state.py`

**Interfaces:**
- Produces: `StatePaths`, `read_json`, `atomic_json_write`, `locked_json`, `new_index`, `validate_index`, `validate_attempt`, `validate_judgment`, `validate_snapshot`.
- Consumes: schema-v3 validation functions retained in `state.py` for migration only.

- [ ] **Step 1: Write failing layout and independent-write tests**

```python
def test_state_paths_create_independent_shards(tmp_path):
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    assert paths.attempt("attempt-1") == tmp_path / "repro-loop/attempts/attempt-1.json"
    assert paths.judgment("attempt-1") == tmp_path / "repro-loop/judgments/attempt-1.json"
    assert paths.lease("attempt--attempt-1") == tmp_path / "repro-loop/leases/attempt--attempt-1.json"


def test_new_index_has_five_runnable_slots():
    assert store.new_index()["max_runnable_attempts"] == 20


def test_independent_attempt_writes_do_not_lose_updates(tmp_path):
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    run_barrier_writers(paths, new_attempt("a1", "p1"), new_attempt("a2", "p2"))
    assert store.read_json(paths.attempt("a1"))["paper_id"] == "p1"
    assert store.read_json(paths.attempt("a2"))["paper_id"] == "p2"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_repro_loop_store.py -q`

Expected: collection fails because `store.py` and schema-v6 constructors do not exist.

- [ ] **Step 3: Implement paths, exact validators, locking, and atomic writes**

```python
@dataclass(frozen=True, slots=True)
class StatePaths:
    index: Path

    @property
    def root(self) -> Path:
        return self.index.with_suffix("")

    def attempt(self, attempt_id: str) -> Path:
        return self.root / "attempts" / f"{validate_id(attempt_id)}.json"

    def judgment(self, attempt_id: str) -> Path:
        return self.root / "judgments" / f"{validate_id(attempt_id)}.json"

    def lease(self, lease_id: str) -> Path:
        return self.root / "leases" / f"{validate_id(lease_id)}.json"


def new_index() -> dict:
    return {
        "version": 6,
        "max_runnable_attempts": 20,
        "attempts": {},
        "history": {},
        "rejections": [],
        "snapshots": {},
        "resource_limits": {"metered_api_reserved_usd": 10.0, "publication_per_provider": 5},
        "total_api_cost_usd": 0.0,
    }
```

Use identifier regex `[A-Za-z0-9._-]+`, `fcntl.flock`, sibling temp files,
`os.replace`, file `fsync`, then parent-directory `fsync`. Index attempt
references contain exactly `path`, `paper_id`, `phase`, and `updated_at`.

- [ ] **Step 4: Run focused and legacy tests**

Run: `uv run pytest tests/test_repro_loop_store.py tests/test_repro_loop_state.py -q`

Expected: PASS; existing schema-v3 tests remain green for migration helpers.

- [ ] **Step 5: Commit**

```bash
git add skills/icml-repro-loop/scripts/store.py skills/icml-repro-loop/scripts/state.py tests/test_repro_loop_store.py
git commit -m "feat: add sharded reproduction state store"
```

### Task 2: Transactional Schema-v3 Migration

**Files:**
- Create: `skills/icml-repro-loop/scripts/migrate_v6.py`
- Create: `tests/fixtures/repro-loop-v3-eeg.json`
- Create: `tests/test_repro_loop_migrate_v6.py`
- Modify: `skills/icml-repro-loop/scripts/state.py`

**Interfaces:**
- Consumes: `StatePaths`, schema-v3 `validate_state`, and store validators.
- Produces: `MigrationPlan`, `plan_v6_migration(v3)`, `apply_v6_migration(paths, plan)`, `recover_transactions(paths)`, `verify_semantic_equivalence(v3, paths)`.

- [ ] **Step 1: Write failing preservation and interruption tests**

```python
def test_migration_preserves_active_eeg_attempt(tmp_path):
    source = load_fixture("repro-loop-v3-eeg.json")
    paths = migrate_fixture(tmp_path, source)
    attempt = only_active_attempt(paths)
    for field in (
        "paper_id", "title", "slug", "project_path", "upstream_revision",
        "target_claims", "design_approved", "estimated_api_cost_usd",
    ):
        assert attempt[field] == source["current"][field]
    assert attempt["phase"] == "implementing"


def test_migration_preserves_history_rejections_and_cost(tmp_path):
    source = load_fixture("repro-loop-v3-eeg.json")
    paths = migrate_fixture(tmp_path, source)
    index = store.read_json(paths.index)
    assert len(index["history"]) == len(source["history"])
    assert index["rejections"] == source["rejections"]
    assert index["total_api_cost_usd"] == source["total_api_cost_usd"]


@pytest.mark.parametrize("fail_after", range(1, 7))
def test_interrupted_migration_recovers_atomically(tmp_path, fail_after):
    paths = interrupt_migration(tmp_path, fail_after)
    migrate_v6.recover_transactions(paths)
    assert_no_dangling_index_references(paths)
```

- [ ] **Step 2: Run migration tests and verify RED**

Run: `uv run pytest tests/test_repro_loop_migrate_v6.py -q`

Expected: FAIL because migration planning and transaction recovery are absent.

- [ ] **Step 3: Implement deterministic planning and write-ahead recovery**

```python
ATTEMPT_NAMESPACE = UUID("b3f93d5c-2d22-4c66-9f70-b7c15ef4bb59")


def attempt_id(paper_id: str, work_kind: str, attempt_number: int) -> str:
    return str(uuid5(ATTEMPT_NAMESPACE, f"{paper_id}:{work_kind}:{attempt_number}"))


def plan_v6_migration(v3: dict) -> MigrationPlan:
    legacy_state.validate_state(v3)
    return MigrationPlan.from_v3(v3, max_runnable_attempts=20)
```

The transaction manifest records source SHA-256, every target path and hash,
staging paths, and status. Install verified shards first and the index last.
Retain `state/repro-loop/v3-backups/<source-sha256>.json`. Re-running migration
must return the existing verified result without changing bytes.

- [ ] **Step 4: Verify dry-run and semantic equivalence**

Run:

```bash
uv run pytest tests/test_repro_loop_migrate_v6.py -q
uv run python skills/icml-repro-loop/scripts/state.py migrate-v6 state/repro-loop.json --dry-run
```

Expected: PASS; dry-run reports one active EEG attempt, one archived diffusion
attempt, all rejections, 20 runnable slots, the exact schema-v3 source SHA-256,
and no writes.

- [ ] **Step 5: Commit**

```bash
git add skills/icml-repro-loop/scripts/migrate_v6.py skills/icml-repro-loop/scripts/state.py tests/fixtures/repro-loop-v3-eeg.json tests/test_repro_loop_migrate_v6.py
git commit -m "feat: migrate single-paper state to twenty attempts"
```

### Task 3: Fenced Resource Leases

**Files:**
- Create: `skills/icml-repro-loop/scripts/leases.py`
- Create: `tests/test_repro_loop_leases.py`
- Modify: `skills/icml-repro-loop/scripts/store.py`

**Interfaces:**
- Produces: `Lease`, `LeaseBusy`, `acquire_lease`, `renew_lease`, `release_lease`, `expire_stale_leases`, `reserve_metered_cost`, `reconcile_metered_cost`.

- [ ] **Step 1: Write failing lease and fencing tests**

```python
def test_different_attempt_writers_acquire_concurrently(paths, now):
    first, second = acquire_with_barrier(paths, "attempt:a1", "attempt:a2", now)
    assert first.resource != second.resource


def test_different_spaces_publish_concurrently_but_same_space_serializes(paths, now):
    first, second = acquire_with_barrier(
        paths, "space:hf--org--paper-a", "space:hf--org--paper-b", now
    )
    assert first.resource != second.resource
    with pytest.raises(leases.LeaseBusy):
        leases.acquire_lease(
            paths,
            "space:hf--org--paper-a",
            "worker-3",
            "a3",
            now,
            TTL,
        )


def test_stale_fencing_token_cannot_write(paths, now):
    old = leases.acquire_lease(paths, "attempt:a1", "worker-1", "a1", now, TTL)
    new = leases.acquire_lease(paths, "attempt:a1", "worker-2", "a1", now + TTL, TTL)
    with pytest.raises(leases.StaleFence):
        attempts.update_attempt(paths, "a1", old, {"phase": "validated"})
    assert new.fencing_token > old.fencing_token


def test_subscription_agents_reserve_zero_metered_cost(paths, now):
    reservation = leases.reserve_metered_cost(paths, "a1", "codex-subscription", 0.0, now)
    assert reservation.amount_usd == 0.0
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/test_repro_loop_leases.py -q`

Expected: FAIL because lease primitives do not exist.

- [ ] **Step 3: Implement leases with monotonic fencing tokens**

```python
@dataclass(frozen=True, slots=True)
class Lease:
    resource: str
    owner: str
    attempt_id: str
    acquired_at: str
    expires_at: str
    fencing_token: int
    released_at: str | None
```

Acquire under the lease-file lock. A live unreleased lease raises `LeaseBusy`.
An expired lease increments the prior fencing token. Cost reservations reject
negative/non-finite amounts, per-paper totals above USD 10, and global reserved
totals above the index limit. Subscription providers always reserve `0.0`.

- [ ] **Step 4: Run lease, store, and state tests**

Run: `uv run pytest tests/test_repro_loop_leases.py tests/test_repro_loop_store.py tests/test_repro_loop_state.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/icml-repro-loop/scripts/leases.py skills/icml-repro-loop/scripts/store.py tests/test_repro_loop_leases.py
git commit -m "feat: fence concurrent reproduction resources"
```

### Task 4: Independent Attempt Lifecycle and Design Review

**Files:**
- Create: `skills/icml-repro-loop/scripts/attempts.py`
- Create: `tests/test_repro_loop_attempts.py`
- Modify: `skills/icml-repro-loop/scripts/state.py`

**Interfaces:**
- Produces: `create_attempt`, `read_attempt`, `update_attempt`, `transition_attempt`, `record_design`, `record_design_review`, `runnable_attempt_ids`.
- Consumes: store validation and attempt-writer leases.

- [ ] **Step 1: Write failing independent-transition tests**

```python
def test_blocking_one_attempt_does_not_change_another(paths, attempts_and_leases, now):
    a1, l1, a2, _ = attempts_and_leases
    attempts.transition_attempt(paths, a1, "blocked", l1, now, blocker="missing dataset")
    assert attempts.read_attempt(paths, a1)["phase"] == "blocked"
    assert attempts.read_attempt(paths, a2)["phase"] == "implementing"


def test_different_agent_may_approve_design(paths, pending_attempt, now):
    attempts.record_design(paths, pending_attempt.id, pending_attempt.lease, "author-a", "design.md", now)
    attempts.record_design_review(paths, pending_attempt.id, pending_attempt.lease, "reviewer-b", "approved", now)
    assert attempts.read_attempt(paths, pending_attempt.id)["phase"] == "implementing"


def test_design_author_cannot_self_approve(paths, pending_attempt, now):
    attempts.record_design(paths, pending_attempt.id, pending_attempt.lease, "agent-a", "design.md", now)
    with pytest.raises(ValueError, match="reviewer"):
        attempts.record_design_review(paths, pending_attempt.id, pending_attempt.lease, "agent-a", "approved", now)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/test_repro_loop_attempts.py -q`

Expected: FAIL because attempt lifecycle functions are absent.

- [ ] **Step 3: Implement exact phase transitions and per-attempt history**

Allow the existing phase graph independently per attempt. Every transition
appends `{from, to, at, owner, fencing_token, snapshot_id}`. `blocked` requires
a nonempty blocker and does not appear in `runnable_attempt_ids`. `complete`
moves the index reference from `attempts` to `history`; abandonment still
requires explicit `abandon=True`.

- [ ] **Step 4: Run focused and integration tests**

Run: `uv run pytest tests/test_repro_loop_attempts.py tests/test_repro_loop_leases.py tests/test_repro_loop_store.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/icml-repro-loop/scripts/attempts.py skills/icml-repro-loop/scripts/state.py tests/test_repro_loop_attempts.py
git commit -m "feat: isolate reproduction attempt lifecycles"
```

### Task 5: Twenty-Paper Scheduler and Independent Judgments

**Files:**
- Create: `skills/icml-repro-loop/scripts/scheduler.py`
- Create: `tests/test_repro_loop_scheduler.py`
- Modify: `skills/icml-repro-loop/scripts/state.py`

**Interfaces:**
- Produces: `SchedulerAssignment`, `SchedulerReport`, `scheduler_pass`, `watch_attempt`, `record_poll`, `record_verdict`.
- Consumes: one immutable snapshot, attempt lifecycle, and fenced leases.

- [ ] **Step 1: Write failing 20-paper admission/refill tests**

```python
def test_scheduler_admits_exactly_five_runnable_attempts(paths, snapshot_id, now):
    report = scheduler.scheduler_pass(paths, snapshot_id, now)
    assert len(report.created_attempt_ids) == 5
    assert len(attempts.runnable_attempt_ids(paths)) == 5


def test_scheduler_refills_completed_and_blocked_slots(paths, five_attempts, snapshot_id, now):
    complete(five_attempts[0])
    block(five_attempts[1], "artifact unavailable")
    report = scheduler.scheduler_pass(paths, snapshot_id, now)
    assert len(report.created_attempt_ids) == 2
    assert len(attempts.runnable_attempt_ids(paths)) == 5


def test_duplicate_paper_can_only_be_claimed_once(paths, duplicate_candidates, now):
    reports = run_simultaneous_scheduler_passes(paths, duplicate_candidates, now)
    assert sum("paper-a" in report.paper_ids for report in reports) == 1


def test_pending_judgment_does_not_block_new_admission(paths, submitted_attempt, snapshot_id, now):
    scheduler.watch_attempt(paths, submitted_attempt.id, 12, now + timedelta(hours=24), now)
    report = scheduler.scheduler_pass(paths, snapshot_id, now)
    assert report.created_attempt_ids
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/test_repro_loop_scheduler.py -q`

Expected: FAIL because scheduler and judgment operations are absent.

- [ ] **Step 3: Implement bounded scheduling**

```python
def scheduler_pass(paths: StatePaths, snapshot_id: str, now: datetime) -> SchedulerReport:
    snapshot = read_fresh_snapshot(paths, snapshot_id, now)
    vacancies = read_json(paths.index)["max_runnable_attempts"] - len(runnable_attempt_ids(paths))
    candidates = rank_eligible_candidates(snapshot)
    return admit_up_to(paths, candidates, max(0, vacancies), snapshot_id, now)
```

Before admission, exclude all paper IDs appearing in active attempts, history,
rejections, verdicts, queued submissions, tagged Spaces, or live candidate
leases. Acquire candidate and writer leases before creating each attempt.
Judgment files have independent locks and retain exact Space ID, submitted SHA,
attempt number, target claims, bounded polls, raw verdict, normalized verdict,
and source revision.

- [ ] **Step 4: Run scheduler and concurrency suites**

Run: `uv run pytest tests/test_repro_loop_scheduler.py tests/test_repro_loop_attempts.py tests/test_repro_loop_leases.py tests/test_repro_loop_store.py -q`

Expected: PASS without timing-only sleeps.

- [ ] **Step 5: Commit**

```bash
git add skills/icml-repro-loop/scripts/scheduler.py skills/icml-repro-loop/scripts/state.py tests/test_repro_loop_scheduler.py
git commit -m "feat: maintain 20 concurrent reproduction attempts"
```

### Task 6: CLI, Skill Contract, and Operational Documentation

**Files:**
- Create: `skills/icml-repro-loop/scripts/refresh.py`
- Create: `tests/test_repro_loop_refresh.py`
- Modify: `skills/icml-repro-loop/scripts/state.py`
- Modify: `skills/icml-repro-loop/SKILL.md`
- Modify: `skills/icml-repro-loop/references/selection-rubric.md`
- Modify: `skills/icml-repro-loop/references/submission-checklist.md`
- Modify: `AGENTS.md`
- Modify: `README.md`, `docs/REMOTE_SETUP.md`
- Test: `tests/test_repro_loop_state.py`, `tests/test_repro_loop_scheduler.py`

**Interfaces:**
- Produces: `refresh.fetch_live_snapshot(client, observed_at, assessment_input=None) -> dict`, `refresh.persist_snapshot(paths, snapshot) -> str`, and `refresh.read_snapshot(paths, snapshot_id) -> dict`, with IDs verified as canonical JSON SHA-256 hashes.
- CLI commands: `migrate-v6`, `refresh-live`, `show-snapshot`, `list-attempts`, `show-attempt`, `scheduler-pass`, `transition-attempt`, `record-design`, `review-design`, `watch-attempt`, `record-poll`, `record-verdict`.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_list_attempts_requires_no_ambiguous_current(tmp_path):
    result = run_state_cli(tmp_path, "list-attempts", "--phase", "implementing")
    assert result.returncode == 0
    assert all(item["phase"] == "implementing" for item in json.loads(result.stdout))


def test_mutation_requires_attempt_and_fencing_token(tmp_path):
    result = run_state_cli(tmp_path, "transition-attempt", "--attempt-id", "a1", "validated")
    assert result.returncode != 0
    assert "fencing-token" in result.stderr


def test_live_snapshot_id_is_content_addressed(tmp_path, recorded_hub_client):
    snapshot = refresh.fetch_live_snapshot(recorded_hub_client, "2026-07-24T18:00:00+00:00")
    snapshot_id = refresh.persist_snapshot(store.StatePaths(tmp_path / "repro-loop.json"), snapshot)
    assert snapshot_id == hashlib.sha256(canonical_json(snapshot)).hexdigest()
    assert snapshot["sources"]["challenge"]["revision"]
    assert snapshot["sources"]["verdicts"]["revision"]
    assert snapshot["spaces"]
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `uv run pytest tests/test_repro_loop_state.py tests/test_repro_loop_scheduler.py tests/test_repro_loop_refresh.py -q`

Expected: FAIL because multi-attempt CLI commands are not registered.

- [ ] **Step 3: Implement explicit CLI commands and update contracts**

Commands must never infer a current attempt. Every attempt mutation takes
`--attempt-id`, `--owner`, and `--fencing-token`. Documentation names all
materially affected attempts and describes 20-paper refill, per-attempt
blockers, independent design review, and subscription-cost accounting.

`refresh-live` is the only network-aware state command. Without assessments it
persists raw discovery for network-free `show-snapshot` inspection. With
`--assessments-json PATH`, it fetches the exact
`ICML-2026-agent-repro/challenge` and `ICML-2026-agent-repro/verdicts`
revisions, current `challenge.json` papers/claims, and current `paper-*` Space
identities, queues, and verdicts. Raw metadata remains unassessed. Assessed
refresh merges only explicit assessments matching the newly fetched revision,
pinned paper, and extracted claims; revision drift is an explicit failure. It
verifies the canonical assessment-document hash, constructs canonical JSON,
writes `snapshots/<sha256>.json` once, and publishes only the immutable ID into
the index. Recorded test clients replace every network call in unit tests.

- [ ] **Step 4: Run all local validation**

Run:

```bash
uv run pytest -q
CODEX_HOME=${CODEX_HOME:-$HOME/.codex}
uv run "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" skills/icml-repro-loop
uv run pre-commit run -a
```

Expected: all tests and hooks pass; NAPE is not executed or formatted.

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md README.md docs/REMOTE_SETUP.md skills/icml-repro-loop tests/test_repro_loop_state.py tests/test_repro_loop_scheduler.py tests/test_repro_loop_refresh.py
git commit -m "docs: operate twenty autonomous paper attempts"
```

### Task 7: Migrate EEG-FM-Bench and Fill the Twenty-Paper Pool

**Files:**
- Modify: `state/repro-loop.json`
- Create: `state/repro-loop/attempts/*.json`
- Create: `state/repro-loop/snapshots/*.json`
- Create: `state/repro-loop/v3-backups/*.json`

**Interfaces:**
- Consumes: validated migration and scheduler CLIs.
- Produces: one migrated EEG-FM-Bench attempt plus up to 19 live-selected
  attempts with isolated worktrees and writer leases.

- [ ] **Step 1: Dry-run and verify migration**

Run:

```bash
uv run python skills/icml-repro-loop/scripts/state.py migrate-v6 state/repro-loop.json --dry-run
```

Expected: one active `vGeNaFHdET` attempt in `implementing`, one archived
`HMu24dTKkJ` attempt, all recorded rejections, no semantic differences.

- [ ] **Step 2: Apply migration and verify bytes/references**

Run:

```bash
expected_source_sha256="<source_state_sha256 from the reviewed dry-run>"
uv run python skills/icml-repro-loop/scripts/state.py migrate-v6 \
  state/repro-loop.json --apply \
  --expected-source-sha256 "$expected_source_sha256"
uv run python skills/icml-repro-loop/scripts/state.py list-attempts state/repro-loop.json
```

Expected: schema version 6, one runnable attempt, nineteen vacancies, and a valid
hash-addressed schema-v3 backup.

- [ ] **Step 3: Discover, inspect, assess in parallel, and persist an assessed snapshot**

Run:

```bash
raw_snapshot_id="$(
  uv run python skills/icml-repro-loop/scripts/state.py refresh-live \
    state/repro-loop.json \
  | uv run python -c 'import json, sys; print(json.load(sys.stdin)["snapshot_id"])'
)"
uv run python skills/icml-repro-loop/scripts/state.py show-snapshot \
  state/repro-loop.json --snapshot-id "$raw_snapshot_id" \
  > state/current-challenge-snapshot.json

assessment_path=state/candidate-assessments.json
# Dispatch independent agents in parallel over disjoint candidates from the raw
# snapshot. Merge their explicit records into one assessment document carrying
# the raw snapshot's exact challenge revision.
snapshot_id="$(
  uv run python skills/icml-repro-loop/scripts/state.py refresh-live \
    state/repro-loop.json --assessments-json "$assessment_path" \
  | uv run python -c 'import json, sys; print(json.load(sys.stdin)["snapshot_id"])'
)"
test -n "$snapshot_id"
```

Expected: the raw snapshot exposes the exact current candidates and claims
without eligibility fields. Assessment agents independently inspect primary
artifacts; the merged file records assessor/time and explicit score, selected
live claims, upstream revision, artifact/CPU/safety/license decisions, and
finite estimated API cost for each assessed candidate. Assessed refresh fetches
the live sources again. If it fails with `challenge_revision`, discard the
assessment document and restart Step 3 from a new raw refresh; never rewrite its
revision or retry stale assessments. The resulting assessed snapshot records
source URLs, revisions, observation time, canonical assessment hash, and
content hashes; stdout returns its SHA-256 ID.

- [ ] **Step 4: Run one scheduler pass**

Run:

```bash
uv run python skills/icml-repro-loop/scripts/state.py scheduler-pass \
  state/repro-loop.json --snapshot-id "$snapshot_id" \
  > state/current-scheduler-pass.json

uv run python -c \
  'import json,sys; data=json.load(open(sys.argv[1])); [print(a["attempt_id"], a["paper_id"], a["owner"], a["fencing_token"], sep="\t") for a in data["assignments"]]' \
  state/current-scheduler-pass.json \
| while IFS="$(printf '\t')" read -r attempt_id paper_id owner fencing_token; do
    uv run python skills/icml-repro-loop/scripts/state.py transition-attempt \
      state/repro-loop.json design-pending \
      --attempt-id "$attempt_id" --owner "$owner" \
      --fencing-token "$fencing_token"
  done
```

Expected: exactly four newly admitted attempts and 20 runnable attempts
total. Scheduler output supplies each distinct paper's attempt ID, paper ID,
writer owner, and fencing token in `selected`; the explicit transitions put
each new attempt in `design-pending`.

- [ ] **Step 5: Dispatch paper-specific design authors and independent reviewers**

For each row captured in `state/current-scheduler-pass.json`, dispatch one
design author. After that author commits the design, use the row's exact fenced
identity (shown below), then dispatch a different read-only reviewer:

```bash
uv run python skills/icml-repro-loop/scripts/state.py record-design \
  state/repro-loop.json \
  --attempt-id "$attempt_id" --owner "$owner" \
  --fencing-token "$fencing_token" \
  --author "$design_author" --design-path "$committed_design_path"

test "$design_reviewer" != "$design_author"
uv run python skills/icml-repro-loop/scripts/state.py review-design \
  state/repro-loop.json \
  --attempt-id "$attempt_id" --owner "$owner" \
  --fencing-token "$fencing_token" \
  --reviewer "$design_reviewer" --decision "$review_decision"
```

Persist the committed design path, author identity, review verdict, and
transition. Approved attempts advance independently to `implementing`; rejected
designs revise without stopping other lanes.

- [ ] **Step 6: Verify the 20-paper pool**

Run:

```bash
uv run python skills/icml-repro-loop/scripts/state.py list-attempts state/repro-loop.json --runnable
uv run pytest -q
uv run pre-commit run -a
```

Expected: up to 20 runnable attempts, no duplicate paper/Space identities, and
root tests and hooks pass. Every attempt shard records its paper, phase,
worktree, owner, next action, and blocker.

- [ ] **Step 7: Commit operational migration**

```bash
git add state/repro-loop.json state/repro-loop
git commit -m "chore: activate twenty autonomous paper attempts"
```
