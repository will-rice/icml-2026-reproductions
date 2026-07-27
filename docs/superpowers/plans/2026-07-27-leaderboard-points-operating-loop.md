# Leaderboard-Points Operating Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rank and operate reproduction attempts by conservative expected leaderboard points per remaining hour, with real process telemetry and nonblocking queue reporting.

**Architecture:** Add a versioned score-rate envelope to assessed candidates, then make the existing scheduler order eligible papers by a deterministic points-per-hour key. Add append-only controller telemetry around the real worker subprocess and controller stages, and expose a read-only report/census through the existing state CLI. Preserve every existing lifecycle attestation and authority boundary.

**Tech Stack:** Python 3.12, standard-library dataclasses/JSON/subprocess/time, pytest, the existing schema-v6 JSON shard store, Hugging Face Hub snapshots, pre-commit.

## Global Constraints

- Official claim points are exactly: `verified=2`, `falsified=2`, `toy=1`, `inconclusive=0`.
- Scheduling estimates never become evidence or official verdicts.
- Existing schema-v6 snapshots and assessment records remain readable.
- New admissions require a score-rate envelope; legacy assessment `score` is not a ranking substitute.
- Workers have internet and whole-worktree write access but no Hub/GitHub credentials or coordinator-state authority.
- Autonomous GPU work remains ineligible.
- Estimated paid API cost remains at most USD 10 per paper.
- Judging and blocked attempts do not consume runnable implementation capacity.
- Telemetry records actual process/stage boundaries; Git and phase timestamps are never reported as worker runtime.
- Do not run, modify, test, or format `submissions/nape/`.
- Preserve the user's existing unstaged `docs/HANDOFF.md` changes.

## File Structure

- Create `skills/icml-repro-loop/scripts/score_rate.py`: score mapping, envelope validation, expected points, priority, and deterministic ranking key.
- Create `skills/icml-repro-loop/scripts/telemetry.py`: append-only event records and derived timing summaries.
- Create `skills/icml-repro-loop/scripts/score_report.py`: official-score aggregation, candidate queue report, and broad unclaimed census.
- Modify `skills/icml-repro-loop/scripts/refresh.py`: accept both legacy assessments and new score-rate assessments.
- Modify `skills/icml-repro-loop/scripts/scheduler.py`: require score-rate data for new admissions and rank by priority.
- Modify `skills/icml-repro-loop/scripts/store.py`: resolve telemetry event paths.
- Modify `skills/icml-repro-loop/scripts/worker_guard.py`: carry contract identity and wrap the actual worker subprocess.
- Modify `skills/icml-repro-loop/scripts/state.py`: add `run-worker`, `score-report`, and `candidate-census`; record controller-stage telemetry.
- Modify focused tests under `tests/` for each component.
- Modify `skills/icml-repro-loop/SKILL.md`, `references/selection-rubric.md`, `references/submission-checklist.md`, and `docs/REMOTE_SETUP.md`: document the new required commands and semantics.
- Preserve the already-modified `docs/HANDOFF.md`; the controller appends the
  implementation milestone without staging or rewriting the unowned edits.

---

### Task 1: Versioned Score-Rate Assessment And Ranking

**Files:**
- Create: `skills/icml-repro-loop/scripts/score_rate.py`
- Modify: `skills/icml-repro-loop/scripts/refresh.py:24-35`
- Modify: `skills/icml-repro-loop/scripts/refresh.py:555-674`
- Modify: `skills/icml-repro-loop/scripts/scheduler.py:146-181`
- Create: `tests/test_repro_loop_score_rate.py`
- Modify: `tests/test_repro_loop_refresh.py:141-170`
- Modify: `tests/test_repro_loop_scheduler.py:77-112`

**Interfaces:**
- Consumes: live claim records containing exact `text`, plus assessment-provided probabilities and P90 time.
- Produces:
  - `score_rate.claim_points(status: str) -> int`
  - `score_rate.validate_envelope(value: object, live_claims: list[dict]) -> None`
  - `score_rate.expected_points(value: dict) -> float`
  - `score_rate.priority(value: dict) -> float`
  - `score_rate.ranking_key(candidate: dict) -> tuple`
  - `refresh._valid_assessment_record(record, *, require_score_rate=False) -> bool`

- [ ] **Step 1: Write failing score and ranking tests**

```python
# tests/test_repro_loop_score_rate.py
from pathlib import Path
import hashlib
import importlib
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/icml-repro-loop/scripts"
sys.path.insert(0, str(SCRIPTS))
score_rate = importlib.import_module("score_rate")


def envelope(*, hours=2.0, deadline_probability=0.8, reusable=False):
    return {
        "claim_expectations": [
            {
                "challenge_claim_sha256": hashlib.sha256(b"Claim A").hexdigest(),
                "p_verified": 0.5,
                "p_falsified": 0.25,
                "p_toy": 0.1,
            },
            {
                "challenge_claim_sha256": hashlib.sha256(b"Claim B").hexdigest(),
                "p_verified": 0.0,
                "p_falsified": 0.5,
                "p_toy": 0.25,
            },
        ],
        "judged_before_deadline_probability": deadline_probability,
        "remaining_hours_p90": hours,
        "reusable_implementation": reusable,
        "direct_artifact_score": 4,
        "full_score_claim_paths": 2,
        "remaining_time_variance_hours2": 0.25,
        "primary_risk": "Artifact schema may have drifted.",
    }


def test_official_claim_point_mapping_is_exact():
    assert score_rate.claim_points("verified") == 2
    assert score_rate.claim_points("falsified") == 2
    assert score_rate.claim_points("toy") == 1
    assert score_rate.claim_points("inconclusive") == 0
    with pytest.raises(ValueError, match="status"):
        score_rate.claim_points("unknown")


def test_expected_points_and_priority_follow_approved_formula():
    value = envelope()
    assert score_rate.expected_points(value) == pytest.approx(2.85)
    assert score_rate.priority(value) == pytest.approx(1.14)


def test_envelope_binds_every_live_claim_once():
    live_claims = [{"text": "Claim A"}, {"text": "Claim B"}]
    score_rate.validate_envelope(envelope(), live_claims)
    invalid = envelope()
    invalid["claim_expectations"].pop()
    with pytest.raises(ValueError, match="claim_expectations"):
        score_rate.validate_envelope(invalid, live_claims)


def test_probability_mass_cannot_exceed_one():
    invalid = envelope()
    invalid["claim_expectations"][0]["p_toy"] = 0.3
    with pytest.raises(ValueError, match="probability"):
        score_rate.validate_envelope(
            invalid, [{"text": "Claim A"}, {"text": "Claim B"}]
        )
```

Add a scheduler test that gives a lower legacy rubric score to the higher
points-per-hour paper and asserts the latter is admitted first. Add a second
test asserting a candidate with a score-rate envelope but `cpu_only=False`
remains ineligible. Add table-driven ranking tests covering deadline
probability, remaining P90 time, and each deterministic tie-break in order:
reusable implementation, direct artifacts, full-score paths, time variance,
paid cost, then paper ID.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_score_rate.py \
  tests/test_repro_loop_refresh.py \
  tests/test_repro_loop_scheduler.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named
'score_rate'`.

- [ ] **Step 3: Implement the score-rate module and versioned assessment validation**

```python
# skills/icml-repro-loop/scripts/score_rate.py
from __future__ import annotations

import hashlib
import math
import re

POINTS = {"verified": 2, "falsified": 2, "toy": 1, "inconclusive": 0}
ENVELOPE_KEYS = {
    "claim_expectations",
    "judged_before_deadline_probability",
    "remaining_hours_p90",
    "reusable_implementation",
    "direct_artifact_score",
    "full_score_claim_paths",
    "remaining_time_variance_hours2",
    "primary_risk",
}
CLAIM_KEYS = {
    "challenge_claim_sha256",
    "p_verified",
    "p_falsified",
    "p_toy",
}


def claim_points(status: str) -> int:
    if type(status) is not str or status.casefold() not in POINTS:
        raise ValueError("status")
    return POINTS[status.casefold()]


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(field)
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(field)
    return number


def validate_envelope(value: object, live_claims: list[dict]) -> None:
    if type(value) is not dict or set(value) != ENVELOPE_KEYS:
        raise ValueError("score_rate")
    expectations = value["claim_expectations"]
    if type(expectations) is not list or len(expectations) != len(live_claims):
        raise ValueError("claim_expectations")
    expected_digests = [
        hashlib.sha256(claim["text"].encode("utf-8")).hexdigest()
        for claim in live_claims
    ]
    actual_digests = []
    for record in expectations:
        if type(record) is not dict or set(record) != CLAIM_KEYS:
            raise ValueError("claim_expectations")
        probabilities = [
            _probability(record[field], field)
            for field in ("p_verified", "p_falsified", "p_toy")
        ]
        if sum(probabilities) > 1.0 + 1e-12:
            raise ValueError("probability")
        digest = record["challenge_claim_sha256"]
        if (
            type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("challenge_claim_sha256")
        actual_digests.append(digest)
    if actual_digests != expected_digests:
        raise ValueError("claim_expectations")
    _probability(
        value["judged_before_deadline_probability"],
        "judged_before_deadline_probability",
    )
    hours = value["remaining_hours_p90"]
    variance = value["remaining_time_variance_hours2"]
    if (
        isinstance(hours, bool)
        or not isinstance(hours, (int, float))
        or not math.isfinite(hours)
        or hours <= 0
    ):
        raise ValueError("remaining_hours_p90")
    if (
        isinstance(variance, bool)
        or not isinstance(variance, (int, float))
        or not math.isfinite(variance)
        or variance < 0
    ):
        raise ValueError("remaining_time_variance_hours2")
    if type(value["reusable_implementation"]) is not bool:
        raise ValueError("reusable_implementation")
    risk = value["primary_risk"]
    if type(risk) is not str or not risk.strip():
        raise ValueError("primary_risk")
    artifact_score = value["direct_artifact_score"]
    full_paths = value["full_score_claim_paths"]
    if type(artifact_score) is not int or artifact_score not in range(6):
        raise ValueError("direct_artifact_score")
    if type(full_paths) is not int or not 0 <= full_paths <= len(live_claims):
        raise ValueError("full_score_claim_paths")


def expected_points(value: dict) -> float:
    total = 0.0
    for claim in value["claim_expectations"]:
        total += 2 * claim["p_verified"]
        total += 2 * claim["p_falsified"]
        total += claim["p_toy"]
    return total


def priority(value: dict) -> float:
    return (
        expected_points(value)
        * value["judged_before_deadline_probability"]
        / max(float(value["remaining_hours_p90"]), 0.25)
    )


def ranking_key(candidate: dict) -> tuple:
    envelope = candidate["score_rate"]
    return (
        -priority(envelope),
        -int(envelope["reusable_implementation"]),
        -envelope["direct_artifact_score"],
        -envelope["full_score_claim_paths"],
        envelope["remaining_time_variance_hours2"],
        candidate["estimated_api_cost_usd"],
        candidate["paper_id"],
    )
```

In `refresh.py`, retain `ASSESSMENT_KEYS` as the legacy key set, add
`SCORE_RATE_ASSESSMENT_KEYS = ASSESSMENT_KEYS | {"score_rate"}`, and validate
either shape when reading historical snapshots. Call
`score_rate.validate_envelope(record["score_rate"], live_claims)` when merging
a new assessment. In `scheduler.rank_eligible_candidates`, require
`candidate.get("score_rate")` and sort with `score_rate.ranking_key`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_score_rate.py \
  tests/test_repro_loop_refresh.py \
  tests/test_repro_loop_scheduler.py -q
```

Expected: all tests pass; historical assessment fixtures remain readable, but
only score-rate candidates are newly admitted.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  skills/icml-repro-loop/scripts/score_rate.py \
  skills/icml-repro-loop/scripts/refresh.py \
  skills/icml-repro-loop/scripts/scheduler.py \
  tests/test_repro_loop_score_rate.py \
  tests/test_repro_loop_refresh.py \
  tests/test_repro_loop_scheduler.py
git commit -m "feat: rank reproduction candidates by expected points per hour"
```

---

### Task 2: Append-Only Telemetry Event Store

**Files:**
- Create: `skills/icml-repro-loop/scripts/telemetry.py`
- Modify: `skills/icml-repro-loop/scripts/store.py:45-120`
- Create: `tests/test_repro_loop_telemetry.py`
- Modify: `tests/test_repro_loop_store.py`

**Interfaces:**
- Consumes: controller-provided session/stage identity and exact event payloads.
- Produces:
  - `StatePaths.telemetry_event(session_id: str, sequence: int, event: str) -> Path`
  - `telemetry.append_event(paths, session_id, sequence, event, payload) -> dict`
  - `telemetry.read_session(paths, session_id) -> list[dict]`
  - `telemetry.iter_sessions(paths) -> Iterator[list[dict]]`
  - `telemetry.summarize_worker_session(events) -> dict`

- [ ] **Step 1: Write failing append-only and open-session tests**

```python
def test_events_are_append_only_and_ordered(paths, telemetry):
    queued = telemetry.append_event(
        paths,
        "session-a",
        0,
        "worker-queued",
        {
            "attempt_id": "attempt-a",
            "paper_id": "paper-a",
            "observed_at": "2026-07-27T00:00:00+00:00",
        },
    )
    assert queued["sequence"] == 0
    with pytest.raises(FileExistsError):
        telemetry.append_event(
            paths,
            "session-a",
            0,
            "worker-queued",
            {
                "attempt_id": "attempt-a",
                "paper_id": "paper-a",
                "observed_at": "2026-07-27T00:00:00+00:00",
            },
        )
    assert telemetry.read_session(paths, "session-a") == [queued]
    with pytest.raises(ValueError, match="reserved"):
        telemetry.append_event(
            paths,
            "session-a",
            1,
            "worker-launched",
            {"session_id": "spoofed"},
        )


def test_missing_exit_is_open_and_has_no_guessed_duration(paths, telemetry):
    telemetry.append_event(
        paths,
        "session-a",
        0,
        "worker-launched",
        {
            "attempt_id": "attempt-a",
            "paper_id": "paper-a",
            "observed_at": "2026-07-27T00:00:00+00:00",
            "monotonic_ns": 100,
        },
    )
    summary = telemetry.summarize_worker_session(
        telemetry.read_session(paths, "session-a")
    )
    assert summary["status"] == "open"
    assert summary["elapsed_seconds"] is None
```

- [ ] **Step 2: Run telemetry tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_telemetry.py tests/test_repro_loop_store.py -q
```

Expected: collection fails because `telemetry.py` and
`StatePaths.telemetry_event` do not exist.

- [ ] **Step 3: Implement immutable event files and summaries**

Add this path resolver:

```python
# store.StatePaths
def telemetry_event(
    self, session_id: str, sequence: int, event: str
) -> Path:
    validate_id(session_id)
    validate_id(event)
    if type(sequence) is not int or sequence < 0:
        raise ValueError("sequence")
    return (
        self.root
        / "telemetry"
        / session_id
        / f"{sequence:04d}-{event}.json"
    )
```

Implement `append_event` with `os.open(path, os.O_WRONLY | os.O_CREAT |
os.O_EXCL, 0o600)`, canonical JSON, `fsync`, and no overwrite path. Create the
session directory before opening the event, reject payload keys that collide
with `version`, `session_id`, `sequence`, or `event`, and store exact top-level
fields:

```python
record = {
    **payload,
    "version": 1,
    "session_id": session_id,
    "sequence": sequence,
    "event": event,
}
```

`summarize_worker_session` must calculate elapsed time only when both a
`worker-launched` and `worker-exited` event exist and the exit monotonic counter
is at least the launch counter. Otherwise it returns `elapsed_seconds=None`.
Reject duplicate/out-of-order sequences and mixed session IDs on read instead
of silently sorting corrupt records into a plausible history.
`iter_sessions` sorts validated session IDs and yields only histories accepted
by `read_session`, giving the dashboard a deterministic complete scan.

- [ ] **Step 4: Run telemetry and store tests and verify GREEN**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_telemetry.py tests/test_repro_loop_store.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add \
  skills/icml-repro-loop/scripts/telemetry.py \
  skills/icml-repro-loop/scripts/store.py \
  tests/test_repro_loop_telemetry.py \
  tests/test_repro_loop_store.py
git commit -m "feat: add append-only reproduction telemetry"
```

---

### Task 3: Instrument The Actual Worker Process

**Files:**
- Modify: `skills/icml-repro-loop/scripts/worker_guard.py:49-59`
- Modify: `skills/icml-repro-loop/scripts/worker_guard.py:229-252`
- Modify: `skills/icml-repro-loop/scripts/state.py:109-305`
- Modify: `skills/icml-repro-loop/scripts/state.py:314-503`
- Modify: `tests/test_repro_loop_worker_guard.py`
- Modify: `tests/test_repro_loop_state.py`

**Interfaces:**
- Consumes: a validated `LaunchSpec`, one schema-v6 `StatePaths`, and the
  current fenced attempt lease.
- Produces:
  - `worker_guard.run_worker(paths, spec, *, timeout_seconds=None, ...) -> dict`
  - CLI:
    `state.py run-worker PATH --attempt-id ID --owner OWNER --fencing-token N --runtime RUNTIME --model MODEL --worktree WORKTREE --contract CONTRACT [--timeout-seconds N]`

- [ ] **Step 1: Write failing real-boundary telemetry tests**

Extend `LaunchSpec` expectations so it includes `attempt_id`, `paper_id`, and
`project_path`. Add:

```python
class CompletedProcess:
    pid = 4242
    returncode = 0

    def wait(self, timeout=None):
        assert timeout == 30
        return self.returncode


def test_run_worker_wraps_process_with_launch_and_exit_events(
    tmp_path, monkeypatch
):
    worktree = tmp_path / "paper-worktree"
    worktree.mkdir()
    contract = write_contract(worktree)
    pass_preflight("codex", worktree)
    spec = worker_guard.launch_spec(
        "codex",
        "model-a",
        worktree,
        contract,
        attempt_id="attempt-a",
        paper_id="paper-a",
        project_path="submissions/paper-a",
    )
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    clock_values = iter(
        [
            "2026-07-27T00:00:00+00:00",
            "2026-07-27T00:00:01+00:00",
            "2026-07-27T00:00:06+00:00",
        ]
    )
    monotonic_values = iter([1_000_000_000, 6_000_000_000])

    result = worker_guard.run_worker(
        paths,
        spec,
        timeout_seconds=30,
        process_factory=lambda *args, **kwargs: CompletedProcess(),
        utc_now=lambda: next(clock_values),
        monotonic_ns=lambda: next(monotonic_values),
        session_id_factory=lambda: "session-a",
        git_head=lambda _path: "a" * 40,
    )

    events = worker_guard.telemetry.read_session(paths, "session-a")
    assert [event["event"] for event in events] == [
        "worker-queued",
        "worker-launched",
        "worker-exited",
    ]
    assert events[1]["pid"] == 4242
    assert events[2]["exit_code"] == 0
    assert events[2]["outcome"] == "proposal"
    assert result["elapsed_seconds"] == 5.0
```

Add a failure test whose `wait()` returns `7`, and an interruption test whose
`wait()` raises `KeyboardInterrupt`. The interruption test must assert an
`outcome="interrupted"` exit event and re-raise `KeyboardInterrupt`. Import
`store` directly in this test module. Add a test proving a runtime-exposed
download byte count is recorded and an unavailable count remains `None`.

- [ ] **Step 2: Run worker/state tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_worker_guard.py \
  tests/test_repro_loop_state.py -q
```

Expected: tests fail because `LaunchSpec` lacks identity fields and
`run_worker`/`run-worker` do not exist.

- [ ] **Step 3: Implement the process wrapper and fenced CLI**

Extend `LaunchSpec`:

```python
@dataclass(frozen=True, slots=True)
class LaunchSpec:
    runtime: str
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    contract: Path
    mode: str
    attempt_id: str
    paper_id: str
    project_path: str
```

Implement `run_worker` so it:

1. hashes the exact contract bytes;
2. records `worker-queued`;
3. calls `subprocess.Popen(spec.argv, cwd=spec.cwd, env=spec.env)`;
4. records `worker-launched` with PID, monotonic counter, contract digest, and
   pre-launch Git SHA;
5. waits with the optional timeout;
6. records `worker-exited` with post-run Git SHA, monotonic counter, return
   code/signal, optional runtime-exposed downloaded bytes, and outcome;
7. records `timed_out` after terminating a timed-out child;
8. records `interrupted` after terminating an interrupted child and re-raises
   the interruption.

In `state.py`, reconstruct the exact attempt fence before creating the launch
spec. Require contract `attempt_id` and `paper_id` to match the attempt shard,
and require phase `implementing` or `improving`. Record `work_kind` as
`implementation` for the former and `correction` for the latter. Preserve
public-network access and whole-worktree write access from the existing worker
launcher while continuing to strip Hub/GitHub credentials and coordinator
authority.

- [ ] **Step 4: Run worker/state tests and verify GREEN**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_worker_guard.py \
  tests/test_repro_loop_state.py -q
```

Expected: all tests pass, including interrupted and failed child outcomes.

- [ ] **Step 5: Commit Task 3**

```bash
git add \
  skills/icml-repro-loop/scripts/worker_guard.py \
  skills/icml-repro-loop/scripts/state.py \
  tests/test_repro_loop_worker_guard.py \
  tests/test_repro_loop_state.py
git commit -m "feat: measure actual reproduction worker processes"
```

---

### Task 4: Record Controller Stage Durations

**Files:**
- Modify: `skills/icml-repro-loop/scripts/telemetry.py`
- Modify: `skills/icml-repro-loop/scripts/state.py:459-502`
- Modify: `tests/test_repro_loop_telemetry.py`
- Modify: `tests/test_repro_loop_controller_validation.py`
- Modify: `tests/test_repro_loop_controller_hub.py`
- Modify: `tests/test_repro_loop_official_verdict.py`

**Interfaces:**
- Consumes: one controller callable, attempt ID, and stage name.
- Produces:
  - `telemetry.run_stage(paths, attempt_id, stage, operation, *, utc_now, monotonic_ns) -> object`
  - append-only `stage-started`, `stage-finished`, and `observation` events;
  - validation, correction, deployment, submission, and verdict observations
    that remain distinct from worker process timing.

- [ ] **Step 1: Write failing stage success/failure tests**

```python
def test_run_stage_records_real_elapsed_time(paths, telemetry):
    utc_values = iter(
        ["2026-07-27T01:00:00+00:00", "2026-07-27T01:00:04+00:00"]
    )
    monotonic_values = iter([2_000_000_000, 6_000_000_000])
    result = telemetry.run_stage(
        paths,
        "attempt-a",
        "validation",
        lambda: {"attestation_id": "a" * 64},
        utc_now=lambda: next(utc_values),
        monotonic_ns=lambda: next(monotonic_values),
        session_id_factory=lambda: "stage-a",
    )
    assert result["attestation_id"] == "a" * 64
    events = telemetry.read_session(paths, "stage-a")
    assert events[-1]["outcome"] == "passed"
    assert events[-1]["elapsed_seconds"] == 4.0


def test_run_stage_records_failure_without_serializing_error_text(
    paths, telemetry
):
    def fail():
        raise RuntimeError("token-shaped-sensitive-text")

    with pytest.raises(RuntimeError, match="token-shaped"):
        telemetry.run_stage(
            paths,
            "attempt-a",
            "deployment",
            fail,
            session_id_factory=lambda: "stage-b",
        )
    event = telemetry.read_session(paths, "stage-b")[-1]
    assert event["outcome"] == "failed"
    assert event["error_type"] == "RuntimeError"
    assert "token-shaped-sensitive-text" not in json.dumps(event)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_telemetry.py \
  tests/test_repro_loop_controller_validation.py \
  tests/test_repro_loop_controller_hub.py \
  tests/test_repro_loop_official_verdict.py -q
```

Expected: telemetry tests fail because `run_stage` does not exist.

- [ ] **Step 3: Implement and integrate stage telemetry**

Implement `run_stage` with one start event and one finish event. The finish
event stores only `outcome`, `error_type`, timing, and an attestation ID when
the successful result exposes one.

Wrap these `state.py` controller calls:

```python
return telemetry.run_stage(
    paths,
    arguments.attempt_id,
    "validation",
    lambda: controller.attest_validation(
        paths,
        arguments.attempt_id,
        lease,
        manifest,
        controller.run_command,
        now,
    ),
)
```

Use the same wrapper for `deployment`. After successful submission
attestation and verdict sync, append a single `observation` event named
`submission-observed` or `verdict-observed` containing the immutable snapshot
and attestation IDs. Treat each `run-worker` session whose fenced attempt is in
`improving` as the correction stage: its queued/launched/exited records carry
`work_kind="correction"`, so correction duration comes from the actual child
process rather than a state transition timestamp.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_telemetry.py \
  tests/test_repro_loop_controller_validation.py \
  tests/test_repro_loop_controller_hub.py \
  tests/test_repro_loop_official_verdict.py -q
```

Expected: all tests pass and exception messages never enter telemetry files.

- [ ] **Step 5: Commit Task 4**

```bash
git add \
  skills/icml-repro-loop/scripts/telemetry.py \
  skills/icml-repro-loop/scripts/state.py \
  tests/test_repro_loop_telemetry.py \
  tests/test_repro_loop_controller_validation.py \
  tests/test_repro_loop_controller_hub.py \
  tests/test_repro_loop_official_verdict.py
git commit -m "feat: measure reproduction controller stages"
```

---

### Task 5: Official Score And Queue Report

**Files:**
- Create: `skills/icml-repro-loop/scripts/score_report.py`
- Modify: `skills/icml-repro-loop/scripts/state.py:123-155`
- Modify: `skills/icml-repro-loop/scripts/state.py:314-387`
- Create: `tests/test_repro_loop_score_report.py`
- Modify: `tests/test_repro_loop_state.py`

**Interfaces:**
- Consumes: one immutable live snapshot, attempt shards, telemetry events, and
  a Hugging Face username; optionally consumes one exact, source-attributed
  leaderboard observation whose username and point total match the snapshot.
- Produces:
  - `score_report.official_points(snapshot, username) -> dict`
  - `score_report.candidate_queue(snapshot) -> list[dict]`
  - `score_report.build_report(paths, snapshot, username, rank_observation=None) -> dict`
  - CLI:
    `state.py score-report PATH --snapshot-id SNAPSHOT --username wrice [--rank-observation-json PATH]`

- [ ] **Step 1: Write failing canonical-score and queue tests**

```python
def test_official_points_use_first_judged_logbook_per_user_and_paper(
    score_report,
):
    snapshot = {
        "verdicts": [
            {
                "space_id": "wrice/first",
                "paper_id": "paper-a",
                "judged_at": "2026-07-27T01:00:00+00:00",
                "claims": [
                    {"verdict": "verified"},
                    {"verdict": "toy"},
                    {"verdict": "inconclusive"},
                ],
            },
            {
                "space_id": "wrice/later",
                "paper_id": "paper-a",
                "judged_at": "2026-07-27T02:00:00+00:00",
                "claims": [{"verdict": "verified"}, {"verdict": "verified"}],
            },
        ]
    }
    result = score_report.official_points(snapshot, "wrice")
    assert result == {"points": 3, "max_points": 6, "judged_papers": 1}


def test_queue_labels_estimates_and_orders_by_priority(score_report):
    snapshot = {"candidates": [candidate("slow"), candidate("fast")]}
    snapshot["candidates"][0]["score_rate"]["remaining_hours_p90"] = 4.0
    snapshot["candidates"][1]["score_rate"]["remaining_hours_p90"] = 1.0
    queue = score_report.candidate_queue(snapshot)
    assert [row["paper_id"] for row in queue] == ["fast", "slow"]
    assert all(row["authority"] == "estimate" for row in queue)
    assert queue[0]["primary_risk"]
```

Add tests that:

- reject a rank observation when its username or points differ from the
  immutable verdict snapshot;
- count pending expected points only for `submitted` and `judging` attempts;
- list attempts awaiting validation/deployment and explicit blockers;
- compute queue time, worker time, implementation/correction session counts,
  validation/deployment time, first-launch-to-submission time, first-pass
  validation rate, and judged points per worker/end-to-end hour;
- return `None` rather than inventing rates when no matching duration exists.

- [ ] **Step 2: Run report tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_score_report.py tests/test_repro_loop_state.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named
'score_report'`.

- [ ] **Step 3: Implement report aggregation and CLI**

Use `score_rate.claim_points` for official verdict statuses. Canonicalize
logbooks by `(username, paper_id)` and the earliest valid `judged_at`, matching
the official leaderboard behavior.

Validate an optional rank observation with exact keys `observed_at`,
`source_url`, `username`, `points`, and `rank`. Require its username and points
to equal the snapshot-derived official result before including it; otherwise
fail closed. The report performs no leaderboard scraping itself.

`build_report` returns exact top-level keys:

```python
{
    "official": {
        "authority": "official-verdict-snapshot",
        "snapshot_id": snapshot["snapshot_id"],
        "username": username,
        "points": points,
        "max_points": max_points,
        "judged_papers": judged_papers,
        "rank_observation": validated_rank_observation_or_none,
    },
    "pending_judgment": {
        "authority": "estimate",
        "papers": pending_papers,
        "expected_points": pending_expected_points,
    },
    "capacity": {
        "max_runnable": index["max_runnable_attempts"],
        "runnable": len(attempts.runnable_attempt_ids(paths)),
        "idle": max_runnable - runnable,
    },
    "phases": phase_counts,
    "awaiting_validation": awaiting_validation,
    "awaiting_deployment": awaiting_deployment,
    "blockers": explicit_blockers,
    "candidate_queue": candidate_queue(snapshot),
    "telemetry": {
        "worker_queue_seconds": worker_queue_seconds,
        "worker_process_seconds": worker_process_seconds,
        "implementation_sessions": implementation_sessions,
        "correction_sessions": correction_sessions,
        "validation_seconds": validation_seconds,
        "deployment_seconds": deployment_seconds,
        "first_launch_to_submission_seconds": first_launch_to_submission_seconds,
        "first_pass_validation_rate": first_pass_validation_rate,
        "judged_points_per_worker_hour": judged_points_per_worker_hour,
        "judged_points_per_end_to_end_hour": judged_points_per_end_to_end_hour,
        "open_sessions": open_sessions,
    },
}
```

Pending points come only from assessed score-rate envelopes and remain labeled
estimates. Rate numerators include only snapshot-derived canonical judged
points whose paper can be joined to an attempt; denominators include only
complete matching telemetry intervals. The report must not write state, fetch
the network, or substitute Git/phase timestamps for missing telemetry.

- [ ] **Step 4: Run report/state tests and verify GREEN**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_score_report.py tests/test_repro_loop_state.py -q
```

Expected: all tests pass; official and estimated fields are visibly distinct.

- [ ] **Step 5: Commit Task 5**

```bash
git add \
  skills/icml-repro-loop/scripts/score_report.py \
  skills/icml-repro-loop/scripts/state.py \
  tests/test_repro_loop_score_report.py \
  tests/test_repro_loop_state.py
git commit -m "feat: report official score and points-per-hour queue"
```

---

### Task 6: Broad Unclaimed Census And Existing-Work Harvest

**Files:**
- Modify: `skills/icml-repro-loop/scripts/score_report.py`
- Modify: `skills/icml-repro-loop/scripts/state.py`
- Modify: `tests/test_repro_loop_score_report.py`
- Modify: `tests/test_repro_loop_state.py`

**Interfaces:**
- Consumes: raw immutable snapshot, durable attempts/history/rejections, and
  explicit registered worktree roots.
- Produces:
  - `score_report.candidate_census(paths, snapshot, worktree_roots, *, git_head=read_git_head) -> list[dict]`
  - CLI:
    `state.py candidate-census PATH --snapshot-id SNAPSHOT --workspace-root ROOT`

- [ ] **Step 1: Write failing census tests**

```python
def test_census_excludes_claimed_and_finds_existing_project(
    paths, tmp_path, score_report
):
    worktree = tmp_path / "paper-worktree"
    project = worktree / "submissions" / "paper-fast"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='paper-fast'\n")
    snapshot = raw_snapshot(
        candidates=[
            live_candidate("paper-fast-id", slug="paper-fast", claims=5),
            live_candidate("paper-claimed", claims=6),
            live_candidate("paper-one-claim", claims=1),
        ],
        tagged_spaces=[{"paper_id": "paper-claimed", "space_id": "org/claimed"}],
    )

    rows = score_report.candidate_census(paths, snapshot, [worktree])

    assert [row["paper_id"] for row in rows] == ["paper-fast-id"]
    assert rows[0]["claim_count"] == 5
    assert rows[0]["existing_projects"] == [
        {"path": str(project), "git_head": "a" * 40}
    ]
    assert rows[0]["authority"] == "research-required"
```

Add a test that excludes attempt history, candidate leases, queued submissions,
tagged Spaces, verdicts, and explicit rejections. Inject the Git-head reader in
tests so the census never infers or fabricates a revision.

- [ ] **Step 2: Run census tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_score_report.py tests/test_repro_loop_state.py -q
```

Expected: tests fail because `candidate_census` and the CLI command do not
exist.

- [ ] **Step 3: Implement the read-only census**

Reuse the scheduler's durable/live exclusion logic without accepting or
guessing feasibility. Include only papers with at least two live claims.
Return:

```python
{
    "paper_id": paper_id,
    "title": candidate["title"],
    "claim_count": len(candidate["live_claims"]),
    "existing_projects": sorted(
        [
            {"path": path, "git_head": exact_worktree_head}
            for path in existing_projects
        ],
        key=lambda record: record["path"],
    ),
    "authority": "research-required",
}
```

Sort by existing project first, then descending claim count, then paper ID.
The CLI discovers registered roots from `git worktree list --porcelain`,
requires every discovered root to be below `--workspace-root`, and passes the
resolved roots to `candidate_census`. A directory is reusable only when the
candidate's validated slug maps exactly to `submissions/<slug>` and `git
rev-parse HEAD` succeeds; otherwise record no project instead of an unpinned
path. The census row remains identified by the candidate's paper ID.

- [ ] **Step 4: Run census/state tests and verify GREEN**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_score_report.py tests/test_repro_loop_state.py -q
```

Expected: all tests pass; census output contains no inferred feasibility or
score.

- [ ] **Step 5: Commit Task 6**

```bash
git add \
  skills/icml-repro-loop/scripts/score_report.py \
  skills/icml-repro-loop/scripts/state.py \
  tests/test_repro_loop_score_report.py \
  tests/test_repro_loop_state.py
git commit -m "feat: inventory unclaimed papers and existing implementations"
```

---

### Task 7: Document The Cutover And Verify The Whole Loop

**Required sub-skill:** `superpowers:writing-skills`

**Files:**
- Modify: `skills/icml-repro-loop/SKILL.md`
- Modify: `skills/icml-repro-loop/references/selection-rubric.md`
- Modify: `skills/icml-repro-loop/references/submission-checklist.md`
- Modify: `docs/REMOTE_SETUP.md`
- Modify: `docs/HANDOFF.md` (controller append only; do not stage)
- Modify: `tests/test_repro_loop_state.py`
- Create: `tests/test_repro_loop_points_pipeline.py`

**Interfaces:**
- Consumes: all commands and schemas from Tasks 1–6.
- Produces: operator instructions that make score-rate assessment, instrumented
  worker launch, reporting, and census the default workflow.

- [ ] **Step 1: Write failing documentation-contract assertions**

```python
def test_operator_docs_require_score_rate_and_instrumented_launch():
    skill = (ROOT / "skills/icml-repro-loop/SKILL.md").read_text()
    remote = (ROOT / "docs/REMOTE_SETUP.md").read_text()
    rubric = (
        ROOT / "skills/icml-repro-loop/references/selection-rubric.md"
    ).read_text()

    assert "expected points per remaining hour" in skill
    assert "state.py run-worker" in skill
    assert "state.py score-report" in skill
    assert "state.py candidate-census" in rubric
    assert "Git timestamps are not worker runtime" in remote
```

Add one end-to-end fixture that starts with a raw census, attaches two
source-bound score-rate assessments, admits the higher-rate eligible paper,
records a complete implementation worker session plus validation/deployment
events, and asserts the final report keeps official points, pending estimates,
capacity, and measured durations distinct. Add explicit regression assertions
that attempts in `judging` or `blocked` do not consume runnable capacity.

- [ ] **Step 2: Run the documentation test and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_state.py::test_operator_docs_require_score_rate_and_instrumented_launch -q
```

Expected: fails on the first missing phrase.

- [ ] **Step 3: Update operator documentation**

Document this exact sequence:

```bash
raw_id=$(env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run python \
  skills/icml-repro-loop/scripts/state.py refresh-live \
  state/repro-loop.json |
  env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run python -c \
  'import json,sys; print(json.load(sys.stdin)["snapshot_id"])')

env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run python \
  skills/icml-repro-loop/scripts/state.py candidate-census \
  state/repro-loop.json \
  --snapshot-id "$raw_id" \
  --workspace-root /home/will/projects/icml-2026-reproductions

env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run python \
  skills/icml-repro-loop/scripts/state.py refresh-live \
  state/repro-loop.json \
  --assessments-json state/candidate-assessments.json

env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run python \
  skills/icml-repro-loop/scripts/state.py scheduler-pass \
  state/repro-loop.json \
  --snapshot-id ASSESSED_SNAPSHOT

env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run python \
  skills/icml-repro-loop/scripts/state.py run-worker \
  state/repro-loop.json \
  --attempt-id ATTEMPT \
  --owner OWNER \
  --fencing-token TOKEN \
  --runtime codex \
  --model MODEL \
  --worktree /ABSOLUTE/WORKTREE \
  --contract /ABSOLUTE/WORKTREE/.superpowers/worker-contract.json

env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run python \
  skills/icml-repro-loop/scripts/state.py score-report \
  state/repro-loop.json \
  --snapshot-id ASSESSED_SNAPSHOT \
  --username wrice \
  --rank-observation-json state/wrice-rank-observation.json
```

Document the exact rank-observation schema and that omitting the file reports
`rank_observation=null`. After all implementation verification passes, use
`apply_patch` to append this exact milestone to `docs/HANDOFF.md` without
altering existing lines:

```markdown
- 2026-07-27: Implemented the approved leaderboard-points operating loop:
  source-bound score-rate scheduling, actual worker/controller telemetry,
  read-only score/capacity reporting, and broad unclaimed-paper census.
```

Do not stage `docs/HANDOFF.md`; its pre-existing changes remain unowned and the
controller will reconcile the combined handoff update separately.

- [ ] **Step 4: Run focused and full verification**

Run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest \
  tests/test_repro_loop_score_rate.py \
  tests/test_repro_loop_telemetry.py \
  tests/test_repro_loop_score_report.py \
  tests/test_repro_loop_refresh.py \
  tests/test_repro_loop_scheduler.py \
  tests/test_repro_loop_worker_guard.py \
  tests/test_repro_loop_controller_validation.py \
  tests/test_repro_loop_controller_hub.py \
  tests/test_repro_loop_official_verdict.py \
  tests/test_repro_loop_state.py \
  tests/test_repro_loop_points_pipeline.py -q

env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q

env UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run \
  /home/will/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/icml-repro-loop

env UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  PRE_COMMIT_HOME=/tmp/icml-repro-pre-commit \
  uv run pre-commit run -a

git diff --check
git status --short
```

Expected: every command passes; `submissions/nape/` remains untouched; the only
remaining unrelated worktree modification is the user's prior
`docs/HANDOFF.md` edit plus the controller's appended milestone.

- [ ] **Step 5: Commit Task 7 without absorbing unrelated handoff changes**

Stage only the skill, references, setup documentation, and tests:

```bash
git add \
  skills/icml-repro-loop/SKILL.md \
  skills/icml-repro-loop/references/selection-rubric.md \
  skills/icml-repro-loop/references/submission-checklist.md \
  docs/REMOTE_SETUP.md \
  tests/test_repro_loop_state.py \
  tests/test_repro_loop_points_pipeline.py
git diff --cached --check
git commit -m "docs: operate reproduction loop by expected leaderboard points"
```

After the commit, verify `git status --short` still shows every unrelated
pre-existing change and no generated telemetry/state files. The controller,
not the implementation worker, then reconciles and commits
`docs/HANDOFF.md` before operational cutover.

---

## Operational Cutover After Implementation

The implementation branch is not score-producing until the controller performs
this separately reviewed operational sequence:

1. fetch a fresh raw live snapshot;
2. run `candidate-census` across registered worktrees;
3. send the top census rows to read-only research scouts;
4. write score-rate assessments bound to the current challenge revision;
5. fetch a fresh assessed snapshot;
6. run `scheduler-pass` to fill available capacity;
7. record/review paper-specific designs;
8. launch workers only through `run-worker`;
9. validate/deploy/submit ready papers continuously;
10. use `score-report` after every worker exit, validation outcome, and verdict.

No operational state, Hub, submission, or verdict mutation is part of this
implementation plan itself.
