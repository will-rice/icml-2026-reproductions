"""Tests for persistent paper-owner release events."""

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sys
import threading

import pytest


SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "icml-repro-loop"
    / "scripts"
)


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def store():
    return load_module("store")


@pytest.fixture
def leases():
    load_module("store")
    return load_module("leases")


@pytest.fixture
def attempts():
    load_module("store")
    load_module("leases")
    load_module("state")
    return load_module("attempts")


@pytest.fixture
def telemetry():
    load_module("store")
    return load_module("telemetry")


@pytest.fixture
def paper_owner():
    load_module("store")
    load_module("leases")
    load_module("state")
    load_module("attempts")
    load_module("telemetry")
    return load_module("paper_owner")


@pytest.fixture
def paths(tmp_path, store):
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    store.atomic_json_write(paths.index, store.new_index(), store.validate_index)
    return paths


@pytest.fixture
def now():
    return datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)


def persist_attempt(store, paths, attempt_id, phase, now, **updates):
    attempt = {
        "attempt_id": attempt_id,
        "paper_id": f"paper-{attempt_id}",
        "phase": phase,
        "snapshot_id": "snapshot-1",
        "updated_at": now.isoformat(),
        **updates,
    }
    store.atomic_json_write(paths.attempt(attempt_id), attempt, store.validate_attempt)
    section = "history" if phase == "complete" else "attempts"
    with store.locked_json(paths.index, store.validate_index) as index:
        index[section][attempt_id] = {
            "path": f"repro-loop/attempts/{attempt_id}.json",
            "paper_id": attempt["paper_id"],
            "phase": phase,
            "updated_at": attempt["updated_at"],
        }


def acquire_attempt_lease(leases, paths, attempt_id, now):
    return leases.acquire_lease(
        paths,
        f"attempt:{attempt_id}",
        "persistent-owner",
        attempt_id,
        now,
        timedelta(hours=1),
    )


@pytest.fixture
def blocked_attempt(paths, store, leases, now):
    persist_attempt(
        store,
        paths,
        "blocked-a",
        "blocked",
        now,
        blocker="external outage",
        next_action="retry after recovery",
    )
    return "blocked-a", acquire_attempt_lease(leases, paths, "blocked-a", now)


@pytest.fixture
def complete_attempt(paths, store, leases, now):
    persist_attempt(
        store,
        paths,
        "complete-a",
        "complete",
        now,
        verdict={"claims": [{"claim_id": "claim-1", "verdict": "verified"}]},
        verdict_source_revision="verdict-revision",
        verdict_at=now.isoformat(),
    )
    return "complete-a", acquire_attempt_lease(leases, paths, "complete-a", now)


@pytest.fixture
def selected_attempt(paths, store, leases, now):
    persist_attempt(store, paths, "selected-a", "selected", now)
    return "selected-a", acquire_attempt_lease(leases, paths, "selected-a", now)


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
    assert result["verdict"] == {
        "claims": [{"claim_id": "claim-1", "verdict": "verified"}]
    }


def test_release_rejects_wrong_phase_or_outcome(
    paths, selected_attempt, now, paper_owner
):
    attempt_id, lease = selected_attempt

    with pytest.raises(ValueError, match="phase"):
        paper_owner.release_paper(paths, attempt_id, lease, "blocked", now)
    with pytest.raises(ValueError, match="outcome"):
        paper_owner.release_paper(paths, attempt_id, lease, "failed", now)


def test_release_rejects_blocked_attempt_without_next_action_before_release(
    paths, store, leases, now, paper_owner
):
    persist_attempt(store, paths, "missing-next", "blocked", now, blocker="outage")
    lease = acquire_attempt_lease(leases, paths, "missing-next", now)

    with pytest.raises(ValueError, match="next_action"):
        paper_owner.release_paper(paths, "missing-next", lease, "blocked", now)

    assert leases.assert_fence(paths, lease, now) == lease


def test_release_rejects_backdated_time_before_event_or_release(
    paths, blocked_attempt, now, paper_owner, telemetry, leases
):
    attempt_id, lease = blocked_attempt

    with pytest.raises(ValueError, match="now"):
        paper_owner.release_paper(
            paths,
            attempt_id,
            lease,
            "blocked",
            now - timedelta(microseconds=1),
            session_id_factory=lambda: "backdated-release",
        )

    assert telemetry.read_session(paths, "backdated-release") == []
    assert leases.assert_fence(paths, lease, now) == lease


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


def test_worker_failure_event_is_durable_before_competing_release(
    paths, blocked_attempt, now, paper_owner, monkeypatch
):
    attempt_id, lease = blocked_attempt
    original_append_event = paper_owner.telemetry.append_event
    release_done = threading.Event()
    release_errors = []
    release_thread = None

    def release():
        try:
            paper_owner.release_paper(
                paths,
                attempt_id,
                lease,
                "blocked",
                now,
                session_id_factory=lambda: "release-after-failure",
            )
        except BaseException as error:  # pragma: no cover - asserted below
            release_errors.append(error)
        finally:
            release_done.set()

    def append_event(*args, **kwargs):
        nonlocal release_thread
        if args[3] == "paper-owner-failed":
            release_thread = threading.Thread(target=release)
            release_thread.start()
            assert not release_done.wait(timeout=0.2)
        return original_append_event(*args, **kwargs)

    monkeypatch.setattr(paper_owner.telemetry, "append_event", append_event)

    failed = paper_owner.record_worker_failure(
        paths,
        attempt_id,
        lease,
        "RuntimeError",
        now,
        session_id_factory=lambda: "failure-before-release",
    )

    assert failed["event"] == "paper-owner-failed"
    assert release_thread is not None
    release_thread.join(timeout=5)
    assert release_done.is_set()
    assert release_errors == []


def test_stale_owner_cannot_emit_or_release(
    paths, blocked_attempt, now, paper_owner, leases, telemetry
):
    attempt_id, lease = blocked_attempt
    replacement = leases.acquire_lease(
        paths,
        f"attempt:{attempt_id}",
        "successor",
        attempt_id,
        now + timedelta(hours=1),
        timedelta(hours=1),
    )

    with pytest.raises(paper_owner.leases.StaleFence):
        paper_owner.release_paper(
            paths,
            attempt_id,
            lease,
            "blocked",
            now + timedelta(hours=1),
            session_id_factory=lambda: "stale-release",
        )
    with pytest.raises(paper_owner.leases.StaleFence):
        paper_owner.record_worker_failure(
            paths,
            attempt_id,
            lease,
            "RuntimeError",
            now + timedelta(hours=1),
            session_id_factory=lambda: "stale-failure",
        )

    assert telemetry.read_session(paths, "stale-release") == []
    assert telemetry.read_session(paths, "stale-failure") == []
    assert leases.assert_fence(paths, replacement, now + timedelta(hours=1)) == replacement


def test_concurrent_release_emits_event_only_for_winning_fence(
    paths, blocked_attempt, now, paper_owner, telemetry, monkeypatch
):
    attempt_id, lease = blocked_attempt
    barrier = threading.Barrier(2)
    original_assert_fence = paper_owner.leases.assert_fence
    results = []
    errors = []

    def synchronized_assert_fence(*args, **kwargs):
        value = original_assert_fence(*args, **kwargs)
        barrier.wait(timeout=5)
        return value

    def release(session_id):
        try:
            results.append(
                paper_owner.release_paper(
                    paths,
                    attempt_id,
                    lease,
                    "blocked",
                    now,
                    session_id_factory=lambda: session_id,
                )
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    monkeypatch.setattr(
        paper_owner.leases, "assert_fence", synchronized_assert_fence
    )
    workers = [
        threading.Thread(target=release, args=(f"race-release-{number}",))
        for number in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    events = [
        *telemetry.read_session(paths, "race-release-0"),
        *telemetry.read_session(paths, "race-release-1"),
    ]
    assert len(results) == 1
    assert len(errors) == 1
    assert len(events) == 1
    assert events[0]["event"] == "paper-owner-released"
