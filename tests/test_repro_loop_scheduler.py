"""Tests for bounded scheduler admission and independent judgments."""

import copy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib
import json
from pathlib import Path
import sys
import threading

import pytest

from repro_loop_attestation_fixtures import add_attestation_fields


SCRIPTS = (
    Path(__file__).resolve().parents[1] / "skills" / "icml-repro-loop" / "scripts"
)
TTL = timedelta(minutes=5)
PHASE_KINDS = {
    "validated": "validation",
    "deployed": "deployment",
    "submitted": "submission",
    "judging": "authority-audit",
    "complete": "verdict",
}


def load_module(name: str):
    sys.path.insert(0, str(SCRIPTS))
    sys.modules.pop(name, None)
    return importlib.import_module(name)


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
def scheduler():
    load_module("store")
    load_module("leases")
    load_module("state")
    load_module("attempts")
    return load_module("scheduler")


@pytest.fixture
def paths(tmp_path, store):
    value = store.StatePaths(tmp_path / "repro-loop.json")
    store.atomic_json_write(value.index, store.new_index(), store.validate_index)
    return value


@pytest.fixture
def now():
    return datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)


def paper(paper_id: str, score: int = 0) -> dict:
    score_rate = {
        "claim_expectations": [
            {
                "challenge_claim_sha256": hashlib.sha256(
                    b"Challenge claim 1"
                ).hexdigest(),
                "p_verified": 0.5,
                "p_falsified": 0.25,
                "p_toy": 0.1,
            },
            {
                "challenge_claim_sha256": hashlib.sha256(
                    b"Challenge claim 2"
                ).hexdigest(),
                "p_verified": 0.0,
                "p_falsified": 0.5,
                "p_toy": 0.25,
            },
        ],
        "judged_before_deadline_probability": 0.8,
        "remaining_hours_p90": 2.0,
        "reusable_implementation": False,
        "direct_artifact_score": 4,
        "full_score_claim_paths": 2,
        "remaining_time_variance_hours2": 0.25,
        "primary_risk": "Artifact schema may have drifted.",
    }
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "slug": paper_id,
        "upstream_revision": f"revision-{paper_id}",
        "target_claims": ["claim-1", "claim-2"],
        "claim_bindings": [
            {
                "target_claim": "claim-1",
                "challenge_claim": "Challenge claim 1",
                "challenge_claim_sha256": hashlib.sha256(
                    b"Challenge claim 1"
                ).hexdigest(),
            },
            {
                "target_claim": "claim-2",
                "challenge_claim": "Challenge claim 2",
                "challenge_claim_sha256": hashlib.sha256(
                    b"Challenge claim 2"
                ).hexdigest(),
            },
        ],
        "live_claims": [
            {"text": "Challenge claim 1", "status": "extracted"},
            {"text": "Challenge claim 2", "status": "extracted"},
        ],
        "estimated_api_cost_usd": 0.0,
        "score": score,
        "score_rate": score_rate,
        "artifact_access": True,
        "cpu_only": True,
        "safety_blocker": None,
        "licensing_blocker": None,
    }


def snapshot_for_candidates(now, candidates):
    return {
        "snapshot_id": "snapshot-1",
        "fetched_at": now.isoformat(),
        "source_revision": "source-1",
        "candidates": candidates,
        "queued_submissions": [],
        "tagged_spaces": [],
        "verdicts": [],
    }


def write_snapshot(
    store,
    paths,
    now,
    candidates,
    snapshot_id="snapshot-1",
    queued_submissions=None,
    tagged_spaces=None,
    verdicts=None,
    fetched_at=None,
):
    payload = {
        "fetched_at": (fetched_at or now).isoformat(),
        "source_revision": "catalog-revision-1",
        "candidates": candidates,
        "queued_submissions": queued_submissions or [],
        "tagged_spaces": tagged_spaces or [],
        "verdicts": verdicts or [],
    }
    snapshot_id = hashlib.sha256(
        json.dumps(
            payload, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    snapshot = {"snapshot_id": snapshot_id, **payload}
    path = paths.root / "snapshots" / f"{snapshot_id}.json"
    store.atomic_json_write(path, snapshot, store.validate_snapshot)
    with store.locked_json(paths.index, store.validate_index) as index:
        index["snapshots"][snapshot_id] = str(path.relative_to(paths.index.parent))
    return snapshot_id


@pytest.fixture
def snapshot_id(store, paths, now):
    return write_snapshot(
        store,
        paths,
        now,
        [paper(f"paper-{number}", 30 - number) for number in range(22)],
    )


def transition_to_submitted(attempts, paths, assignment, now):
    attempt_id = assignment.attempt_id
    lease = assignment.writer_lease
    attempts.transition_attempt(paths, attempt_id, "design-pending", lease, now)
    attempts.record_design(paths, attempt_id, lease, "author", "design.md", now)
    attempts.record_design_review(
        paths, attempt_id, lease, "reviewer", "approved", now
    )
    for phase in ("validated", "deployed", "submitted"):
        transition_attested(attempts, paths, attempt_id, phase, lease, now)
    attempts.update_attempt(
        paths,
        attempt_id,
        lease,
        now,
        space_id=f"org/{attempt_id}",
        deployed_sha=f"sha-{attempt_id}",
    )


def transition_attested(
    attempts,
    paths,
    attempt_id,
    phase,
    lease,
    now,
    attempt_number=1,
):
    record = {
        "kind": PHASE_KINDS[phase],
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "observed_at": now.isoformat(),
        "source_commit": "abc123",
        "payload_sha256": "1" * 64,
    }
    add_attestation_fields(record)
    attestation_id = attempts.attestations.persist(
        paths,
        record,
    )
    return attempts.transition_attested(
        paths, attempt_id, phase, attestation_id, {}, lease, now
    )


def resource_lock_is_held(paths, resource: str) -> bool:
    path = paths.resource_lease(resource)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return False


def test_scheduler_admits_exactly_twenty_runnable_attempts(
    paths, snapshot_id, now, scheduler, attempts
):
    report = scheduler.scheduler_pass(paths, snapshot_id, now)

    assert len(report.created_attempt_ids) == 20
    assert len(attempts.runnable_attempt_ids(paths)) == 20
    assert report.paper_ids == tuple(
        sorted(f"paper-{number}" for number in range(22))[:20]
    )


def test_scheduler_admits_higher_expected_points_per_hour_before_legacy_score(
    scheduler, now
):
    higher_legacy_score = paper("paper-legacy", 100)
    higher_legacy_score["score_rate"]["judged_before_deadline_probability"] = 0.2
    higher_points_per_hour = paper("paper-rate", 1)
    higher_points_per_hour["score_rate"]["judged_before_deadline_probability"] = 0.9

    ranked = scheduler.rank_eligible_candidates(
        snapshot_for_candidates(now, [higher_legacy_score, higher_points_per_hour])
    )

    assert [candidate["paper_id"] for candidate in ranked] == [
        "paper-rate",
        "paper-legacy",
    ]


def test_scheduler_keeps_non_cpu_candidate_with_score_rate_ineligible(scheduler, now):
    candidate = paper("paper-a", 10)
    candidate["cpu_only"] = False

    assert scheduler.rank_eligible_candidates(snapshot_for_candidates(now, [candidate])) == []


@pytest.mark.parametrize(
    "cost",
    [None, True, -0.01, 10.01, float("nan"), float("inf"), float("-inf")],
)
def test_scheduler_rejects_missing_or_invalid_estimated_api_cost(
    scheduler, now, cost
):
    candidate = paper("paper-a", 10)
    if cost is None:
        candidate.pop("estimated_api_cost_usd")
    else:
        candidate["estimated_api_cost_usd"] = cost
    snapshot = {
        "snapshot_id": "snapshot-1",
        "fetched_at": now.isoformat(),
        "source_revision": "source-1",
        "candidates": [candidate],
        "queued_submissions": [],
        "tagged_spaces": [],
        "verdicts": [],
    }

    assert scheduler.rank_eligible_candidates(snapshot) == []


@pytest.mark.parametrize("cost", [0, 10, 0.0, 10.0])
def test_scheduler_accepts_estimated_api_cost_boundaries(scheduler, now, cost):
    candidate = paper("paper-a", 10)
    candidate["estimated_api_cost_usd"] = cost
    snapshot = {
        "snapshot_id": "snapshot-1",
        "fetched_at": now.isoformat(),
        "source_revision": "source-1",
        "candidates": [candidate],
        "queued_submissions": [],
        "tagged_spaces": [],
        "verdicts": [],
    }

    assert scheduler.rank_eligible_candidates(snapshot) == [candidate]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda candidate: candidate.pop("claim_bindings"),
        lambda candidate: candidate.update(
            {"claim_bindings": [candidate["claim_bindings"][0]]}
        ),
    ],
    ids=["missing-bindings", "target-list-differs"],
)
def test_scheduler_rejects_missing_or_mismatched_claim_bindings(
    scheduler, now, mutation
):
    candidate = paper("paper-a", 10)
    mutation(candidate)
    snapshot = {
        "snapshot_id": "snapshot-1",
        "fetched_at": now.isoformat(),
        "source_revision": "source-1",
        "candidates": [candidate],
        "queued_submissions": [],
        "tagged_spaces": [],
        "verdicts": [],
    }

    assert scheduler.rank_eligible_candidates(snapshot) == []


def test_read_fresh_snapshot_rejects_payload_not_matching_snapshot_id(
    paths, store, now, scheduler
):
    snapshot_id = write_snapshot(store, paths, now, [paper("paper-a", 10)])
    path = paths.root / "snapshots" / f"{snapshot_id}.json"
    snapshot = store.read_json(path)
    snapshot["candidates"][0]["title"] = "corrupted"
    store.atomic_json_write(path, snapshot, store.validate_snapshot)

    with pytest.raises(ValueError, match="snapshot_id"):
        scheduler.read_fresh_snapshot(paths, snapshot_id, now)


def test_scheduler_accepts_snapshot_at_five_minute_boundary(
    paths, store, now, scheduler
):
    snapshot_id = write_snapshot(
        store,
        paths,
        now,
        [paper("paper-a", 10)],
        fetched_at=now - TTL,
    )

    report = scheduler.scheduler_pass(paths, snapshot_id, now)

    assert report.paper_ids == ("paper-a",)


def test_scheduler_rejects_snapshot_one_microsecond_past_window(
    paths, store, now, scheduler
):
    snapshot_id = write_snapshot(
        store,
        paths,
        now,
        [paper("paper-a", 10)],
        fetched_at=now - TTL - timedelta(microseconds=1),
    )

    with pytest.raises(ValueError, match="fetched_at"):
        scheduler.scheduler_pass(paths, snapshot_id, now)


def test_scheduler_rejects_future_snapshot(paths, store, now, scheduler):
    snapshot_id = write_snapshot(
        store,
        paths,
        now,
        [paper("paper-a", 10)],
        fetched_at=now + timedelta(microseconds=1),
    )

    with pytest.raises(ValueError, match="fetched_at"):
        scheduler.scheduler_pass(paths, snapshot_id, now)


def test_scheduler_refills_completed_and_blocked_slots(
    paths, snapshot_id, now, scheduler, attempts
):
    first = scheduler.scheduler_pass(paths, snapshot_id, now)
    complete = first.assignments[0]
    blocked = first.assignments[1]
    for assignment in (complete, blocked):
        attempt_id = assignment.attempt_id
        lease = assignment.writer_lease
        attempts.transition_attempt(paths, attempt_id, "design-pending", lease, now)
        attempts.record_design(paths, attempt_id, lease, "author", "design.md", now)
        attempts.record_design_review(
            paths, attempt_id, lease, "reviewer", "approved", now
        )
    for phase in ("validated", "deployed", "submitted", "judging", "complete"):
        transition_attested(
            attempts, paths, complete.attempt_id, phase, complete.writer_lease, now
        )
    attempts.transition_attempt(
        paths,
        blocked.attempt_id,
        "blocked",
        blocked.writer_lease,
        now,
        blocker="artifact unavailable",
    )

    report = scheduler.scheduler_pass(paths, snapshot_id, now)

    assert len(report.created_attempt_ids) == 2
    assert len(attempts.runnable_attempt_ids(paths)) == 20


def test_duplicate_paper_can_only_be_claimed_once(
    paths, store, now, scheduler
):
    snapshot_id = write_snapshot(store, paths, now, [paper("paper-a", 10)])
    barrier = threading.Barrier(2)
    reports = []
    errors = []

    def schedule():
        try:
            barrier.wait()
            reports.append(scheduler.scheduler_pass(paths, snapshot_id, now))
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [threading.Thread(target=schedule) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert sum("paper-a" in report.paper_ids for report in reports) == 1


def test_scheduler_excludes_every_durable_or_live_claim_source(
    paths, store, leases, now, scheduler
):
    excluded = {
        "active",
        "history",
        "rejected",
        "queued",
        "tagged",
        "verdict",
        "leased",
    }
    index = store.read_json(paths.index)
    for section, paper_id, phase in (
        ("attempts", "active", "selected"),
        ("history", "history", "complete"),
    ):
        index[section][paper_id] = {
            "path": f"repro-loop/attempts/{paper_id}.json",
            "paper_id": paper_id,
            "phase": phase,
            "updated_at": now.isoformat(),
        }
    index["rejections"].append({"paper_id": "rejected"})
    store.atomic_json_write(paths.index, index, store.validate_index)
    leases.acquire_lease(
        paths, "candidate:leased", "other-scheduler", "other", now, TTL
    )
    snapshot_id = write_snapshot(
        store,
        paths,
        now,
        [paper(candidate, 10) for candidate in sorted(excluded)]
        + [paper("eligible", 1)],
        queued_submissions=[
            {"paper_id": "queued", "space_id": "wrice/queued"}
        ],
        tagged_spaces=[
            {"paper_id": "tagged", "space_id": "wrice/tagged"}
        ],
        verdicts=[
            {
                "paper_id": "verdict",
                "space_id": "wrice/verdict",
                "source_revision": "verdict-rev",
            }
        ],
    )

    report = scheduler.scheduler_pass(paths, snapshot_id, now)

    assert report.paper_ids == ("eligible",)


def test_external_contributor_records_do_not_claim_scheduler_candidate(
    paths, store, now, scheduler
):
    snapshot_id = write_snapshot(
        store,
        paths,
        now,
        [paper("paper-a", 10)],
        queued_submissions=[
            {"paper_id": "paper-a", "space_id": "other/queued"}
        ],
        tagged_spaces=[
            {"paper_id": "paper-a", "space_id": "other/tagged"}
        ],
        verdicts=[
            {
                "paper_id": "paper-a",
                "space_id": "other/verdict",
                "source_revision": "verdict-rev",
            }
        ],
    )

    report = scheduler.scheduler_pass(paths, snapshot_id, now)

    assert report.paper_ids == ("paper-a",)


def test_rejection_committed_before_insertion_wins(
    paths, store, now, scheduler, monkeypatch
):
    snapshot_id = write_snapshot(store, paths, now, [paper("paper-a", 10)])
    prechecked = threading.Event()
    continue_insertion = threading.Event()
    reports = []
    errors = []
    real_assert = scheduler.attempts._assert_attempt_fence

    def pause_before_insertion(*args):
        real_assert(*args)
        prechecked.set()
        assert continue_insertion.wait(timeout=5)

    monkeypatch.setattr(
        scheduler.attempts, "_assert_attempt_fence", pause_before_insertion
    )

    def schedule():
        try:
            reports.append(scheduler.scheduler_pass(paths, snapshot_id, now))
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    thread = threading.Thread(target=schedule)
    thread.start()
    assert prechecked.wait(timeout=5)
    with store.locked_json(paths.index, store.validate_index) as index:
        index["rejections"].append({"paper_id": "paper-a"})
    continue_insertion.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert reports[0].created_attempt_ids == ()
    assert "paper-a" not in {
        reference["paper_id"]
        for reference in store.read_json(paths.index)["attempts"].values()
    }


def test_pending_judgment_does_not_block_new_admission(
    paths, store, now, scheduler
):
    first_snapshot = write_snapshot(store, paths, now, [paper("submitted", 10)])
    submitted = scheduler.scheduler_pass(paths, first_snapshot, now).assignments[0]
    transition_to_submitted(scheduler.attempts, paths, submitted, now)
    scheduler.watch_attempt(
        paths,
        submitted.attempt_id,
        submitted.writer_lease,
        12,
        now + timedelta(hours=24),
        now,
    )
    second_snapshot = write_snapshot(
        store,
        paths,
        now,
        [paper("submitted", 10), paper("new-paper", 9)],
        snapshot_id="snapshot-2",
    )

    report = scheduler.scheduler_pass(paths, second_snapshot, now)

    assert report.created_attempt_ids
    assert report.paper_ids == ("new-paper",)


def test_judgment_retains_submission_identity_and_bounded_poll_provenance(
    paths, store, now, scheduler
):
    snapshot_id = write_snapshot(store, paths, now, [paper("paper-a", 10)])
    assignment = scheduler.scheduler_pass(paths, snapshot_id, now).assignments[0]
    transition_to_submitted(scheduler.attempts, paths, assignment, now)
    deadline = now + timedelta(hours=24)

    watched = scheduler.watch_attempt(
        paths, assignment.attempt_id, assignment.writer_lease, 2, deadline, now
    )
    scheduler.record_poll(
        paths, assignment.attempt_id, assignment.writer_lease, "pending", now
    )
    scheduler.record_poll(
        paths,
        assignment.attempt_id,
        assignment.writer_lease,
        "ready",
        now + timedelta(minutes=2),
    )
    recorded = store.read_json(paths.judgment(assignment.attempt_id))

    assert watched["space_id"] == f"org/{assignment.attempt_id}"
    assert watched["submitted_sha"] == f"sha-{assignment.attempt_id}"
    assert watched["attempt_number"] == 1
    assert watched["target_claims"] == ["claim-1", "claim-2"]
    assert watched["poll_limit"] == 2
    assert watched["poll_deadline"] == deadline.isoformat()
    assert len(recorded["polls"]) == 2
    assert recorded["raw_verdict"] is None
    assert recorded["normalized_verdict"] is None
    assert recorded["source_revision"] is None
    assert recorded["verdict_at"] is None
    assert store.read_json(paths.judgment(assignment.attempt_id)) == recorded


def test_second_judgment_archives_finalized_first_round(
    paths, store, now, scheduler
):
    snapshot_id = write_snapshot(store, paths, now, [paper("paper-a", 10)])
    assignment = scheduler.scheduler_pass(paths, snapshot_id, now).assignments[0]
    transition_to_submitted(scheduler.attempts, paths, assignment, now)
    first = scheduler.watch_attempt(
        paths, assignment.attempt_id, assignment.writer_lease, 2, now + TTL, now
    )
    finalized = copy.deepcopy(first)
    finalized.update(
        {
            "raw_verdict": {"result": "official"},
            "normalized_verdict": normalized_verdict(),
            "source_revision": "verdict-revision-1",
            "verdict_at": (now + timedelta(minutes=1)).isoformat(),
            "updated_at": (now + timedelta(minutes=1)).isoformat(),
        }
    )
    store.atomic_json_write(
        paths.judgment(assignment.attempt_id),
        finalized,
        scheduler.validate_judgment_record,
    )
    scheduler.attempts.transition_attempt(
        paths,
        assignment.attempt_id,
        "improving",
        assignment.writer_lease,
        now + timedelta(minutes=2),
        improvement_attempts=1,
        improvement_reason="official verdict requested stronger evidence",
    )
    scheduler.attempts.update_attempt(
        paths,
        assignment.attempt_id,
        assignment.writer_lease,
        now + timedelta(minutes=2),
        deployed_sha=f"improved-{assignment.attempt_id}",
    )
    for phase in ("validated", "deployed", "submitted"):
        transition_attested(
            scheduler.attempts,
            paths,
            assignment.attempt_id,
            phase,
            assignment.writer_lease,
            now + timedelta(minutes=2),
            attempt_number=2,
        )

    second = scheduler.watch_attempt(
        paths,
        assignment.attempt_id,
        assignment.writer_lease,
        3,
        now + TTL * 2,
        now + timedelta(minutes=2),
    )

    assert first["attempt_number"] == 1
    assert second["attempt_number"] == 2
    assert second["submitted_sha"] == f"improved-{assignment.attempt_id}"
    assert (
        store.read_json(paths.judgment_archive(assignment.attempt_id, 1))
        == finalized
    )
    assert store.read_json(paths.judgment(assignment.attempt_id)) == second


def test_judgment_poll_limit_and_deadline_are_enforced(
    paths, store, now, scheduler
):
    snapshot_id = write_snapshot(store, paths, now, [paper("paper-a", 10)])
    assignment = scheduler.scheduler_pass(paths, snapshot_id, now).assignments[0]
    transition_to_submitted(scheduler.attempts, paths, assignment, now)
    deadline = now + timedelta(minutes=1)
    scheduler.watch_attempt(
        paths, assignment.attempt_id, assignment.writer_lease, 1, deadline, now
    )
    scheduler.record_poll(
        paths, assignment.attempt_id, assignment.writer_lease, "pending", now
    )

    with pytest.raises(ValueError, match="poll_limit"):
        scheduler.record_poll(
            paths,
            assignment.attempt_id,
            assignment.writer_lease,
            "pending",
            now + timedelta(minutes=1),
        )

    other_snapshot = write_snapshot(
        store,
        paths,
        now,
        [paper("paper-b", 10)],
        snapshot_id="snapshot-2",
    )
    other = scheduler.scheduler_pass(paths, other_snapshot, now).assignments[0]
    transition_to_submitted(scheduler.attempts, paths, other, now)
    scheduler.watch_attempt(
        paths, other.attempt_id, other.writer_lease, 1, deadline, now
    )
    with pytest.raises(ValueError, match="poll_deadline"):
        scheduler.record_poll(
            paths,
            other.attempt_id,
            other.writer_lease,
            "pending",
            deadline + timedelta(seconds=1),
        )


@pytest.mark.parametrize(
    ("with_poll", "finalized", "updated_minute"),
    [(False, False, 1), (True, False, 1), (True, True, 3), (False, True, 2)],
)
def test_judgment_validator_requires_updated_at_consistent_with_events(
    paths, store, now, scheduler, with_poll, finalized, updated_minute
):
    snapshot_id = write_snapshot(store, paths, now, [paper("paper-a", 10)])
    assignment = scheduler.scheduler_pass(paths, snapshot_id, now).assignments[0]
    transition_to_submitted(scheduler.attempts, paths, assignment, now)
    judgment = scheduler.watch_attempt(
        paths, assignment.attempt_id, assignment.writer_lease, 2, now + TTL, now
    )
    if with_poll:
        judgment["polls"].append(
            {"at": (now + timedelta(minutes=2)).isoformat(), "status": "pending"}
        )
    judgment["updated_at"] = (
        now + timedelta(minutes=updated_minute)
    ).isoformat()
    if finalized:
        judgment["raw_verdict"] = {"result": "complete"}
        judgment["normalized_verdict"] = normalized_verdict()
        judgment["source_revision"] = "verdict-revision"
        judgment["verdict_at"] = (now + timedelta(minutes=1)).isoformat()

    with pytest.raises(ValueError, match="verdict_at|updated_at"):
        scheduler.validate_judgment_record(judgment)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("space_id", "org/shared-space"),
        ("deployed_sha", "shared-submitted-sha"),
    ],
)
def test_judgments_reject_duplicate_submission_identity(
    paths, store, now, scheduler, field, value
):
    snapshot_id = write_snapshot(
        store,
        paths,
        now,
        [paper("paper-a", 10), paper("paper-b", 9)],
    )
    first, second = scheduler.scheduler_pass(paths, snapshot_id, now).assignments
    for assignment in (first, second):
        transition_to_submitted(scheduler.attempts, paths, assignment, now)
        scheduler.attempts.update_attempt(
            paths,
            assignment.attempt_id,
            assignment.writer_lease,
            now,
            **{field: value},
        )
    scheduler.watch_attempt(
        paths, first.attempt_id, first.writer_lease, 1, now + TTL, now
    )

    with pytest.raises(ValueError, match="space_id|submitted_sha"):
        scheduler.watch_attempt(
            paths, second.attempt_id, second.writer_lease, 1, now + TTL, now
        )


def test_identity_takeover_cannot_overtake_judgment_creation(
    paths, store, leases, now, scheduler, monkeypatch
):
    snapshot_id = write_snapshot(store, paths, now, [paper("paper-a", 10)])
    assignment = scheduler.scheduler_pass(paths, snapshot_id, now).assignments[0]
    transition_to_submitted(scheduler.attempts, paths, assignment, now)
    space_id = f"org/{assignment.attempt_id}"
    submitted_sha = f"sha-{assignment.attempt_id}"
    resources = sorted(
        [f"judgment-space:{space_id}", f"submitted-sha:{submitted_sha}"]
    )
    reached_scan = threading.Event()
    continue_scan = threading.Event()
    successor_acquired = threading.Event()
    events = []
    errors = []
    real_assert_unique = scheduler._assert_unique_submission
    real_write = scheduler.store._transaction_write

    def pause_before_scan(*args):
        reached_scan.set()
        assert continue_scan.wait(timeout=5)
        return real_assert_unique(*args)

    def track_write(path, value, validator):
        real_write(path, value, validator)
        if path == paths.judgment(assignment.attempt_id):
            events.append("judgment-written")

    monkeypatch.setattr(scheduler, "_assert_unique_submission", pause_before_scan)
    monkeypatch.setattr(scheduler.store, "_transaction_write", track_write)

    def watch():
        try:
            scheduler.watch_attempt(
                paths,
                assignment.attempt_id,
                assignment.writer_lease,
                1,
                now + TTL * 2,
                now,
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    def take_over_identities():
        try:
            for resource in resources:
                leases.acquire_lease(
                    paths,
                    resource,
                    "identity-successor",
                    assignment.attempt_id,
                    now + TTL,
                    TTL,
                )
            events.append("successor-acquired")
            successor_acquired.set()
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    watcher = threading.Thread(target=watch)
    watcher.start()
    assert reached_scan.wait(timeout=5)
    identity_fences_held = all(
        resource_lock_is_held(paths, resource) for resource in resources
    )
    successor = threading.Thread(target=take_over_identities)
    successor.start()
    if not identity_fences_held:
        assert successor_acquired.wait(timeout=5)
    continue_scan.set()
    watcher.join(timeout=5)
    successor.join(timeout=5)

    assert not watcher.is_alive()
    assert not successor.is_alive()
    assert errors == []
    assert events == ["judgment-written", "successor-acquired"]
    judgment_path = paths.judgment(assignment.attempt_id)
    if events.index("successor-acquired") < events.index("judgment-written"):
        assert not judgment_path.exists()
    judgments = []
    for path in (paths.root / "judgments").glob("*.json"):
        judgment = store.read_json(path)
        if (
            judgment["space_id"] == space_id
            or judgment["submitted_sha"] == submitted_sha
        ):
            judgments.append(judgment)
    assert len(judgments) == 1


def normalized_verdict() -> dict:
    return {
        "claims": [
            {"claim": "claim-1", "status": "verified"},
            {"claim": "claim-2", "status": "inconclusive"},
        ]
    }


def mutate_judgment(scheduler, paths, assignment, lease, operation, now):
    if operation == "watch":
        return scheduler.watch_attempt(
            paths, assignment.attempt_id, lease, 2, now + TTL, now
        )
    if operation == "poll":
        return scheduler.record_poll(
            paths, assignment.attempt_id, lease, "pending", now
        )
    raise AssertionError(operation)


@pytest.mark.parametrize("operation", ["watch", "poll"])
def test_judgment_mutations_reject_stale_writer_after_successor_takeover(
    paths, store, leases, now, scheduler, operation
):
    snapshot_id = write_snapshot(store, paths, now, [paper("paper-a", 10)])
    assignment = scheduler.scheduler_pass(paths, snapshot_id, now).assignments[0]
    transition_to_submitted(scheduler.attempts, paths, assignment, now)
    if operation != "watch":
        scheduler.watch_attempt(
            paths,
            assignment.attempt_id,
            assignment.writer_lease,
            2,
            now + TTL * 2,
            now,
        )
    takeover_at = now + TTL
    leases.acquire_lease(
        paths,
        f"attempt:{assignment.attempt_id}",
        "successor",
        assignment.attempt_id,
        takeover_at,
        TTL,
    )

    with pytest.raises(scheduler.leases.StaleFence):
        mutate_judgment(
            scheduler,
            paths,
            assignment,
            assignment.writer_lease,
            operation,
            takeover_at,
        )


@pytest.mark.parametrize("operation", ["watch", "poll"])
def test_judgment_mutations_reject_other_attempt_writer(
    paths, store, now, scheduler, operation
):
    snapshot_id = write_snapshot(
        store,
        paths,
        now,
        [paper("paper-a", 10), paper("paper-b", 9)],
    )
    assignment, unauthorized = scheduler.scheduler_pass(
        paths, snapshot_id, now
    ).assignments
    transition_to_submitted(scheduler.attempts, paths, assignment, now)
    if operation != "watch":
        scheduler.watch_attempt(
            paths,
            assignment.attempt_id,
            assignment.writer_lease,
            2,
            now + TTL,
            now,
        )

    with pytest.raises(scheduler.leases.StaleFence):
        mutate_judgment(
            scheduler,
            paths,
            assignment,
            unauthorized.writer_lease,
            operation,
            now,
        )
