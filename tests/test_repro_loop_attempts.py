"""Tests for independent fenced reproduction-attempt lifecycles."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import importlib
from pathlib import Path
import sys
import threading

import pytest

from repro_loop_attestation_fixtures import add_attestation_fields


SCRIPTS = (
    Path(__file__).resolve().parents[1] / "skills" / "icml-repro-loop" / "scripts"
)
TTL = timedelta(minutes=5)
ATTESTED_PHASES = ("validated", "deployed", "submitted", "judging", "complete")
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
def attestations():
    load_module("store")
    return load_module("attestations")


@pytest.fixture
def paths(tmp_path, store):
    value = store.StatePaths(tmp_path / "repro-loop.json")
    store.atomic_json_write(value.index, store.new_index(), store.validate_index)
    return value


@pytest.fixture
def now():
    return datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)


def paper(paper_id: str) -> dict:
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "slug": f"paper-{paper_id}",
        "upstream_revision": "abc123",
        "target_claims": ["claim-1", "claim-2"],
        "estimated_api_cost_usd": 0.0,
    }


def create(attempts, leases, paths, attempt_id, paper_id, owner, now):
    lease = leases.acquire_lease(
        paths, f"attempt:{attempt_id}", owner, attempt_id, now, TTL
    )
    attempts.create_attempt(
        paths, attempt_id, paper(paper_id), lease, "snapshot-1", now
    )
    return lease


def transition_attested_to(
    attempts, attestations, paths, attempt_id, phase, lease, now
):
    for target in ATTESTED_PHASES:
        if target == "complete" and phase != "complete":
            break
        attestation_id = attestations.persist(
            paths, attestation_record(PHASE_KINDS[target], attempt_id)
        )
        attempts.transition_attested(
            paths, attempt_id, target, attestation_id, {}, lease, now
        )
        if target == phase:
            return
    raise AssertionError(f"Unsupported attested phase: {phase}")


def attestation_record(kind, attempt_id="a1", attempt_number=1, **updates):
    record = {
        "kind": kind,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "observed_at": "2026-07-24T12:00:00+00:00",
        "source_commit": "abc123",
        "payload_sha256": "1" * 64,
    }
    add_attestation_fields(record)
    record.update(updates)
    return record


def transition_through_correction(
    attempts, attestations, paths, attempt_id, lease, now
):
    first_id = attestations.persist(
        paths, attestation_record("validation", attempt_id)
    )
    attempts.transition_attested(
        paths, attempt_id, "validated", first_id, {}, lease, now
    )
    attempts.transition_attempt(
        paths,
        attempt_id,
        "improving",
        lease,
        now,
        improvement_attempts=1,
        improvement_reason="Correct validation evidence",
    )
    second_id = attestations.persist(
        paths,
        attestation_record(
            "validation",
            attempt_id,
            attempt_number=2,
            source_commit="7" * 40,
        ),
    )
    return attempts.transition_attested(
        paths, attempt_id, "validated", second_id, {}, lease, now
    )


def interrupt_transaction(monkeypatch, attempts, fail_after):
    writes = 0

    def interrupted(path, value, validator):
        nonlocal writes
        writes += 1
        if writes == fail_after:
            raise OSError("simulated interruption")
        validator(value)
        attempts.store._atomic_json_write(path, value)

    monkeypatch.setattr(
        attempts.store, "_transaction_write", interrupted, raising=False
    )


def restore_transaction_writes(monkeypatch, attempts):
    def write(path, value, validator):
        validator(value)
        attempts.store._atomic_json_write(path, value)

    monkeypatch.setattr(attempts.store, "_transaction_write", write, raising=False)


@pytest.fixture
def attempts_and_leases(attempts, leases, paths, now):
    first = create(attempts, leases, paths, "a1", "p1", "worker-1", now)
    second = create(attempts, leases, paths, "a2", "p2", "worker-2", now)
    for attempt_id, lease in (("a1", first), ("a2", second)):
        attempts.transition_attempt(paths, attempt_id, "design-pending", lease, now)
        attempts.record_design(
            paths, attempt_id, lease, f"author-{attempt_id}", "design.md", now
        )
        attempts.record_design_review(
            paths, attempt_id, lease, f"reviewer-{attempt_id}", "approved", now
        )
    return "a1", first, "a2", second


@dataclass(frozen=True)
class PendingAttempt:
    id: str
    lease: object


@pytest.fixture
def pending_attempt(attempts, leases, paths, now):
    lease = create(attempts, leases, paths, "a1", "p1", "worker-1", now)
    attempts.transition_attempt(paths, "a1", "design-pending", lease, now)
    return PendingAttempt("a1", lease)


def test_blocking_one_attempt_does_not_change_another(
    paths, attempts_and_leases, attempts, now
):
    a1, l1, a2, _ = attempts_and_leases
    attempts.transition_attempt(
        paths, a1, "blocked", l1, now, blocker="missing dataset"
    )
    assert attempts.read_attempt(paths, a1)["phase"] == "blocked"
    assert attempts.read_attempt(paths, a2)["phase"] == "implementing"


def test_different_agent_may_approve_design(paths, pending_attempt, attempts, now):
    attempts.record_design(
        paths,
        pending_attempt.id,
        pending_attempt.lease,
        "author-a",
        "design.md",
        now,
    )
    attempts.record_design_review(
        paths,
        pending_attempt.id,
        pending_attempt.lease,
        "reviewer-b",
        "approved",
        now,
    )
    assert attempts.read_attempt(paths, pending_attempt.id)["phase"] == "implementing"


def test_design_author_cannot_self_approve(paths, pending_attempt, attempts, now):
    attempts.record_design(
        paths,
        pending_attempt.id,
        pending_attempt.lease,
        "agent-a",
        "design.md",
        now,
    )
    with pytest.raises(ValueError, match="reviewer"):
        attempts.record_design_review(
            paths,
            pending_attempt.id,
            pending_attempt.lease,
            "agent-a",
            "approved",
            now,
        )


def test_transition_appends_fenced_attempt_history(
    paths, pending_attempt, attempts, now
):
    attempts.record_design(
        paths, pending_attempt.id, pending_attempt.lease, "author-a", "design.md", now
    )
    attempts.record_design_review(
        paths,
        pending_attempt.id,
        pending_attempt.lease,
        "reviewer-b",
        "approved",
        now,
    )

    assert attempts.read_attempt(paths, pending_attempt.id)["transitions"][-1] == {
        "from": "design-pending",
        "to": "implementing",
        "at": now.isoformat(),
        "owner": "worker-1",
        "fencing_token": pending_attempt.lease.fencing_token,
        "snapshot_id": "snapshot-1",
    }


def test_blocked_attempt_is_not_runnable_and_resumes_its_origin(
    paths, attempts_and_leases, attempts, now
):
    a1, lease, a2, _ = attempts_and_leases
    attempts.transition_attempt(paths, a1, "blocked", lease, now, blocker="offline")
    assert attempts.runnable_attempt_ids(paths) == [a2]

    attempts.transition_attempt(paths, a1, "implementing", lease, now)
    assert attempts.runnable_attempt_ids(paths) == [a1, a2]
    resumed = attempts.read_attempt(paths, a1)
    assert "blocker" not in resumed
    assert "blocked_from" not in resumed


@pytest.mark.parametrize("phase", ATTESTED_PHASES)
def test_generic_transition_rejects_attested_phase(
    paths, attempts_and_leases, attempts, attestations, now, phase
):
    attempt_id, lease, _, _ = attempts_and_leases
    if phase != "validated":
        previous = ATTESTED_PHASES[ATTESTED_PHASES.index(phase) - 1]
        transition_attested_to(
            attempts, attestations, paths, attempt_id, previous, lease, now
        )

    with pytest.raises(ValueError, match="attestation"):
        attempts.transition_attempt(paths, attempt_id, phase, lease, now)


@pytest.mark.parametrize("phase", ATTESTED_PHASES)
def test_generic_transition_cannot_resume_blocked_attested_phase(
    paths, attempts_and_leases, attempts, store, now, phase
):
    attempt_id, lease, _, _ = attempts_and_leases
    attempts.transition_attempt(
        paths, attempt_id, "blocked", lease, now, blocker="external outage"
    )
    with store.locked_json(paths.attempt(attempt_id), store.validate_attempt) as attempt:
        attempt["blocked_from"] = phase

    with pytest.raises(ValueError, match="attestation"):
        attempts.transition_attempt(paths, attempt_id, phase, lease, now)


def test_attested_transition_requires_matching_kind_attempt_and_number(
    paths, attempts_and_leases, attempts, attestations, now
):
    attempt_id, lease, _, _ = attempts_and_leases
    mismatches = (
        attestation_record("deployment"),
        attestation_record("validation", attempt_id="a2"),
        attestation_record("validation", attempt_number=2),
    )

    for record in mismatches:
        attestation_id = attestations.persist(paths, record)
        with pytest.raises(ValueError, match="attestation"):
            attempts.transition_attested(
                paths,
                attempt_id,
                "validated",
                attestation_id,
                {},
                lease,
                now,
            )


def test_attested_transition_advances_and_resumes_external_phase(
    paths, attempts_and_leases, attempts, attestations, now
):
    attempt_id, lease, _, _ = attempts_and_leases
    attestation_id = attestations.persist(
        paths, attestation_record("validation")
    )

    transitioned = attempts.transition_attested(
        paths,
        attempt_id,
        "validated",
        attestation_id,
        {"validation_command": "uv run pytest"},
        lease,
        now,
    )
    attempts.transition_attempt(
        paths, attempt_id, "blocked", lease, now, blocker="external outage"
    )
    resumed = attempts.transition_attested(
        paths,
        attempt_id,
        "validated",
        attestation_id,
        {},
        lease,
        now,
    )

    assert transitioned["phase"] == "validated"
    assert transitioned["validation_command"] == "uv run pytest"
    assert resumed["phase"] == "validated"
    assert "blocked_from" not in resumed


def test_attested_transition_records_attestation_id_in_provenance(
    paths, attempts_and_leases, attempts, attestations, now
):
    attempt_id, lease, _, _ = attempts_and_leases
    attestation_id = attestations.persist(
        paths, attestation_record("validation", attempt_id)
    )

    transitioned = attempts.transition_attested(
        paths, attempt_id, "validated", attestation_id, {}, lease, now
    )

    assert transitioned["transitions"][-1]["attestation_id"] == attestation_id


def test_validated_attempt_allows_one_predeployment_correction(
    paths, attempts_and_leases, attempts, attestations, now
):
    attempt_id, lease, _, _ = attempts_and_leases
    first_id = attestations.persist(
        paths, attestation_record("validation", attempt_id)
    )
    attempts.transition_attested(
        paths, attempt_id, "validated", first_id, {}, lease, now
    )
    original = paths.attestation("validation", attempt_id, 1).read_bytes()

    improving = attempts.transition_attempt(
        paths,
        attempt_id,
        "improving",
        lease,
        now,
        improvement_attempts=1,
        improvement_reason="Correct the validation evidence bundle",
    )

    assert improving["phase"] == "improving"
    assert improving["improvement_attempts"] == 1
    assert (
        improving["improvement_reason"]
        == "Correct the validation evidence bundle"
    )
    assert paths.attestation("validation", attempt_id, 1).read_bytes() == original


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"improvement_reason": "Correct evidence"}, "improvement_attempts"),
        (
            {"improvement_attempts": 0, "improvement_reason": "Correct evidence"},
            "improvement_attempts",
        ),
        (
            {"improvement_attempts": 2, "improvement_reason": "Correct evidence"},
            "improvement_attempts",
        ),
        (
            {
                "improvement_attempts": True,
                "improvement_reason": "Correct evidence",
            },
            "improvement_attempts",
        ),
        (
            {
                "improvement_attempts": "1",
                "improvement_reason": "Correct evidence",
            },
            "improvement_attempts",
        ),
        ({"improvement_attempts": 1}, "improvement_reason"),
        (
            {"improvement_attempts": 1, "improvement_reason": ""},
            "improvement_reason",
        ),
        (
            {"improvement_attempts": 1, "improvement_reason": "   "},
            "improvement_reason",
        ),
        (
            {"improvement_attempts": 1, "improvement_reason": 7},
            "improvement_reason",
        ),
    ],
)
def test_predeployment_correction_requires_exact_count_and_nonempty_reason(
    paths,
    attempts_and_leases,
    attempts,
    attestations,
    now,
    updates,
    error,
):
    attempt_id, lease, _, _ = attempts_and_leases
    first_id = attestations.persist(
        paths, attestation_record("validation", attempt_id)
    )
    attempts.transition_attested(
        paths, attempt_id, "validated", first_id, {}, lease, now
    )

    with pytest.raises(ValueError, match=error):
        attempts.transition_attempt(
            paths, attempt_id, "improving", lease, now, **updates
        )

    assert attempts.read_attempt(paths, attempt_id)["phase"] == "validated"


def test_predeployment_correction_allows_next_numbered_improvement(
    paths, attempts_and_leases, attempts, attestations, now
):
    attempt_id, lease, _, _ = attempts_and_leases
    first_id = attestations.persist(
        paths, attestation_record("validation", attempt_id)
    )
    attempts.transition_attested(
        paths, attempt_id, "validated", first_id, {}, lease, now
    )
    attempts.transition_attempt(
        paths,
        attempt_id,
        "improving",
        lease,
        now,
        improvement_attempts=1,
        improvement_reason="Correct evidence",
    )
    second_id = attestations.persist(
        paths,
        attestation_record(
            "validation",
            attempt_id,
            attempt_number=2,
            source_commit="7" * 40,
        ),
    )
    attempts.transition_attested(
        paths, attempt_id, "validated", second_id, {}, lease, now
    )

    improved = attempts.transition_attempt(
        paths,
        attempt_id,
        "improving",
        lease,
        now,
        improvement_attempts=2,
        improvement_reason="Correct a second observed predeployment failure",
    )

    assert improved["phase"] == "improving"
    assert improved["improvement_attempts"] == 2

    third_id = attestations.persist(
        paths,
        attestation_record(
            "validation",
            attempt_id,
            attempt_number=3,
            source_commit="8" * 40,
        ),
    )
    validated = attempts.transition_attested(
        paths, attempt_id, "validated", third_id, {}, lease, now
    )
    assert validated["phase"] == "validated"


def persist_verdict_snapshot(paths, *, sha="space-sha-1", paper_id="p1"):
    refresh = load_module("refresh")
    return refresh.persist_snapshot(
        paths,
        {
            "fetched_at": "2026-07-24T13:00:00+00:00",
            "source_revision": "source-a",
            "sources": {},
            "assessments": None,
            "candidates": [],
            "queued_submissions": [],
            "tagged_spaces": [],
            "verdicts": [
                {
                    "paper_id": paper_id,
                    "space_id": "wrice/repro-paper-1",
                    "sha": sha,
                    "judged_at": "2026-07-24T12:30:00+00:00",
                }
            ],
            "spaces": [],
        },
    )


def deploy_attempt(attempts, attestations, paths, attempt_id, lease, now):
    for target, updates in (
        ("validated", {}),
        (
            "deployed",
            {
                "space_id": "wrice/repro-paper-1",
                "deployed_sha": "space-sha-1",
            },
        ),
    ):
        attestation_id = attestations.persist(
            paths, attestation_record(PHASE_KINDS[target], attempt_id)
        )
        attempts.transition_attested(
            paths, attempt_id, target, attestation_id, updates, lease, now
        )


def test_deployed_attempt_allows_postverdict_correction(
    paths, attempts_and_leases, attempts, attestations, now
):
    attempt_id, lease, _, _ = attempts_and_leases
    deploy_attempt(attempts, attestations, paths, attempt_id, lease, now)
    snapshot_id = persist_verdict_snapshot(paths)

    improving = attempts.transition_attempt(
        paths,
        attempt_id,
        "improving",
        lease,
        now,
        improvement_attempts=1,
        improvement_reason="Official verdict scored every claim inconclusive",
        verdict_snapshot_id=snapshot_id,
    )

    assert improving["phase"] == "improving"
    assert improving["improvement_attempts"] == 1
    assert "verdict_snapshot_id" not in improving


def test_postverdict_correction_requires_matching_official_verdict(
    paths, attempts_and_leases, attempts, attestations, now
):
    attempt_id, lease, _, _ = attempts_and_leases
    deploy_attempt(attempts, attestations, paths, attempt_id, lease, now)
    unjudged = persist_verdict_snapshot(paths, sha="other-sha")

    with pytest.raises(ValueError, match="verdict"):
        attempts.transition_attempt(
            paths,
            attempt_id,
            "improving",
            lease,
            now,
            improvement_attempts=1,
            improvement_reason="Official verdict scored every claim inconclusive",
            verdict_snapshot_id=unjudged,
        )
    with pytest.raises(ValueError, match="verdict_snapshot_id"):
        attempts.transition_attempt(
            paths,
            attempt_id,
            "improving",
            lease,
            now,
            improvement_attempts=1,
            improvement_reason="Official verdict scored every claim inconclusive",
        )

    assert attempts.read_attempt(paths, attempt_id)["phase"] == "deployed"


@pytest.mark.parametrize(
    "deployed_update",
    [
        {"deployed_sha": "space-sha"},
        {"space_id": "owner/reproduction"},
    ],
)
def test_predeployment_correction_rejects_deployment_metadata(
    paths,
    attempts_and_leases,
    attempts,
    attestations,
    now,
    deployed_update,
):
    attempt_id, lease, _, _ = attempts_and_leases
    first_id = attestations.persist(
        paths, attestation_record("validation", attempt_id)
    )
    attempts.transition_attested(
        paths, attempt_id, "validated", first_id, {}, lease, now
    )
    attempts.update_attempt(paths, attempt_id, lease, now, **deployed_update)

    with pytest.raises(ValueError, match="deployment"):
        attempts.transition_attempt(
            paths,
            attempt_id,
            "improving",
            lease,
            now,
            improvement_attempts=1,
            improvement_reason="Correct evidence",
        )


def test_predeployment_correction_rejects_authoritative_deployment_attestation(
    paths,
    attempts_and_leases,
    attempts,
    attestations,
    store,
    now,
):
    attempt_id, lease, _, _ = attempts_and_leases
    first_id = attestations.persist(
        paths, attestation_record("validation", attempt_id)
    )
    attempts.transition_attested(
        paths, attempt_id, "validated", first_id, {}, lease, now
    )
    deployment_id = attestations.persist(
        paths, attestation_record("deployment", attempt_id)
    )
    deployment = attestations.read(paths, deployment_id)
    target = paths.attestation("deployment", attempt_id, 1)
    store.atomic_json_write(
        target,
        deployment,
        lambda record: attestations.validate_target(paths, target, record),
    )

    with pytest.raises(ValueError, match="deployment"):
        attempts.transition_attempt(
            paths,
            attempt_id,
            "improving",
            lease,
            now,
            improvement_attempts=1,
            improvement_reason="Correct evidence",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("improvement_attempts", 0),
        ("improvement_reason", ""),
        ("improvement_reason", "Rewrite correction history"),
    ],
)
def test_update_cannot_reset_or_rewrite_consumed_correction(
    paths,
    attempts_and_leases,
    attempts,
    attestations,
    now,
    field,
    value,
):
    attempt_id, lease, _, _ = attempts_and_leases
    expected = transition_through_correction(
        attempts, attestations, paths, attempt_id, lease, now
    )

    with pytest.raises(ValueError, match=field):
        attempts.update_attempt(
            paths, attempt_id, lease, now, **{field: value}
        )

    assert attempts.read_attempt(paths, attempt_id) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("improvement_attempts", 0),
        ("improvement_reason", "Rewrite correction history"),
    ],
)
def test_other_generic_transition_cannot_rewrite_consumed_correction(
    paths,
    attempts_and_leases,
    attempts,
    attestations,
    now,
    field,
    value,
):
    attempt_id, lease, _, _ = attempts_and_leases
    expected = transition_through_correction(
        attempts, attestations, paths, attempt_id, lease, now
    )

    with pytest.raises(ValueError, match=field):
        attempts.transition_attempt(
            paths,
            attempt_id,
            "blocked",
            lease,
            now,
            blocker="External outage",
            **{field: value},
        )

    assert attempts.read_attempt(paths, attempt_id) == expected


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"improvement_attempts": 0}, "improvement_reason"),
        ({"improvement_attempts": -1}, "improvement_attempts"),
        ({"improvement_reason": ""}, "improvement_reason"),
        ({"improvement_reason": None}, "improvement_reason"),
    ],
)
def test_persisted_attempt_validates_improvement_relationship(
    paths,
    attempts_and_leases,
    attempts,
    attestations,
    store,
    now,
    mutation,
    error,
):
    attempt_id, lease, _, _ = attempts_and_leases
    malformed = transition_through_correction(
        attempts, attestations, paths, attempt_id, lease, now
    )
    malformed.update(mutation)

    with pytest.raises(ValueError, match=error):
        store.validate_attempt(malformed)


def test_persisted_attempt_rejects_missing_reason_for_consumed_correction(
    paths,
    attempts_and_leases,
    attempts,
    attestations,
    store,
    now,
):
    attempt_id, lease, _, _ = attempts_and_leases
    malformed = transition_through_correction(
        attempts, attestations, paths, attempt_id, lease, now
    )
    malformed.pop("improvement_reason")

    with pytest.raises(ValueError, match="improvement_reason"):
        store.validate_attempt(malformed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_path", "submissions/other-paper"),
        ("target_claims", ["forged-claim-1", "forged-claim-2"]),
    ],
)
def test_predeployment_correction_rejects_surplus_updates(
    paths,
    attempts_and_leases,
    attempts,
    attestations,
    now,
    field,
    value,
):
    attempt_id, lease, _, _ = attempts_and_leases
    first_id = attestations.persist(
        paths, attestation_record("validation", attempt_id)
    )
    validated = attempts.transition_attested(
        paths, attempt_id, "validated", first_id, {}, lease, now
    )

    with pytest.raises(ValueError, match="updates"):
        attempts.transition_attempt(
            paths,
            attempt_id,
            "improving",
            lease,
            now,
            improvement_attempts=1,
            improvement_reason="Correct evidence",
            **{field: value},
        )

    assert attempts.read_attempt(paths, attempt_id) == validated


def test_consumed_correction_rejects_later_judgment_improvement(
    paths, attempts_and_leases, attempts, attestations, now
):
    attempt_id, lease, _, _ = attempts_and_leases
    transition_through_correction(
        attempts, attestations, paths, attempt_id, lease, now
    )
    for phase in ("deployed", "submitted", "judging"):
        attestation_id = attestations.persist(
            paths,
            attestation_record(
                PHASE_KINDS[phase], attempt_id, attempt_number=2
            ),
        )
        attempts.transition_attested(
            paths, attempt_id, phase, attestation_id, {}, lease, now
        )

    with pytest.raises(ValueError, match="attestation"):
        attempts.transition_attempt(
            paths,
            attempt_id,
            "improving",
            lease,
            now,
            improvement_attempts=1,
            improvement_reason="Official verdict requested stronger evidence",
        )

    assert attempts.read_attempt(paths, attempt_id)["phase"] == "judging"


def test_generic_judgment_improvement_requires_verdict_attestation(
    paths, attempts_and_leases, attempts, attestations, now
):
    attempt_id, lease, _, _ = attempts_and_leases
    transition_attested_to(
        attempts, attestations, paths, attempt_id, "judging", lease, now
    )
    expected = attempts.read_attempt(paths, attempt_id)

    with pytest.raises(ValueError, match="attestation"):
        attempts.transition_attempt(
            paths,
            attempt_id,
            "improving",
            lease,
            now,
            improvement_attempts=1,
            improvement_reason="Official verdict requested stronger evidence",
        )

    assert attempts.read_attempt(paths, attempt_id) == expected


def test_deployment_for_delimiter_prefixed_attempt_does_not_block_correction(
    paths,
    attempts_and_leases,
    attempts,
    attestations,
    store,
    now,
):
    attempt_id, lease, _, _ = attempts_and_leases
    first_id = attestations.persist(
        paths, attestation_record("validation", attempt_id)
    )
    attempts.transition_attested(
        paths, attempt_id, "validated", first_id, {}, lease, now
    )
    other_id = f"{attempt_id}--other"
    deployment_id = attestations.persist(
        paths, attestation_record("deployment", other_id)
    )
    deployment = attestations.read(paths, deployment_id)
    target = paths.attestation("deployment", other_id, 1)
    store.atomic_json_write(
        target,
        deployment,
        lambda record: attestations.validate_target(paths, target, record),
    )

    improving = attempts.transition_attempt(
        paths,
        attempt_id,
        "improving",
        lease,
        now,
        improvement_attempts=1,
        improvement_reason="Correct evidence",
    )

    assert improving["phase"] == "improving"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("improvement_attempts", 1),
        ("improvement_reason", "Forged correction"),
    ],
)
def test_attested_transition_rejects_improvement_metadata_update(
    paths, attempts_and_leases, attempts, attestations, now, field, value
):
    attempt_id, lease, _, _ = attempts_and_leases
    attestation_id = attestations.persist(
        paths, attestation_record("validation", attempt_id)
    )

    with pytest.raises(ValueError, match=field):
        attempts.transition_attested(
            paths,
            attempt_id,
            "validated",
            attestation_id,
            {field: value},
            lease,
            now,
        )

    assert attempts.read_attempt(paths, attempt_id)["phase"] == "implementing"


def test_rejected_attested_transition_leaves_slot_free_for_corrected_record(
    paths, attempts_and_leases, attempts, attestations, now
):
    attempt_id, lease, _, _ = attempts_and_leases
    rejected_id = attestations.persist(
        paths,
        attestation_record(
            "validation", attempt_id, source_commit="6" * 40
        ),
    )

    with pytest.raises(ValueError, match="attestation"):
        attempts.transition_attested(
            paths, attempt_id, "deployed", rejected_id, {}, lease, now
        )
    assert not paths.attestation("validation", attempt_id).exists()

    corrected_id = attestations.persist(
        paths,
        attestation_record(
            "validation", attempt_id, source_commit="7" * 40
        ),
    )
    transitioned = attempts.transition_attested(
        paths, attempt_id, "validated", corrected_id, {}, lease, now
    )

    assert transitioned["phase"] == "validated"
    assert transitioned["transitions"][-1]["attestation_id"] == corrected_id


def test_authoritative_attestation_slot_rejects_different_record_reuse(
    paths, attempts_and_leases, attempts, attestations, now
):
    attempt_id, lease, _, _ = attempts_and_leases
    first_id = attestations.persist(
        paths, attestation_record("validation", attempt_id)
    )
    attempts.transition_attested(
        paths, attempt_id, "validated", first_id, {}, lease, now
    )
    attempts.transition_attempt(
        paths, attempt_id, "blocked", lease, now, blocker="retry requested"
    )
    original = paths.attestation("validation", attempt_id).read_bytes()
    conflicting_id = attestations.persist(
        paths,
        attestation_record(
            "validation", attempt_id, source_commit="8" * 40
        ),
    )

    with pytest.raises(ValueError, match="attestation"):
        attempts.transition_attested(
            paths, attempt_id, "validated", conflicting_id, {}, lease, now
        )

    assert paths.attestation("validation", attempt_id).read_bytes() == original
    assert attempts.read_attempt(paths, attempt_id)["phase"] == "blocked"


def test_blocking_requires_nonempty_blocker(paths, attempts_and_leases, attempts, now):
    a1, lease, _, _ = attempts_and_leases
    with pytest.raises(ValueError, match="blocker"):
        attempts.transition_attempt(paths, a1, "blocked", lease, now, blocker="")


def test_completion_moves_attempt_reference_to_history(
    paths, attempts_and_leases, attempts, attestations, store, now
):
    a1, lease, _, _ = attempts_and_leases
    transition_attested_to(
        attempts, attestations, paths, a1, "complete", lease, now
    )

    index = store.read_json(paths.index)
    assert a1 not in index["attempts"]
    assert index["history"][a1]["phase"] == "complete"


def test_blocked_abandonment_must_be_explicit(
    paths, attempts_and_leases, attempts, store, now
):
    a1, lease, _, _ = attempts_and_leases
    attempts.transition_attempt(paths, a1, "blocked", lease, now, blocker="offline")
    with pytest.raises(ValueError, match="abandon"):
        attempts.transition_attempt(paths, a1, "idle", lease, now)

    attempts.transition_attempt(paths, a1, "idle", lease, now, abandon=True)
    index = store.read_json(paths.index)
    assert a1 not in index["attempts"]
    assert a1 in index["history"]


def test_stale_attempt_writer_cannot_update_successor(
    paths, attempts, leases, now
):
    stale = create(attempts, leases, paths, "a1", "p1", "worker-1", now)
    current = leases.claim_attempt(
        paths, "a1", "worker-2", stale.fencing_token, now + TTL
    )

    with pytest.raises(attempts.leases.StaleFence):
        attempts.update_attempt(
            paths, "a1", stale, now + TTL, assigned_agents=["stale"]
        )

    attempts.update_attempt(
        paths, "a1", current, now + TTL, assigned_agents=["current"]
    )
    assert attempts.read_attempt(paths, "a1")["assigned_agents"] == ["current"]


@pytest.mark.parametrize("fail_after", range(1, 5))
def test_interrupted_creation_recovers_without_orphan(
    paths, attempts, leases, store, now, monkeypatch, fail_after
):
    lease = leases.acquire_lease(paths, "attempt:a1", "worker-1", "a1", now, TTL)
    interrupt_transaction(monkeypatch, attempts, fail_after)

    with pytest.raises(OSError, match="simulated interruption"):
        attempts.create_attempt(paths, "a1", paper("p1"), lease, "snapshot-1", now)

    restore_transaction_writes(monkeypatch, attempts)
    attempts.recover_transactions(paths)
    index = store.read_json(paths.index)
    if paths.attempt("a1").exists():
        assert index["attempts"]["a1"]["path"].endswith("attempts/a1.json")
    else:
        assert "a1" not in index["attempts"]


@pytest.mark.parametrize("fail_after", range(1, 5))
def test_interrupted_update_recovers_matching_shard_and_reference(
    paths, attempts, leases, store, now, monkeypatch, fail_after
):
    lease = create(attempts, leases, paths, "a1", "p1", "worker-1", now)
    later = now + timedelta(minutes=1)
    interrupt_transaction(monkeypatch, attempts, fail_after)

    with pytest.raises(OSError, match="simulated interruption"):
        attempts.update_attempt(paths, "a1", lease, later, worktree="worktrees/a1")

    restore_transaction_writes(monkeypatch, attempts)
    attempts.recover_transactions(paths)
    attempt = attempts.read_attempt(paths, "a1")
    reference = store.read_json(paths.index)["attempts"]["a1"]
    assert reference["phase"] == attempt["phase"]
    assert reference["updated_at"] == attempt["updated_at"]


@pytest.mark.parametrize("fail_after", range(1, 5))
def test_interrupted_completion_never_recovers_complete_attempt_as_active(
    paths,
    attempts_and_leases,
    attempts,
    attestations,
    store,
    now,
    monkeypatch,
    fail_after,
):
    attempt_id, lease, _, _ = attempts_and_leases
    transition_attested_to(
        attempts, attestations, paths, attempt_id, "judging", lease, now
    )
    attestation_id = attestations.persist(
        paths, attestation_record("verdict", attempt_id)
    )
    interrupt_transaction(monkeypatch, attempts, fail_after)

    with pytest.raises(OSError, match="simulated interruption"):
        attempts.transition_attested(
            paths, attempt_id, "complete", attestation_id, {}, lease, now
        )

    restore_transaction_writes(monkeypatch, attempts)
    attempts.recover_transactions(paths)
    attempt = attempts.read_attempt(paths, attempt_id)
    index = store.read_json(paths.index)
    assert not (
        attempt["phase"] == "complete" and attempt_id in index["attempts"]
    )
    if attempt["phase"] == "complete":
        assert attempt_id in index["history"]


def test_attested_transition_transaction_recovers_attestation_attempt_and_index(
    paths,
    attempts_and_leases,
    attempts,
    attestations,
    store,
    now,
    monkeypatch,
):
    attempt_id, lease, _, _ = attempts_and_leases
    attestation_id = attestations.persist(
        paths, attestation_record("validation", attempt_id)
    )
    assert not paths.attestation("validation", attempt_id).exists()
    interrupt_transaction(monkeypatch, attempts, fail_after=3)

    with pytest.raises(OSError, match="simulated interruption"):
        attempts.transition_attested(
            paths, attempt_id, "validated", attestation_id, {}, lease, now
        )

    planned = [
        store.read_json(path)
        for path in (paths.root / "transactions" / "attempts").glob("*.json")
        if store.read_json(path)["status"] == "planned"
    ]
    assert len(planned) == 1
    assert {target["path"] for target in planned[0]["targets"]} == {
        str(
            attestations.object_path(paths, attestation_id).relative_to(
                paths.index.parent
            )
        ),
        str(
            paths.attestation("validation", attempt_id).relative_to(
                paths.index.parent
            )
        ),
        str(paths.attempt(attempt_id).relative_to(paths.index.parent)),
        paths.index.name,
    }

    restore_transaction_writes(monkeypatch, attempts)
    attempts.recover_transactions(paths)
    recovered = attempts.read_attempt(paths, attempt_id)
    assert recovered["phase"] == "validated"
    assert recovered["transitions"][-1]["attestation_id"] == attestation_id
    assert attestations.read(paths, attestation_id)["attempt_id"] == attempt_id


def test_recovery_rejects_attestation_target_path_mismatching_identity(
    paths,
    attempts_and_leases,
    attempts,
    attestations,
    store,
    now,
    monkeypatch,
):
    attempt_id, lease, _, _ = attempts_and_leases
    attestation_id = attestations.persist(
        paths, attestation_record("validation", attempt_id)
    )
    interrupt_transaction(monkeypatch, attempts, fail_after=2)

    with pytest.raises(OSError, match="simulated interruption"):
        attempts.transition_attested(
            paths, attempt_id, "validated", attestation_id, {}, lease, now
        )

    transaction_path = next(
        path
        for path in (paths.root / "transactions" / "attempts").glob("*.json")
        if store.read_json(path)["status"] == "planned"
    )
    manifest = store.read_json(transaction_path)
    attestation_target = next(
        target
        for target in manifest["targets"]
        if "/attestations/" in target["path"]
    )
    attestation_target["path"] = (
        "repro-loop/attestations/validation/different-attempt--1.json"
    )
    store._atomic_json_write(transaction_path, manifest)
    restore_transaction_writes(monkeypatch, attempts)

    with pytest.raises(ValueError, match="attestation"):
        attempts.recover_transactions(paths)
    assert not (
        paths.root
        / "attestations"
        / "validation"
        / "different-attempt--1.json"
    ).exists()


def test_successor_takeover_between_precheck_and_write_fences_stale_writer(
    paths, attempts, leases, now, monkeypatch
):
    stale = create(attempts, leases, paths, "a1", "p1", "worker-1", now)
    prechecked = threading.Event()
    continue_write = threading.Event()
    errors = []
    real_assert = attempts._assert_attempt_fence

    def pause_after_precheck(*args):
        real_assert(*args)
        prechecked.set()
        assert continue_write.wait(timeout=5)

    monkeypatch.setattr(attempts, "_assert_attempt_fence", pause_after_precheck)

    def stale_update():
        try:
            attempts.update_attempt(
                paths,
                "a1",
                stale,
                now + timedelta(minutes=1),
                assigned_agents=["stale"],
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    thread = threading.Thread(target=stale_update)
    thread.start()
    assert prechecked.wait(timeout=5)
    successor = leases.acquire_lease(
        paths, "attempt:a1", "worker-2", "a1", now + TTL, TTL
    )
    continue_write.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], attempts.leases.StaleFence)
    assert "assigned_agents" not in attempts.read_attempt(paths, "a1")
    attempts.update_attempt(
        paths, "a1", successor, now + TTL, assigned_agents=["current"]
    )


def test_update_cannot_inject_design_or_review(paths, pending_attempt, attempts, now):
    for field, value in (
        ("design", {"author": "forged", "path": "design.md"}),
        ("design_review", {"reviewer": "forged", "decision": "approved"}),
    ):
        with pytest.raises(ValueError, match=field):
            attempts.update_attempt(
                paths,
                pending_attempt.id,
                pending_attempt.lease,
                now,
                **{field: value},
            )


def test_transition_cannot_inject_approval(paths, pending_attempt, attempts, now):
    with pytest.raises(ValueError, match="design_review"):
        attempts.transition_attempt(
            paths,
            pending_attempt.id,
            "implementing",
            pending_attempt.lease,
            now,
            design_review={
                "reviewer": "forged-reviewer",
                "decision": "approved",
            },
        )


def test_twenty_first_runnable_attempt_is_rejected(paths, attempts, leases, now):
    for number in range(20):
        create(
            attempts,
            leases,
            paths,
            f"a{number}",
            f"p{number}",
            f"worker-{number}",
            now,
        )
    lease = leases.acquire_lease(
        paths, "attempt:a20", "worker-20", "a20", now, TTL
    )

    with pytest.raises(ValueError, match="max_runnable_attempts"):
        attempts.create_attempt(
            paths, "a20", paper("p20"), lease, "snapshot-1", now
        )

    assert len(attempts.runnable_attempt_ids(paths)) == 20


def test_concurrent_admission_never_exceeds_twenty(paths, attempts, leases, now):
    barrier = threading.Barrier(21)
    created = []
    errors = []

    def admit(number):
        attempt_id = f"a{number}"
        lease = leases.acquire_lease(
            paths,
            f"attempt:{attempt_id}",
            f"worker-{number}",
            attempt_id,
            now,
            TTL,
        )
        barrier.wait()
        try:
            attempts.create_attempt(
                paths,
                attempt_id,
                paper(f"p{number}"),
                lease,
                "snapshot-1",
                now,
            )
            created.append(attempt_id)
        except ValueError as error:
            errors.append(error)

    threads = [
        threading.Thread(target=admit, args=(number,)) for number in range(21)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(created) == 20
    assert len(errors) == 1
    assert str(errors[0]) == "max_runnable_attempts"
    assert len(attempts.runnable_attempt_ids(paths)) == 20


def test_repeated_payload_after_intervening_update_is_persisted(
    paths, attempts, leases, now
):
    lease = create(attempts, leases, paths, "a1", "p1", "worker-1", now)
    attempts.update_attempt(paths, "a1", lease, now, assigned_agents=["agent-a"])
    attempts.update_attempt(paths, "a1", lease, now, assigned_agents=["agent-b"])

    returned = attempts.update_attempt(
        paths, "a1", lease, now, assigned_agents=["agent-a"]
    )

    assert returned["assigned_agents"] == ["agent-a"]
    assert attempts.read_attempt(paths, "a1")["assigned_agents"] == ["agent-a"]
