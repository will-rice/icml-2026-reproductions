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
    """Release one terminal paper-owner lease after recording its event."""
    if outcome not in OUTCOMES:
        raise ValueError("outcome")
    _require_exact_attempt_lease(attempt_id, lease)
    with leases.hold_fence(paths, lease, now) as current:
        if now < datetime.fromisoformat(current.acquired_at):
            raise ValueError("now")
        attempt = attempts.read_attempt(paths, attempt_id)
        _validate_release_attempt(attempt, outcome)
        event = telemetry.append_event(
            paths,
            (session_id_factory or (lambda: uuid4().hex))(),
            0,
            "paper-owner-released",
            _release_payload(attempt_id, lease, outcome, now, attempt),
        )
        leases._release_held_fence(paths, lease, now)
        return event


def record_worker_failure(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    error_type: str,
    now: datetime,
    *,
    session_id_factory=None,
) -> dict:
    """Record an owner failure while retaining its lease for recovery."""
    if type(error_type) is not str or not error_type.strip():
        raise ValueError("error_type")
    _require_exact_attempt_lease(attempt_id, lease)
    with leases.hold_fence(paths, lease, now) as current:
        if now < datetime.fromisoformat(current.acquired_at):
            raise ValueError("now")
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


def _require_exact_attempt_lease(attempt_id: str, lease: leases.Lease) -> None:
    if lease.resource != f"attempt:{attempt_id}" or lease.attempt_id != attempt_id:
        raise leases.StaleFence(f"attempt:{attempt_id}")


def _validate_release_attempt(attempt: dict, outcome: str) -> None:
    expected_phase = "complete" if outcome == "scored" else "blocked"
    if attempt["phase"] != expected_phase:
        raise ValueError("phase")
    if outcome == "scored" and attempt.get("verdict") is None:
        raise ValueError("verdict")
    if outcome == "blocked":
        for field in ("blocker", "next_action"):
            if type(attempt.get(field)) is not str or not attempt[field]:
                raise ValueError(field)


def _release_payload(
    attempt_id: str,
    lease: leases.Lease,
    outcome: str,
    now: datetime,
    attempt: dict,
) -> dict:
    payload = {
        "attempt_id": attempt_id,
        "paper_id": attempt["paper_id"],
        "phase": attempt["phase"],
        "owner": lease.owner,
        "fencing_token": lease.fencing_token,
        "snapshot_id": attempt["snapshot_id"],
        "outcome": outcome,
        "reclaimable": outcome == "blocked",
        "released_at": now.isoformat(),
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
    return payload
