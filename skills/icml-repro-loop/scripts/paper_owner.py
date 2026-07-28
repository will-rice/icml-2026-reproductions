"""Persistent paper-owner iteration release and coordinator events."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import attempts
import leases
import store
import telemetry


OUTCOMES = {"scored", "blocked"}
RELEASE_TRANSACTION_KEYS = {
    "version",
    "status",
    "attempt_id",
    "resource",
    "owner",
    "fencing_token",
    "lease_acquired_at",
    "lease_expires_at",
    "outcome",
    "released_at",
    "event",
}
RELEASE_EVENT_REQUIRED_KEYS = {
    "version",
    "session_id",
    "sequence",
    "event",
    "attempt_id",
    "paper_id",
    "phase",
    "owner",
    "fencing_token",
    "snapshot_id",
    "outcome",
    "reclaimable",
    "released_at",
}
RELEASE_EVENT_OPTIONAL_KEYS = {
    "space_id",
    "deployed_sha",
    "blocker",
    "next_action",
    "verdict",
}


def release_paper(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    outcome: str,
    now: datetime,
    *,
    session_id_factory=None,
) -> dict:
    """Durably and idempotently release one terminal paper-owner lease."""
    if outcome not in OUTCOMES:
        raise ValueError("outcome")
    _require_exact_attempt_lease(attempt_id, lease)
    observed_at = _datetime(now)
    transaction_path = paths.paper_owner_release(
        attempt_id, lease.fencing_token
    )
    existing = _preflight_release(
        paths,
        transaction_path,
        attempt_id,
        lease,
        outcome,
        observed_at,
    )
    session_id = None
    if existing is None:
        session_id = (session_id_factory or (lambda: uuid4().hex))()
        store.validate_id(session_id)

    # A valid mutating command is also a recovery boundary. Invalid or stale
    # inputs return above without repairing unrelated state.
    recover_release_transactions(paths)
    return _release_one(
        paths,
        transaction_path,
        attempt_id,
        lease,
        outcome,
        observed_at,
        session_id,
    )


def recover_release_transactions(paths: store.StatePaths) -> list[dict]:
    """Finish every prepared paper-owner release in deterministic order."""
    directory = paths.root / "paper-owner-releases"
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError("release transactions")
    recovered = []
    for transaction_path in sorted(directory.glob("*.json")):
        if not transaction_path.is_file():
            raise ValueError("release transaction")
        with store._exclusive_lock(transaction_path):
            transaction = store.read_json(transaction_path)
            _validate_release_transaction(paths, transaction_path, transaction)
            if transaction["status"] == "prepared":
                recovered.append(
                    _finish_release_transaction(paths, transaction_path, transaction)
                )
            else:
                _verify_completed_release(paths, transaction)
    return recovered


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


def _preflight_release(
    paths: store.StatePaths,
    transaction_path: Path,
    attempt_id: str,
    lease: leases.Lease,
    outcome: str,
    observed_at: datetime,
) -> dict | None:
    if transaction_path.exists():
        with store._exclusive_lock(transaction_path):
            transaction = store.read_json(transaction_path)
            _validate_release_transaction(
                paths, transaction_path, transaction
            )
            _require_matching_transaction(
                transaction, attempt_id, lease, outcome, observed_at
            )
            return transaction

    try:
        with leases.hold_fence(paths, lease, observed_at) as current:
            if observed_at < datetime.fromisoformat(current.acquired_at):
                raise ValueError("now")
            attempt = attempts.read_attempt(paths, attempt_id)
            _validate_release_attempt(attempt, outcome)
    except leases.StaleFence:
        # A concurrent identical release may have prepared its journal while
        # this caller waited for the attempt fence.
        if transaction_path.exists():
            with store._exclusive_lock(transaction_path):
                transaction = store.read_json(transaction_path)
                _validate_release_transaction(
                    paths, transaction_path, transaction
                )
                _require_matching_transaction(
                    transaction, attempt_id, lease, outcome, observed_at
                )
                return transaction
        raise
    return None


def _release_one(
    paths: store.StatePaths,
    transaction_path: Path,
    attempt_id: str,
    lease: leases.Lease,
    outcome: str,
    observed_at: datetime,
    session_id: str | None,
) -> dict:
    with store._exclusive_lock(transaction_path):
        if transaction_path.exists():
            transaction = store.read_json(transaction_path)
            _validate_release_transaction(
                paths, transaction_path, transaction
            )
            _require_matching_transaction(
                transaction, attempt_id, lease, outcome, observed_at
            )
            if transaction["status"] == "complete":
                _verify_completed_release(paths, transaction)
                return transaction["event"]
            return _finish_release_transaction(
                paths, transaction_path, transaction
            )

        if session_id is None:  # pragma: no cover - guarded by preflight
            raise ValueError("session_id")
        with leases.hold_fence(paths, lease, observed_at) as current:
            attempt = attempts.read_attempt(paths, attempt_id)
            _validate_release_attempt(attempt, outcome)
            event = {
                **_release_payload(
                    attempt_id, lease, outcome, observed_at, attempt
                ),
                "version": 1,
                "session_id": session_id,
                "sequence": 0,
                "event": "paper-owner-released",
            }
            transaction = {
                "version": 1,
                "status": "prepared",
                "attempt_id": attempt_id,
                "resource": lease.resource,
                "owner": lease.owner,
                "fencing_token": lease.fencing_token,
                "lease_acquired_at": lease.acquired_at,
                "lease_expires_at": lease.expires_at,
                "outcome": outcome,
                "released_at": observed_at.isoformat(),
                "event": event,
            }
            _validate_release_transaction(
                paths, transaction_path, transaction
            )
            store._atomic_json_write(transaction_path, transaction)
            recorded = telemetry.append_event(
                paths,
                session_id,
                0,
                "paper-owner-released",
                _event_payload(event),
            )
            if recorded != event:  # pragma: no cover - append contract
                raise ValueError("release transaction")
            leases._release_held_fence(paths, lease, observed_at)
        transaction["status"] = "complete"
        store._atomic_json_write(transaction_path, transaction)
        return event


def _finish_release_transaction(
    paths: store.StatePaths,
    transaction_path: Path,
    transaction: dict,
) -> dict:
    event = transaction["event"]
    lease_path = paths.resource_lease(transaction["resource"])
    released_at = _parse(transaction["released_at"], "released_at")
    with store._exclusive_lock(lease_path):
        current = leases._read_lease(lease_path)
        _require_transaction_lease(transaction, current)
        event_path = paths.telemetry_event(
            event["session_id"], event["sequence"], event["event"]
        )
        if event_path.exists():
            if store.read_json(event_path) != event:
                raise ValueError("release transaction")
        else:
            if current.released_at is not None:
                raise ValueError("release transaction")
            recorded = telemetry.append_event(
                paths,
                event["session_id"],
                event["sequence"],
                event["event"],
                _event_payload(event),
            )
            if recorded != event:  # pragma: no cover - append contract
                raise ValueError("release transaction")

        if current.released_at is None:
            retry_lease = leases.Lease(
                resource=current.resource,
                owner=current.owner,
                attempt_id=current.attempt_id,
                acquired_at=current.acquired_at,
                expires_at=current.expires_at,
                fencing_token=current.fencing_token,
            )
            leases._release_held_fence(paths, retry_lease, released_at)
        elif current.released_at != transaction["released_at"]:
            raise ValueError("release transaction")

    transaction["status"] = "complete"
    store._atomic_json_write(transaction_path, transaction)
    return event


def _verify_completed_release(
    paths: store.StatePaths, transaction: dict
) -> None:
    event = transaction["event"]
    event_path = paths.telemetry_event(
        event["session_id"], event["sequence"], event["event"]
    )
    if not event_path.exists() or store.read_json(event_path) != event:
        raise ValueError("release transaction")
    lease_path = paths.resource_lease(transaction["resource"])
    with store._exclusive_lock(lease_path):
        current = leases._read_lease(lease_path)
        if (
            current is not None
            and current.resource == transaction["resource"]
            and current.attempt_id == transaction["attempt_id"]
            and current.fencing_token > transaction["fencing_token"]
            and _parse(current.acquired_at, "release transaction")
            >= _parse(transaction["released_at"], "release transaction")
        ):
            return
        _require_transaction_lease(transaction, current)
        if current.released_at != transaction["released_at"]:
            raise ValueError("release transaction")


def _validate_release_transaction(
    paths: store.StatePaths,
    transaction_path: Path,
    transaction: dict,
) -> None:
    if (
        type(transaction) is not dict
        or set(transaction) != RELEASE_TRANSACTION_KEYS
        or transaction["version"] != 1
        or transaction["status"] not in {"prepared", "complete"}
        or transaction["outcome"] not in OUTCOMES
    ):
        raise ValueError("release transaction")
    attempt_id = store.validate_id(transaction["attempt_id"])
    if transaction_path != paths.paper_owner_release(
        attempt_id, transaction["fencing_token"]
    ):
        raise ValueError("release transaction")
    if transaction["resource"] != f"attempt:{attempt_id}":
        raise ValueError("release transaction")
    store.validate_id(transaction["owner"])
    _parse(transaction["lease_acquired_at"], "release transaction")
    expires_at = _parse(
        transaction["lease_expires_at"], "release transaction"
    )
    released_at = _parse(transaction["released_at"], "release transaction")
    if (
        released_at < _parse(
            transaction["lease_acquired_at"], "release transaction"
        )
        or released_at >= expires_at
    ):
        raise ValueError("release transaction")
    event = transaction["event"]
    telemetry._validate_event_record(event, event.get("session_id"), 0)
    if (
        not RELEASE_EVENT_REQUIRED_KEYS.issubset(event)
        or not set(event).issubset(
            RELEASE_EVENT_REQUIRED_KEYS | RELEASE_EVENT_OPTIONAL_KEYS
        )
        or event.get("event") != "paper-owner-released"
        or event.get("attempt_id") != attempt_id
        or event.get("owner") != transaction["owner"]
        or event.get("fencing_token") != transaction["fencing_token"]
        or event.get("outcome") != transaction["outcome"]
        or event.get("reclaimable")
        is not (transaction["outcome"] == "blocked")
        or event.get("released_at") != transaction["released_at"]
        or type(event.get("paper_id")) is not str
        or not event["paper_id"]
        or type(event.get("phase")) is not str
        or not event["phase"]
        or type(event.get("snapshot_id")) is not str
        or not event["snapshot_id"]
    ):
        raise ValueError("release transaction")


def _require_matching_transaction(
    transaction: dict,
    attempt_id: str,
    lease: leases.Lease,
    outcome: str,
    observed_at: datetime,
) -> None:
    if (
        transaction["attempt_id"] != attempt_id
        or transaction["resource"] != lease.resource
        or transaction["owner"] != lease.owner
        or transaction["fencing_token"] != lease.fencing_token
        or transaction["lease_acquired_at"] != lease.acquired_at
        or transaction["lease_expires_at"] != lease.expires_at
        or transaction["outcome"] != outcome
        or observed_at
        < _parse(transaction["released_at"], "release transaction")
        or lease.released_at not in {None, transaction["released_at"]}
    ):
        raise ValueError("release transaction")


def _require_transaction_lease(
    transaction: dict, current: leases.Lease | None
) -> None:
    if (
        current is None
        or current.resource != transaction["resource"]
        or current.attempt_id != transaction["attempt_id"]
        or current.owner != transaction["owner"]
        or current.fencing_token != transaction["fencing_token"]
        or current.acquired_at != transaction["lease_acquired_at"]
        or current.expires_at != transaction["lease_expires_at"]
    ):
        raise leases.StaleFence(transaction["resource"])


def _require_exact_attempt_lease(
    attempt_id: str, lease: leases.Lease
) -> None:
    if (
        lease.resource != f"attempt:{attempt_id}"
        or lease.attempt_id != attempt_id
    ):
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


def _event_payload(event: dict) -> dict:
    return {
        key: value
        for key, value in event.items()
        if key not in telemetry.RESERVED_KEYS
    }


def _datetime(value: object) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError("now")
    return value.astimezone(timezone.utc)


def _parse(value: object, field: str) -> datetime:
    if type(value) is not str:
        raise ValueError(field)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(field) from error
    try:
        return _datetime(parsed)
    except ValueError as error:
        raise ValueError(field) from error
