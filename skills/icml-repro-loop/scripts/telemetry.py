"""Append-only controller telemetry for reproduction-loop operations."""

from collections.abc import Iterator
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from uuid import uuid4


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from store import StatePaths, read_json, validate_id  # noqa: E402


RESERVED_KEYS = {"version", "session_id", "sequence", "event"}
EVENT_FILE_PATTERN = re.compile(r"[0-9]+-[A-Za-z0-9._-]+\.json")


def run_stage(
    paths: StatePaths,
    attempt_id: str,
    stage: str,
    operation,
    *,
    utc_now=None,
    monotonic_ns=time.monotonic_ns,
    session_id_factory=None,
) -> object:
    """Measure one authoritative controller operation without changing it."""
    utc_now = utc_now or _utc_now
    session_id_factory = session_id_factory or (lambda: uuid4().hex)
    session_id = session_id_factory()
    started_counter = monotonic_ns()
    append_event(
        paths,
        session_id,
        0,
        "stage-started",
        {
            "attempt_id": attempt_id,
            "stage": stage,
            "observed_at": utc_now(),
            "monotonic_ns": started_counter,
        },
    )
    try:
        result = operation()
    except BaseException as error:
        try:
            finished_counter = monotonic_ns()
            append_event(
                paths,
                session_id,
                1,
                "stage-finished",
                {
                    "observed_at": utc_now(),
                    "monotonic_ns": finished_counter,
                    "elapsed_seconds": (
                        finished_counter - started_counter
                    )
                    / 1_000_000_000,
                    "outcome": "failed",
                    "error_type": type(error).__name__,
                },
            )
        except BaseException:
            pass
        raise

    try:
        finished_counter = monotonic_ns()
        payload = {
            "observed_at": utc_now(),
            "monotonic_ns": finished_counter,
            "elapsed_seconds": (finished_counter - started_counter)
            / 1_000_000_000,
            "outcome": "passed",
        }
        attestation_id = _result_attestation_id(result)
        if attestation_id is not None:
            payload["attestation_id"] = attestation_id
        append_event(paths, session_id, 1, "stage-finished", payload)
    except BaseException:
        pass
    return result


def record_observation(
    paths: StatePaths,
    attempt_id: str,
    name: str,
    snapshot_id: str,
    result: object,
    *,
    utc_now=None,
    session_id_factory=None,
) -> dict:
    """Record one successful immutable controller observation."""
    utc_now = utc_now or _utc_now
    session_id_factory = session_id_factory or (lambda: uuid4().hex)
    attestation_id = _result_attestation_id(result)
    if attestation_id is None:
        raise ValueError("attestation_id")
    return append_event(
        paths,
        session_id_factory(),
        0,
        "observation",
        {
            "name": name,
            "attempt_id": attempt_id,
            "snapshot_id": snapshot_id,
            "attestation_id": attestation_id,
            "observed_at": utc_now(),
        },
    )


def append_event(
    paths: StatePaths,
    session_id: str,
    sequence: int,
    event: str,
    payload: dict,
) -> dict:
    """Durably create one immutable controller event record."""
    path = paths.telemetry_event(session_id, sequence, event)
    if type(payload) is not dict:
        raise ValueError("payload")
    if RESERVED_KEYS & payload.keys():
        raise ValueError("reserved payload key")
    record = {
        **payload,
        "version": 1,
        "session_id": session_id,
        "sequence": sequence,
        "event": event,
    }
    encoded = (
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())
    _fsync_directory(path.parent)
    return record


def read_session(paths: StatePaths, session_id: str) -> list[dict]:
    """Return one verified telemetry history without repairing corruption."""
    validate_id(session_id)
    directory = paths.root / "telemetry" / session_id
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError("session")

    event_paths = []
    for path in directory.iterdir():
        if not path.is_file() or EVENT_FILE_PATTERN.fullmatch(path.name) is None:
            raise ValueError("event file")
        event_paths.append(path)
    event_paths.sort(key=_event_path_key)

    records = []
    for expected_sequence, path in enumerate(event_paths):
        record = read_json(path)
        _validate_event_record(record, session_id, expected_sequence)
        if path != paths.telemetry_event(
            session_id, record["sequence"], record["event"]
        ):
            raise ValueError("event file")
        records.append(record)
    return records


def iter_sessions(paths: StatePaths) -> Iterator[list[dict]]:
    """Yield every validated session history in deterministic session order."""
    directory = paths.root / "telemetry"
    if not directory.exists():
        return
    if not directory.is_dir():
        raise ValueError("telemetry")
    session_ids = []
    for path in directory.iterdir():
        if not path.is_dir():
            raise ValueError("session")
        session_ids.append(validate_id(path.name))
    for session_id in sorted(session_ids):
        yield read_session(paths, session_id)


def summarize_worker_session(events: list[dict]) -> dict:
    """Derive status and duration only from a complete monotonic interval."""
    if type(events) is not list or not events:
        raise ValueError("events")
    session_id = _validate_event_history(events)
    launched = next(
        (event for event in events if event["event"] == "worker-launched"),
        None,
    )
    exited = next(
        (event for event in events if event["event"] == "worker-exited"),
        None,
    )
    elapsed_seconds = None
    if launched is not None and exited is not None:
        launch_counter = launched.get("monotonic_ns")
        exit_counter = exited.get("monotonic_ns")
        if _is_monotonic_counter(launch_counter) and _is_monotonic_counter(
            exit_counter
        ):
            if exit_counter >= launch_counter:
                elapsed_seconds = (exit_counter - launch_counter) / 1_000_000_000
    return {
        "session_id": session_id,
        "status": "exited" if exited is not None else "open",
        "elapsed_seconds": elapsed_seconds,
    }


def _event_path_key(path: Path) -> tuple[int, str]:
    sequence, _separator, _event = path.name.partition("-")
    return (int(sequence), path.name)


def _validate_event_history(events: list[dict]) -> str:
    session_id: str | None = None
    for expected_sequence, event in enumerate(events):
        if type(event) is not dict:
            raise ValueError("event")
        if session_id is None:
            value = event.get("session_id")
            validate_id(value)
            session_id = value
        _validate_event_record(event, session_id, expected_sequence)
    if session_id is None:  # pragma: no cover - guarded by caller
        raise ValueError("events")
    return session_id


def _validate_event_record(
    record: dict, session_id: str, expected_sequence: int
) -> None:
    if type(record.get("version")) is not int or record["version"] != 1:
        raise ValueError("version")
    if record.get("session_id") != session_id:
        raise ValueError("session_id")
    if (
        type(record.get("sequence")) is not int
        or record["sequence"] != expected_sequence
    ):
        raise ValueError("sequence")
    validate_id(record.get("event"))


def _is_monotonic_counter(value: object) -> bool:
    return type(value) is int and value >= 0


def _result_attestation_id(result: object) -> str | None:
    if type(result) is not dict:
        return None
    attestation_id = result.get("attestation_id")
    if type(attestation_id) is str:
        return attestation_id
    transitions = result.get("transitions")
    if type(transitions) is not list or not transitions:
        return None
    transition = transitions[-1]
    if type(transition) is not dict:
        return None
    attestation_id = transition.get("attestation_id")
    return attestation_id if type(attestation_id) is str else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
