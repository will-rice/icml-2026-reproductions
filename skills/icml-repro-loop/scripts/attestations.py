"""Immutable attestations for authoritative external lifecycle observations."""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import store  # noqa: E402


COMMON_KEYS = {
    "attestation_id",
    "kind",
    "attempt_id",
    "attempt_number",
    "observed_at",
    "source_commit",
    "payload_sha256",
}
KINDS = {"validation", "deployment", "submission", "verdict", "authority-audit"}
KIND_KEYS = {kind: frozenset() for kind in KINDS}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def persist(paths: store.StatePaths, record: dict) -> str:
    """Persist one canonical content-addressed attestation exactly once."""
    payload = copy.deepcopy(record)
    _validate_record(payload, persisted=False)
    attestation_id = _canonical_id(payload)
    supplied_id = payload.get("attestation_id")
    if supplied_id is not None and supplied_id != attestation_id:
        raise ValueError("attestation_id")
    persisted = {"attestation_id": attestation_id, **payload}
    _validate_record(persisted, persisted=True)
    path = _object_path(paths, attestation_id)
    expected_bytes = _file_bytes(persisted)
    with store._exclusive_lock(path):
        if path.exists():
            if path.read_bytes() != expected_bytes:
                raise ValueError("attestation")
        else:
            store._atomic_json_write(path, persisted)
    return attestation_id


def read(paths: store.StatePaths, attestation_id: str) -> dict:
    """Read and content-verify one immutable attestation by its canonical ID."""
    _sha256(attestation_id, "attestation_id")
    path = _object_path(paths, attestation_id)
    if not path.exists():
        raise ValueError("attestation_id")
    record = store.read_json(path)
    validate_record(record)
    if record["attestation_id"] != attestation_id:
        raise ValueError("attestation_id")
    return copy.deepcopy(record)


def validate_record(record: dict) -> None:
    """Validate an exact persisted attestation and its content address."""
    _validate_record(record, persisted=True)
    if _canonical_id(record) != record["attestation_id"]:
        raise ValueError("attestation_id")


def validate_target(paths: store.StatePaths, path: Path, record: dict) -> None:
    """Validate one immutable record against its authoritative slot path."""
    validate_record(record)
    expected = paths.attestation(
        record["kind"], record["attempt_id"], record["attempt_number"]
    )
    if path != expected:
        raise ValueError("attestation")
    if path.exists() and path.read_bytes() != _file_bytes(record):
        raise ValueError("attestation")


def _validate_record(record: object, *, persisted: bool) -> None:
    if type(record) is not dict:
        raise ValueError("attestation")
    kind = record.get("kind")
    if kind not in KINDS:
        raise ValueError("kind")
    expected = COMMON_KEYS | KIND_KEYS[kind]
    allowed = expected if persisted else expected - {"attestation_id"}
    if not persisted and "attestation_id" in record:
        allowed = expected
    if set(record) != allowed:
        raise ValueError("keys")
    if "attestation_id" in record:
        _sha256(record["attestation_id"], "attestation_id")
    store.validate_id(record["attempt_id"])
    if type(record["attempt_number"]) is not int or record["attempt_number"] < 1:
        raise ValueError("attempt_number")
    _timestamp(record["observed_at"])
    _nonempty_string(record["source_commit"], "source_commit")
    _sha256(record["payload_sha256"], "payload_sha256")


def _canonical_id(record: dict) -> str:
    payload = {
        key: value for key, value in record.items() if key != "attestation_id"
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _canonical_json(value: dict) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _object_path(paths: store.StatePaths, attestation_id: str) -> Path:
    return paths.root / "attestation-objects" / f"{attestation_id}.json"


def _file_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _nonempty_string(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(field)
    return value


def _sha256(value: object, field: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(field)
    return value


def _timestamp(value: object) -> str:
    value = _nonempty_string(value, "observed_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("observed_at") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("observed_at")
    return value
