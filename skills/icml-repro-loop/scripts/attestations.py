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
KIND_KEYS["validation"] = frozenset(
    {
        "worktree",
        "branch",
        "base_sha",
        "project_path",
        "design_path",
        "commands",
        "checks",
        "environment",
        "source_tree",
        "source_tree_sha256",
        "environment_sha256",
    }
)
KIND_KEYS["deployment"] = frozenset(
    {
        "space_id",
        "space_sha",
        "owner",
        "tags",
        "runtime_stage",
        "validation_attestation_id",
        "source_tree_sha256",
    }
)
KIND_KEYS["submission"] = frozenset(
    {
        "snapshot_id",
        "verdict_revision",
        "space_id",
        "space_sha",
        "paper_id",
        "queue_status",
        "deployment_attestation_id",
    }
)
KIND_KEYS["authority-audit"] = frozenset(
    {
        "submission_attestation_id",
        "poll_limit",
        "poll_deadline",
        "space_id",
        "space_sha",
    }
)
KIND_KEYS["verdict"] = frozenset(
    {
        "snapshot_id",
        "verdict_revision",
        "submission_attestation_id",
        "authority_attestation_id",
        "space_id",
        "space_sha",
        "paper_id",
        "judged_at",
        "claims",
    }
)
OFFICIAL_VERDICT_STATUSES = {
    "verified",
    "falsified",
    "toy",
    "inconclusive",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
RESULT_KEYS = {"argv", "returncode", "stdout_sha256", "stderr_sha256"}


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
    if (
        type(record) is not dict
        or type(record.get("attestation_id")) is not str
        or _canonical_id(record) != record["attestation_id"]
    ):
        raise ValueError("attestation_id")
    _validate_record(record, persisted=True)


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
    if kind == "validation":
        _validate_validation(record)
    elif kind == "deployment":
        _validate_deployment(record)
    elif kind == "submission":
        _validate_submission(record)
    elif kind == "authority-audit":
        _validate_authority_audit(record)
    elif kind == "verdict":
        _validate_verdict(record)


def _validate_validation(record: dict) -> None:
    _git_sha(record["source_commit"], "source_commit")
    for field in (
        "worktree",
        "branch",
        "project_path",
        "design_path",
    ):
        _nonempty_string(record[field], field)
    _git_sha(record["base_sha"], "base_sha")
    _git_sha(record["source_tree"], "source_tree")
    _sha256(record["source_tree_sha256"], "source_tree_sha256")
    _sha256(record["environment_sha256"], "environment_sha256")
    for field in ("commands", "checks", "environment"):
        results = record[field]
        if type(results) is not list:
            raise ValueError(field)
        for result in results:
            _validate_result(result, field)
    _validate_payload_digest(record)


def _validate_deployment(record: dict) -> None:
    _git_sha(record["source_commit"], "source_commit")
    _nonempty_string(record["space_id"], "space_id")
    _nonempty_string(record["space_sha"], "space_sha")
    owner = _nonempty_string(record["owner"], "owner")
    if record["space_id"].partition("/")[0] != owner:
        raise ValueError("owner")
    tags = record["tags"]
    if (
        type(tags) is not list
        or any(type(tag) is not str or not tag for tag in tags)
        or tags != sorted(set(tags))
    ):
        raise ValueError("tags")
    if record["runtime_stage"] != "RUNNING":
        raise ValueError("runtime_stage")
    _sha256(
        record["validation_attestation_id"], "validation_attestation_id"
    )
    _sha256(record["source_tree_sha256"], "source_tree_sha256")
    _validate_payload_digest(record)


def _validate_submission(record: dict) -> None:
    _git_sha(record["source_commit"], "source_commit")
    _sha256(record["snapshot_id"], "snapshot_id")
    for field in (
        "verdict_revision",
        "space_id",
        "paper_id",
        "queue_status",
    ):
        _nonempty_string(record[field], field)
    _nonempty_string(record["space_sha"], "space_sha")
    _sha256(
        record["deployment_attestation_id"],
        "deployment_attestation_id",
    )
    _validate_payload_digest(record)


def _validate_authority_audit(record: dict) -> None:
    _git_sha(record["source_commit"], "source_commit")
    _sha256(
        record["submission_attestation_id"],
        "submission_attestation_id",
    )
    if type(record["poll_limit"]) is not int or record["poll_limit"] <= 0:
        raise ValueError("poll_limit")
    _timestamp(record["poll_deadline"])
    _nonempty_string(record["space_id"], "space_id")
    _nonempty_string(record["space_sha"], "space_sha")
    _validate_payload_digest(record)


def _validate_verdict(record: dict) -> None:
    _git_sha(record["source_commit"], "source_commit")
    _sha256(record["snapshot_id"], "snapshot_id")
    _sha256(
        record["submission_attestation_id"],
        "submission_attestation_id",
    )
    _sha256(
        record["authority_attestation_id"],
        "authority_attestation_id",
    )
    for field in ("verdict_revision", "space_id", "space_sha", "paper_id"):
        _nonempty_string(record[field], field)
    _timestamp(record["judged_at"])
    claims = record["claims"]
    if type(claims) is not list or not claims:
        raise ValueError("claims")
    targets = []
    challenge_claims = []
    for claim in claims:
        if type(claim) is not dict or set(claim) != {
            "target_claim",
            "claim",
            "status",
            "evidence",
        }:
            raise ValueError("claims")
        targets.append(_nonempty_string(claim["target_claim"], "claims"))
        challenge_claims.append(_nonempty_string(claim["claim"], "claims"))
        if claim["status"] not in OFFICIAL_VERDICT_STATUSES:
            raise ValueError("claims")
        if type(claim["evidence"]) is not str:
            raise ValueError("claims")
    if len(targets) != len(set(targets)) or len(challenge_claims) != len(
        set(challenge_claims)
    ):
        raise ValueError("claims")
    _validate_payload_digest(record)


def _validate_payload_digest(record: dict) -> None:
    payload = {key: record[key] for key in KIND_KEYS[record["kind"]]}
    expected = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if record["payload_sha256"] != expected:
        raise ValueError("payload_sha256")


def _validate_result(result: object, field: str) -> None:
    if type(result) is not dict or set(result) != RESULT_KEYS:
        raise ValueError(field)
    argv = result["argv"]
    if (
        type(argv) is not list
        or not argv
        or any(type(argument) is not str or not argument for argument in argv)
        or type(result["returncode"]) is not int
        or result["returncode"] != 0
    ):
        raise ValueError(field)
    _sha256(result["stdout_sha256"], field)
    _sha256(result["stderr_sha256"], field)


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


def _git_sha(value: object, field: str) -> str:
    if type(value) is not str or GIT_SHA_PATTERN.fullmatch(value) is None:
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
