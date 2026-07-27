"""Crash-safe JSON shards for schema-v6 reproduction-loop state."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


Validator = Callable[[dict], None]
ValidatorForPath = Callable[[Path], Validator]
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
INDEX_KEYS = {
    "version",
    "max_runnable_attempts",
    "attempts",
    "history",
    "rejections",
    "snapshots",
    "resource_limits",
    "total_api_cost_usd",
}
ATTEMPT_REFERENCE_KEYS = {"path", "paper_id", "phase", "updated_at"}
RESOURCE_LIMIT_KEYS = {
    "metered_api_reserved_usd",
    "publication_per_provider",
}
TRANSACTION_KEYS = {"status", "targets"}
TRANSACTION_TARGET_KEYS = {"path", "value"}


def validate_id(identifier: str) -> str:
    """Return a filesystem-safe state identifier."""
    if type(identifier) is not str or IDENTIFIER_PATTERN.fullmatch(identifier) is None:
        raise ValueError("identifier")
    return identifier


@dataclass(frozen=True, slots=True)
class StatePaths:
    """Resolve the index and its independently writable shard paths."""

    index: Path

    @property
    def root(self) -> Path:
        return self.index.with_suffix("")

    def attempt(self, attempt_id: str) -> Path:
        return self.root / "attempts" / f"{validate_id(attempt_id)}.json"

    def judgment(self, attempt_id: str) -> Path:
        return self.root / "judgments" / f"{validate_id(attempt_id)}.json"

    def judgment_archive(self, attempt_id: str, attempt_number: int) -> Path:
        if type(attempt_number) is not int or attempt_number < 1:
            raise ValueError("attempt_number")
        return (
            self.root
            / "judgments"
            / "archive"
            / f"{validate_id(attempt_id)}--{attempt_number}.json"
        )

    def attestation(
        self, kind: str, attempt_id: str, attempt_number: int = 1
    ) -> Path:
        """Resolve one immutable external-lifecycle attestation slot."""
        validate_id(kind)
        validate_id(attempt_id)
        if type(attempt_number) is not int or attempt_number < 1:
            raise ValueError("attempt_number")
        return (
            self.root
            / "attestations"
            / kind
            / f"{attempt_id}--{attempt_number}.json"
        )

    def authority_audit(self, report_id: str) -> Path:
        return (
            self.root
            / "authority-audits"
            / f"{validate_id(report_id)}.json"
        )

    def quarantine(self, attempt_id: str) -> Path:
        return self.root / "quarantine" / validate_id(attempt_id)

    def quarantine_manifest(self, attempt_id: str) -> Path:
        return self.quarantine(attempt_id) / "manifest.json"

    def lease(self, lease_id: str) -> Path:
        return self.root / "leases" / f"{validate_id(lease_id)}.json"

    def resource_lease(self, resource: str) -> Path:
        """Resolve a resource identity to a collision-resistant lease shard."""
        if type(resource) is not str or not resource:
            raise ValueError("resource")
        kind, separator, _identity = resource.partition(":")
        if not separator:
            raise ValueError("resource")
        validate_id(kind)
        digest = hashlib.sha256(resource.encode("utf-8")).hexdigest()
        return self.lease(f"{kind}--{digest}")

    def cost_reservation(self, attempt_id: str, provider: str) -> Path:
        """Resolve one attempt/provider pair to a durable reservation shard."""
        validate_id(attempt_id)
        if type(provider) is not str or not provider:
            raise ValueError("provider")
        digest = hashlib.sha256(provider.encode("utf-8")).hexdigest()
        return self.root / "cost-reservations" / f"{attempt_id}--{digest}.json"

    def telemetry_event(
        self, session_id: str, sequence: int, event: str
    ) -> Path:
        """Resolve one immutable controller telemetry event slot."""
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


def new_index() -> dict:
    """Return an empty schema-v6 coordinator index."""
    return {
        "version": 6,
        "max_runnable_attempts": 20,
        "attempts": {},
        "history": {},
        "rejections": [],
        "snapshots": {},
        "resource_limits": {
            "metered_api_reserved_usd": 10.0,
            "publication_per_provider": 20,
        },
        "total_api_cost_usd": 0.0,
    }


def _require_dict(value: object, field: str) -> dict:
    if type(value) is not dict:
        raise ValueError(field)
    return value


def _require_nonempty_string(value: object, field: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(field)


def _require_nonnegative_number(value: object, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(field)


def _validate_reference_map(value: object, field: str) -> None:
    references = _require_dict(value, field)
    for identifier, reference in references.items():
        validate_id(identifier)
        reference = _require_dict(reference, field)
        if set(reference) != ATTEMPT_REFERENCE_KEYS:
            raise ValueError(field)
        for key in ATTEMPT_REFERENCE_KEYS:
            _require_nonempty_string(reference[key], field)


def validate_index(index: dict) -> None:
    """Validate the exact schema-v6 index and its attempt references."""
    index = _require_dict(index, "index")
    if set(index) != INDEX_KEYS:
        raise ValueError("keys")
    if index["version"] != 6:
        raise ValueError("version")
    if index["max_runnable_attempts"] != 20:
        raise ValueError("max_runnable_attempts")
    _validate_reference_map(index["attempts"], "attempts")
    _validate_reference_map(index["history"], "history")
    if type(index["rejections"]) is not list:
        raise ValueError("rejections")
    if type(index["snapshots"]) is not dict:
        raise ValueError("snapshots")
    for snapshot_id, reference in index["snapshots"].items():
        validate_id(snapshot_id)
        _require_nonempty_string(reference, "snapshots")
    limits = _require_dict(index["resource_limits"], "resource_limits")
    if set(limits) != RESOURCE_LIMIT_KEYS:
        raise ValueError("resource_limits")
    _require_nonnegative_number(
        limits["metered_api_reserved_usd"], "metered_api_reserved_usd"
    )
    if (
        type(limits["publication_per_provider"]) is not int
        or limits["publication_per_provider"] < 1
    ):
        raise ValueError("publication_per_provider")
    _require_nonnegative_number(index["total_api_cost_usd"], "total_api_cost_usd")


def _validate_timestamped_shard(
    value: dict, identity_field: str, kind: str
) -> None:
    value = _require_dict(value, kind)
    _require_nonempty_string(value.get(identity_field), identity_field)
    validate_id(value[identity_field])
    timestamp_fields = {
        "updated_at",
        "created_at",
        "checked_at",
        "fetched_at",
        "refreshed_at",
    }
    if not any(
        type(value.get(field)) is str and bool(value[field])
        for field in timestamp_fields
    ):
        raise ValueError("timestamp")


def validate_attempt(attempt: dict) -> None:
    """Validate the stable identity fields of an attempt shard."""
    _validate_timestamped_shard(attempt, "attempt_id", "attempt")
    _require_nonempty_string(attempt.get("paper_id"), "paper_id")
    _require_nonempty_string(attempt.get("phase"), "phase")
    improvement_attempts = attempt.get("improvement_attempts", 0)
    if (
        type(improvement_attempts) is not int
        or improvement_attempts not in {0, 1}
    ):
        raise ValueError("improvement_attempts")
    if improvement_attempts == 0:
        if "improvement_reason" in attempt:
            raise ValueError("improvement_reason")
    elif (
        type(attempt.get("improvement_reason")) is not str
        or not attempt["improvement_reason"].strip()
    ):
        raise ValueError("improvement_reason")


def validate_judgment(judgment: dict) -> None:
    """Validate the stable identity fields of a judgment shard."""
    _validate_timestamped_shard(judgment, "attempt_id", "judgment")


def validate_snapshot(snapshot: dict) -> None:
    """Validate the stable identity fields of a catalog snapshot shard."""
    _validate_timestamped_shard(snapshot, "snapshot_id", "snapshot")


def read_json(path: Path) -> dict:
    """Read a JSON object from path."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    return _require_dict(value, "json")


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(value, file, allow_nan=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_json_write(path: Path, value: dict, validator: Validator) -> None:
    """Validate and atomically replace a JSON shard under an exclusive lock."""
    validator(value)
    with _exclusive_lock(path):
        _atomic_json_write(path, value)


@contextmanager
def locked_json(path: Path, validator: Validator) -> Iterator[dict]:
    """Lock, read, and atomically persist a mutable JSON object."""
    with _exclusive_lock(path):
        value = read_json(path)
        yield value
        validator(value)
        _atomic_json_write(path, value)


def commit_json_transaction(
    transaction_path: Path,
    root: Path,
    targets: list[tuple[Path, dict, Validator]],
) -> None:
    """Durably record and install an ordered set of JSON replacements."""
    manifest_targets = []
    validators = {}
    for path, value, validator in targets:
        relative = _relative_transaction_target(path, root)
        validator(value)
        manifest_targets.append({"path": relative, "value": value})
        validators[relative] = validator
    expected = {"status": "planned", "targets": manifest_targets}
    with _exclusive_lock(transaction_path):
        if transaction_path.exists():
            manifest = read_json(transaction_path)
            _validate_transaction(manifest)
            if manifest["targets"] != expected["targets"]:
                raise ValueError("transaction")
        else:
            manifest = expected
            _transaction_write(transaction_path, manifest, _validate_transaction)
        _finish_json_transaction(transaction_path, root, manifest, validators)


def recover_json_transactions(
    directory: Path,
    root: Path,
    validator_for_path: ValidatorForPath,
) -> None:
    """Replay every durable incomplete JSON transaction in a directory."""
    for transaction_path in sorted(directory.glob("*.json")):
        with _exclusive_lock(transaction_path):
            manifest = read_json(transaction_path)
            _validate_transaction(manifest)
            validators = {
                target["path"]: validator_for_path(root / target["path"])
                for target in manifest["targets"]
            }
            _finish_json_transaction(
                transaction_path, root, manifest, validators
            )


def _finish_json_transaction(
    transaction_path: Path,
    root: Path,
    manifest: dict,
    validators: dict[str, Validator],
) -> None:
    if manifest["status"] == "complete":
        return
    for target in manifest["targets"]:
        path = root / target["path"]
        validator = validators[target["path"]]
        _transaction_write(path, target["value"], validator)
    manifest["status"] = "complete"
    _transaction_write(transaction_path, manifest, _validate_transaction)


def _validate_transaction(manifest: dict) -> None:
    if type(manifest) is not dict or set(manifest) != TRANSACTION_KEYS:
        raise ValueError("transaction")
    if manifest["status"] not in {"planned", "complete"}:
        raise ValueError("transaction")
    if type(manifest["targets"]) is not list or not manifest["targets"]:
        raise ValueError("transaction")
    paths = []
    for target in manifest["targets"]:
        if type(target) is not dict or set(target) != TRANSACTION_TARGET_KEYS:
            raise ValueError("transaction")
        if type(target["path"]) is not str or type(target["value"]) is not dict:
            raise ValueError("transaction")
        path = Path(target["path"])
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("transaction")
        paths.append(target["path"])
    if len(paths) != len(set(paths)):
        raise ValueError("transaction")


def _relative_transaction_target(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("transaction") from error
    if not relative.parts or ".." in relative.parts:
        raise ValueError("transaction")
    return str(relative)


def _transaction_write(path: Path, value: dict, validator: Validator) -> None:
    validator(value)
    _atomic_json_write(path, value)
