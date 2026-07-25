"""Direct, transactional migration from schema-v3 to schema-v6 state."""

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from uuid import UUID, uuid5

import state as legacy_state
import store


ATTEMPT_NAMESPACE = UUID("b3f93d5c-2d22-4c66-9f70-b7c15ef4bb59")
MIGRATION_TIMESTAMP_PREFIX = "schema-v3-migration:"
MANIFEST_KEYS = {"source_sha256", "status", "targets"}
MANIFEST_TARGET_KEYS = {"target", "sha256", "staging"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def attempt_id(paper_id: str, work_kind: str, attempt_number: int) -> str:
    """Return a stable identifier for a legacy paper attempt."""
    return str(uuid5(ATTEMPT_NAMESPACE, f"{paper_id}:{work_kind}:{attempt_number}"))


def _json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _sha256(value: dict) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _reference(paths: store.StatePaths, shard: dict) -> dict:
    path = paths.attempt(shard["attempt_id"])
    return {
        "path": str(path.relative_to(paths.index.parent)),
        "paper_id": shard["paper_id"],
        "phase": shard["phase"],
        "updated_at": shard["updated_at"],
    }


def effective_archived_phase(record: dict) -> str:
    """Return the phase represented by an archived legacy attempt."""
    return record.get("blocked_from", "complete")


def _attempt_shard(
    record: dict,
    phase: str,
    work_kind: str,
    attempt_number: int,
    source_sha256: str,
) -> dict:
    shard = copy.deepcopy(record)
    shard["attempt_id"] = attempt_id(shard["paper_id"], work_kind, attempt_number)
    shard["phase"] = phase
    shard["updated_at"] = f"{MIGRATION_TIMESTAMP_PREFIX}{source_sha256}"
    return shard


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    """Deterministic target documents for one validated schema-v3 source."""

    source_sha256: str
    source: dict
    index: dict
    targets: dict[Path, dict]

    @classmethod
    def from_v3(
        cls, v3: dict, max_runnable_attempts: int = 20
    ) -> "MigrationPlan":
        source = copy.deepcopy(v3)
        source_sha256 = _sha256(source)
        paths = store.StatePaths(Path("repro-loop.json"))
        index = store.new_index()
        index["max_runnable_attempts"] = max_runnable_attempts
        index["rejections"] = copy.deepcopy(source["rejections"])
        index["total_api_cost_usd"] = source["total_api_cost_usd"]
        shards = []
        if source["current"] is not None:
            shards.append(
                (
                    "attempts",
                    _attempt_shard(
                        source["current"],
                        source["phase"],
                        "active",
                        1,
                        source_sha256,
                    ),
                )
            )
        for number, record in enumerate(source["history"], start=1):
            shards.append(
                (
                    "history",
                    _attempt_shard(
                        record,
                        effective_archived_phase(record),
                        "history",
                        number,
                        source_sha256,
                    ),
                )
            )
        targets = {}
        for section, shard in shards:
            shard_path = paths.attempt(shard["attempt_id"])
            targets[shard_path] = shard
            index[section][shard["attempt_id"]] = _reference(paths, shard)
        store.validate_index(index)
        return cls(source_sha256, source, index, targets)

    def resolved_targets(self, paths: store.StatePaths) -> dict[Path, dict]:
        """Resolve plan-relative targets for a caller's state location."""
        resolved = {
            paths.index.parent / relative_path: value
            for relative_path, value in self.targets.items()
        }
        backup = paths.root / "v3-backups" / f"{self.source_sha256}.json"
        resolved[backup] = self.source
        resolved[paths.index] = self.index
        return resolved


def plan_v6_migration(v3: dict) -> MigrationPlan:
    """Validate schema-v3 input and produce deterministic schema-v6 targets."""
    legacy_state.validate_state(v3)
    return MigrationPlan.from_v3(v3, max_runnable_attempts=20)


def _validate_manifest(manifest: dict) -> None:
    if type(manifest) is not dict or set(manifest) != MANIFEST_KEYS:
        raise ValueError("manifest")
    if (
        type(manifest["source_sha256"]) is not str
        or SHA256_PATTERN.fullmatch(manifest["source_sha256"]) is None
        or manifest["status"] not in {"planned", "staged", "complete"}
        or type(manifest["targets"]) is not list
    ):
        raise ValueError("manifest")
    for target in manifest["targets"]:
        if (
            type(target) is not dict
            or set(target) != MANIFEST_TARGET_KEYS
            or any(
                type(target[field]) is not str or not target[field]
                for field in target
            )
        ):
            raise ValueError("manifest")
        if SHA256_PATTERN.fullmatch(target["sha256"]) is None:
            raise ValueError("manifest")
        for field in ("target", "staging"):
            path = Path(target[field])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("manifest")


def _validator_for(path: Path, paths: store.StatePaths, plan: MigrationPlan):
    if path == paths.index:
        return store.validate_index
    if path == paths.root / "v3-backups" / f"{plan.source_sha256}.json":
        return legacy_state.validate_state
    return store.validate_attempt


def _write_json(path: Path, value: dict, validator) -> None:
    store.atomic_json_write(path, value, validator)


def _manifest_path(paths: store.StatePaths, source_sha256: str) -> Path:
    return paths.root / "transactions" / f"{source_sha256}.json"


def _manifest(paths: store.StatePaths, plan: MigrationPlan) -> dict:
    targets = plan.resolved_targets(paths)
    transaction_root = paths.root / "transactions" / plan.source_sha256
    entries = []
    for number, (target, value) in enumerate(
        sorted(targets.items(), key=lambda item: str(item[0]))
    ):
        entries.append(
            {
                "target": str(target.relative_to(paths.index.parent)),
                "sha256": _sha256(value),
                "staging": str(
                    (transaction_root / "staging" / f"{number}.json").relative_to(
                        paths.index.parent
                    )
                ),
            }
        )
    return {
        "source_sha256": plan.source_sha256,
        "status": "planned",
        "targets": entries,
    }


def _authenticate_manifest(
    paths: store.StatePaths, plan: MigrationPlan, manifest: dict
) -> None:
    """Authenticate a manifest against the only deterministic plan allowed."""
    _validate_manifest(manifest)
    expected = _manifest(paths, plan)
    if (
        manifest["source_sha256"] != expected["source_sha256"]
        or manifest["targets"] != expected["targets"]
    ):
        raise ValueError("manifest")


def _matches(path: Path, expected_sha256: str) -> bool:
    return (
        path.exists()
        and hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
    )


def _finish_transaction(
    paths: store.StatePaths, plan: MigrationPlan, manifest: dict
) -> None:
    targets = plan.resolved_targets(paths)
    by_target = {
        str(path.relative_to(paths.index.parent)): (path, value)
        for path, value in targets.items()
    }
    for entry in manifest["targets"]:
        target, value = by_target[entry["target"]]
        staging = paths.index.parent / entry["staging"]
        validator = _validator_for(target, paths, plan)
        if not _matches(staging, entry["sha256"]):
            _write_json(staging, value, validator)
            if not _matches(staging, entry["sha256"]):
                raise ValueError("staging hash")
    if manifest["status"] == "planned":
        manifest["status"] = "staged"
        _write_json(
            _manifest_path(paths, plan.source_sha256),
            manifest,
            _validate_manifest,
        )
    install_order = sorted(
        manifest["targets"], key=lambda entry: entry["target"] == paths.index.name
    )
    for entry in install_order:
        target, value = by_target[entry["target"]]
        if not _matches(target, entry["sha256"]):
            _write_json(target, value, _validator_for(target, paths, plan))
            if not _matches(target, entry["sha256"]):
                raise ValueError("target hash")
    if manifest["status"] != "complete":
        manifest["status"] = "complete"
        _write_json(
            _manifest_path(paths, plan.source_sha256),
            manifest,
            _validate_manifest,
        )


def apply_v6_migration(paths: store.StatePaths, plan: MigrationPlan) -> dict:
    """Apply a write-ahead migration, installing the index only after shards."""
    manifest_path = _manifest_path(paths, plan.source_sha256)
    if manifest_path.exists():
        manifest = store.read_json(manifest_path)
        _authenticate_manifest(paths, plan, manifest)
    else:
        manifest = _manifest(paths, plan)
        _write_json(manifest_path, manifest, _validate_manifest)
    if manifest["status"] == "complete":
        for target, value in plan.resolved_targets(paths).items():
            if not _matches(target, _sha256(value)):
                raise ValueError("completed target hash")
        return plan.index
    _finish_transaction(paths, plan, manifest)
    return plan.index


def recover_transactions(paths: store.StatePaths) -> None:
    """Resume every incomplete transaction, including a pre-manifest crash."""
    if paths.index.exists():
        current = store.read_json(paths.index)
        if current.get("version") == legacy_state.SCHEMA_V3_VERSION:
            plan = plan_v6_migration(current)
            apply_v6_migration(paths, plan)
            return
    transaction_dir = paths.root / "transactions"
    if not transaction_dir.exists():
        return
    for manifest_path in sorted(transaction_dir.glob("*.json")):
        manifest = store.read_json(manifest_path)
        _validate_manifest(manifest)
        backup = (
            paths.root
            / "v3-backups"
            / f"{manifest['source_sha256']}.json"
        )
        if not backup.exists():
            raise ValueError("missing migration source")
        plan = plan_v6_migration(store.read_json(backup))
        _authenticate_manifest(paths, plan, manifest)
        apply_v6_migration(paths, plan)


def plan_for_existing_migration(paths: store.StatePaths) -> MigrationPlan:
    """Load the deterministic plan from either v3 state or its v6 backup."""
    current = store.read_json(paths.index)
    if current.get("version") == legacy_state.SCHEMA_V3_VERSION:
        return plan_v6_migration(current)
    store.validate_index(current)
    transaction_dir = paths.root / "transactions"
    manifests = sorted(transaction_dir.glob("*.json"))
    if len(manifests) != 1:
        raise ValueError("migration manifest")
    manifest = store.read_json(manifests[0])
    _validate_manifest(manifest)
    backup = (
        paths.root / "v3-backups" / f"{manifest['source_sha256']}.json"
    )
    if not backup.exists():
        raise ValueError("missing migration source")
    plan = plan_v6_migration(store.read_json(backup))
    _authenticate_manifest(paths, plan, manifest)
    return plan


def _legacy_fields(shard: dict) -> dict:
    ignored = {"attempt_id", "phase", "updated_at"}
    return {key: value for key, value in shard.items() if key not in ignored}


def verify_semantic_equivalence(v3: dict, paths: store.StatePaths) -> dict:
    """Verify that every legacy record and coordinator value was preserved."""
    legacy_state.validate_state(v3)
    index = store.read_json(paths.index)
    store.validate_index(index)
    active = [
        store.read_json(paths.index.parent / reference["path"])
        for reference in index["attempts"].values()
    ]
    archived = [
        store.read_json(paths.index.parent / reference["path"])
        for reference in index["history"].values()
    ]
    expected_active = [] if v3["current"] is None else [v3["current"]]
    if [_legacy_fields(shard) for shard in active] != expected_active:
        raise ValueError("active attempts")
    if [_legacy_fields(shard) for shard in archived] != v3["history"]:
        raise ValueError("archived attempts")
    if index["rejections"] != v3["rejections"]:
        raise ValueError("rejections")
    if index["total_api_cost_usd"] != v3["total_api_cost_usd"]:
        raise ValueError("total_api_cost_usd")
    return {
        "active_attempts": len(active),
        "archived_attempts": len(archived),
        "rejections": len(index["rejections"]),
        "max_runnable_attempts": index["max_runnable_attempts"],
        "total_api_cost_usd": index["total_api_cost_usd"],
    }
