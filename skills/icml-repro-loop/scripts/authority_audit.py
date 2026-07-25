"""Audit and quarantine unsupported schema-v6 completion authority."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from uuid import uuid4


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import attestations  # noqa: E402
import leases  # noqa: E402
import refresh  # noqa: E402
import store  # noqa: E402


REPORT_VERSION = 1
VERDICT_ALIAS_FIELDS = {
    "verdict",
    "verdicts",
    "verdict_at",
    "verdict_source_revision",
}


def audit(paths: store.StatePaths, snapshot_id: str) -> dict:
    """Classify every local completion against one immutable live snapshot."""
    snapshot = refresh.read_snapshot(paths, snapshot_id)
    index_bytes = paths.index.read_bytes()
    index = json.loads(index_bytes)
    store.validate_index(index)
    snapshot_path = _snapshot_path(paths, snapshot_id, index)
    decisions = []
    seen_attempt_ids = set()
    for section in ("attempts", "history"):
        for attempt_id, reference in sorted(index[section].items()):
            if reference["phase"] != "complete":
                continue
            if attempt_id in seen_attempt_ids:
                raise ValueError("attempt_id")
            seen_attempt_ids.add(attempt_id)
            attempt_path = paths.index.parent / reference["path"]
            if attempt_path != paths.attempt(attempt_id):
                raise ValueError("attempt")
            attempt_bytes = attempt_path.read_bytes()
            attempt = json.loads(attempt_bytes)
            store.validate_attempt(attempt)
            if (
                attempt["attempt_id"] != attempt_id
                or attempt["paper_id"] != reference["paper_id"]
                or attempt["phase"] != "complete"
            ):
                raise ValueError("attempt")
            files = _attempt_source_files(paths, attempt_id)
            decision = {
                "attempt_id": attempt_id,
                "paper_id": attempt["paper_id"],
                "source_section": section,
                "attempt_path": str(
                    attempt_path.relative_to(paths.index.parent)
                ),
                "attempt_sha256": hashlib.sha256(attempt_bytes).hexdigest(),
                "files": [
                    {
                        "source_path": str(
                            path.relative_to(paths.index.parent)
                        ),
                        "sha256": _file_sha256(path),
                    }
                    for path in files
                ],
            }
            if _has_exact_official_verdict(snapshot, attempt):
                decision.update(
                    {
                        "classification": "valid-official",
                        "blocked_from": None,
                        "reasons": [],
                    }
                )
            else:
                blocked_from, reasons = _last_proven_phase(
                    paths, snapshot, attempt
                )
                decision.update(
                    {
                        "classification": "unsupported-completion",
                        "blocked_from": blocked_from,
                        "reasons": reasons,
                    }
                )
            decisions.append(decision)
    payload = {
        "version": REPORT_VERSION,
        "snapshot_id": snapshot_id,
        "snapshot_sha256": _file_sha256(snapshot_path),
        "index_path": str(paths.index.relative_to(paths.index.parent)),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "decisions": decisions,
    }
    return {"report_id": _sha256_json(payload), **payload}


def repair(
    paths: store.StatePaths,
    report: dict,
    now: datetime,
) -> dict:
    """Quarantine unsupported completion bytes and restore blocked attempts."""
    _validate_report(report)
    observed_at = _timestamp(now)
    invalid = [
        decision
        for decision in report["decisions"]
        if decision["classification"] == "unsupported-completion"
    ]
    index = store.read_json(paths.index)
    store.validate_index(index)
    already_repaired = {
        decision["attempt_id"]
        for decision in invalid
        if _is_repaired_attempt(paths, decision, report["report_id"])
    }
    current_index_sha256 = _file_sha256(paths.index)
    invalid_ids = {decision["attempt_id"] for decision in invalid}
    if current_index_sha256 == report["index_sha256"]:
        if audit(paths, report["snapshot_id"]) != report:
            raise ValueError("report")
    elif invalid_ids and already_repaired == invalid_ids:
        report_path = paths.authority_audit(report["report_id"])
        if (
            not report_path.exists()
            or store.read_json(report_path) != report
        ):
            raise ValueError("report")
    else:
        raise ValueError("index_sha256")

    manifests = {}
    for decision in invalid:
        attempt_id = decision["attempt_id"]
        if attempt_id in already_repaired:
            manifest = store.read_json(paths.quarantine_manifest(attempt_id))
            _validate_quarantine_manifest(manifest)
            if manifest["report_id"] != report["report_id"]:
                raise ValueError("quarantine")
            manifests[attempt_id] = manifest
            continue
        manifests[attempt_id] = _quarantine_manifest(paths, report, decision)
    _preflight_quarantine(paths, manifests)

    report_path = paths.authority_audit(report["report_id"])
    _write_json_once(report_path, report, _validate_report, "report")
    for attempt_id, manifest in sorted(manifests.items()):
        if attempt_id in already_repaired:
            continue
        for entry in manifest["files"]:
            source = paths.index.parent / entry["source_path"]
            target = paths.index.parent / entry["copy_path"]
            _write_bytes_once(target, source.read_bytes())
        _write_json_once(
            paths.quarantine_manifest(attempt_id),
            manifest,
            _validate_quarantine_manifest,
            "quarantine",
        )

    updated_index = copy.deepcopy(index)
    targets = [
        target
        for attempt_id in sorted(
            decision["attempt_id"] for decision in invalid
        )
        if attempt_id not in already_repaired
        and (
            target := _released_lease_target(
                paths,
                attempt_id,
                observed_at,
            )
        )
        is not None
    ]
    mutations = 0
    decisions = {decision["attempt_id"]: decision for decision in invalid}
    for attempt_id in sorted(decisions):
        if attempt_id in already_repaired:
            continue
        decision = decisions[attempt_id]
        attempt = store.read_json(paths.attempt(attempt_id))
        store.validate_attempt(attempt)
        if attempt["phase"] != "complete":
            raise ValueError("phase")
        for field in VERDICT_ALIAS_FIELDS:
            attempt.pop(field, None)
        attempt["phase"] = "blocked"
        attempt["blocked_from"] = decision["blocked_from"]
        attempt["blocker"] = (
            "Authority audit found unsupported completion: "
            + "; ".join(decision["reasons"])
        )
        attempt["authority_repair"] = {
            "report_id": report["report_id"],
            "snapshot_id": report["snapshot_id"],
            "repaired_at": observed_at,
            "requires_fresh_lease": True,
        }
        attempt.setdefault("transitions", []).append(
            {
                "from": "complete",
                "to": "blocked",
                "at": observed_at,
                "owner": "authority-audit",
                "fencing_token": 0,
                "snapshot_id": report["snapshot_id"],
                "report_id": report["report_id"],
            }
        )
        attempt["updated_at"] = observed_at
        store.validate_attempt(attempt)
        updated_index["history"].pop(attempt_id, None)
        if (
            attempt_id in updated_index["attempts"]
            and updated_index["attempts"][attempt_id]["phase"] != "complete"
        ):
            raise ValueError("attempt_id")
        updated_index["attempts"][attempt_id] = _attempt_reference(
            paths, attempt
        )
        targets.append(
            (paths.attempt(attempt_id), attempt, store.validate_attempt)
        )
        mutations += 1
    if mutations:
        targets.append((paths.index, updated_index, store.validate_index))
        transaction_path = (
            paths.root
            / "transactions"
            / "authority-audit"
            / f"{uuid4()}.json"
        )
        store.commit_json_transaction(
            transaction_path,
            paths.index.parent,
            targets,
        )
    return {
        "report_id": report["report_id"],
        "mutations": mutations,
        "quarantined_attempt_ids": sorted(decisions),
    }


def recover_transactions(paths: store.StatePaths) -> None:
    """Finish any interrupted authority-repair transaction."""
    store.recover_json_transactions(
        paths.root / "transactions" / "authority-audit",
        paths.index.parent,
        lambda path: _repair_validator(paths, path),
    )


def reusable_repair_report(paths: store.StatePaths, fresh: dict) -> dict:
    """Reuse the exact persisted report when its repair is already installed."""
    _validate_report(fresh)
    fresh_decisions = {
        decision["attempt_id"]: decision
        for decision in fresh["decisions"]
    }
    matches = []
    for path in sorted((paths.root / "authority-audits").glob("*.json")):
        candidate = store.read_json(path)
        _validate_report(candidate)
        if (
            candidate["snapshot_id"] != fresh["snapshot_id"]
            or path != paths.authority_audit(candidate["report_id"])
        ):
            continue
        unsupported = [
            decision
            for decision in candidate["decisions"]
            if decision["classification"] == "unsupported-completion"
        ]
        valid = {
            decision["attempt_id"]: decision
            for decision in candidate["decisions"]
            if decision["classification"] == "valid-official"
        }
        if (
            unsupported
            and fresh_decisions == valid
            and all(
                _is_repaired_attempt(
                    paths,
                    decision,
                    candidate["report_id"],
                )
                for decision in unsupported
            )
        ):
            matches.append(candidate)
    if len(matches) > 1:
        raise ValueError("report")
    return fresh if not matches else matches[0]


def _has_exact_official_verdict(snapshot: dict, attempt: dict) -> bool:
    paper_id = attempt.get("paper_id")
    space_id = attempt.get("space_id")
    deployed_sha = attempt.get("deployed_sha")
    if not all(
        type(value) is str and bool(value)
        for value in (paper_id, space_id, deployed_sha)
    ):
        return False
    verdict_revision = _verdict_revision(snapshot)
    exact = [
        verdict
        for verdict in snapshot["verdicts"]
        if verdict.get("paper_id") == paper_id
        and verdict.get("space_id") == space_id
        and verdict.get("sha") == deployed_sha
        and verdict.get("source_revision") == verdict_revision
    ]
    return len(exact) == 1


def _last_proven_phase(
    paths: store.StatePaths,
    snapshot: dict,
    attempt: dict,
) -> tuple[str, list[str]]:
    reasons = ["no exact official verdict for the paper, Space, and SHA"]
    if _snapshot_observes_submission(snapshot, attempt):
        reasons.append("exact tagged Space is live but has no exact verdict")
        return "judging", reasons
    if _valid_deployment_attestation(paths, attempt):
        reasons.append("deployment is attested but submission is unproven")
        return "deployed", reasons
    if _valid_phase_attestation(paths, attempt, "validation", "validated"):
        reasons.append("validation is attested but deployment is unproven")
        return "validated", reasons
    reasons.append("no controller validation attestation exists")
    return "implementing", reasons


def _snapshot_observes_submission(snapshot: dict, attempt: dict) -> bool:
    paper_id = attempt.get("paper_id")
    space_id = attempt.get("space_id")
    deployed_sha = attempt.get("deployed_sha")
    spaces = [
        space
        for space in snapshot.get("spaces", [])
        if space.get("space_id") == space_id
        and space.get("revision") == deployed_sha
        and space.get("paper_ids") == [paper_id]
    ]
    tagged = [
        record
        for record in snapshot.get("tagged_spaces", [])
        if record.get("paper_id") == paper_id
        and record.get("space_id") == space_id
        and record.get("revision") == deployed_sha
    ]
    return len(spaces) == 1 and len(tagged) == 1


def _valid_deployment_attestation(
    paths: store.StatePaths, attempt: dict
) -> bool:
    record = _phase_attestation(paths, attempt, "deployment", "deployed")
    return bool(
        record is not None
        and record.get("space_id") == attempt.get("space_id")
        and record.get("space_sha") == attempt.get("deployed_sha")
        and record.get("runtime_stage") == "RUNNING"
        and _valid_phase_attestation(
            paths, attempt, "validation", "validated"
        )
    )


def _valid_phase_attestation(
    paths: store.StatePaths,
    attempt: dict,
    kind: str,
    phase: str,
) -> bool:
    return _phase_attestation(paths, attempt, kind, phase) is not None


def _phase_attestation(
    paths: store.StatePaths,
    attempt: dict,
    kind: str,
    phase: str,
) -> dict | None:
    attempt_number = attempt.get("improvement_attempts", 0) + 1
    if type(attempt_number) is not int or attempt_number < 1:
        return None
    path = paths.attestation(kind, attempt["attempt_id"], attempt_number)
    if not path.exists():
        return None
    try:
        record = store.read_json(path)
        attestations.validate_target(paths, path, record)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    transitions = [
        transition
        for transition in attempt.get("transitions", [])
        if transition.get("to") == phase
        and transition.get("attestation_id") == record["attestation_id"]
    ]
    if (
        record["kind"] != kind
        or record["attempt_id"] != attempt["attempt_id"]
        or record["attempt_number"] != attempt_number
        or len(transitions) != 1
    ):
        return None
    return record


def _attempt_source_files(
    paths: store.StatePaths, attempt_id: str
) -> list[Path]:
    files = [paths.attempt(attempt_id)]
    judgment = paths.judgment(attempt_id)
    if judgment.exists():
        files.append(judgment)
    files.extend(
        sorted(
            (paths.root / "judgments" / "archive").glob(
                f"{attempt_id}--*.json"
            )
        )
    )
    return files


def _quarantine_manifest(
    paths: store.StatePaths,
    report: dict,
    decision: dict,
) -> dict:
    attempt_id = decision["attempt_id"]
    entries = []
    for source in (
        paths.index,
        *(
            paths.index.parent / record["source_path"]
            for record in decision["files"]
        ),
    ):
        copy_name = _quarantine_copy_name(paths, attempt_id, source)
        target = paths.quarantine(attempt_id) / copy_name
        entries.append(
            {
                "source_path": str(source.relative_to(paths.index.parent)),
                "copy_path": str(target.relative_to(paths.index.parent)),
                "sha256": _file_sha256(source),
            }
        )
    return {
        "version": 1,
        "attempt_id": attempt_id,
        "report_id": report["report_id"],
        "files": entries,
    }


def _quarantine_copy_name(
    paths: store.StatePaths, attempt_id: str, source: Path
) -> str:
    if source == paths.index:
        return "index.json"
    if source == paths.attempt(attempt_id):
        return "attempt.json"
    if source == paths.judgment(attempt_id):
        return "judgment.json"
    if source.parent == paths.root / "judgments" / "archive":
        return f"judgment-archive--{source.name}"
    raise ValueError("quarantine")


def _preflight_quarantine(
    paths: store.StatePaths,
    manifests: dict[str, dict],
) -> None:
    for manifest in manifests.values():
        _validate_quarantine_manifest(manifest)
        for entry in manifest["files"]:
            source = paths.index.parent / entry["source_path"]
            target = paths.index.parent / entry["copy_path"]
            source_bytes = source.read_bytes()
            if hashlib.sha256(source_bytes).hexdigest() != entry["sha256"]:
                existing = (
                    target.exists()
                    and hashlib.sha256(target.read_bytes()).hexdigest()
                    == entry["sha256"]
                )
                if not existing:
                    raise ValueError("source_sha256")
            if target.exists() and target.read_bytes() != source_bytes:
                if hashlib.sha256(target.read_bytes()).hexdigest() != entry[
                    "sha256"
                ]:
                    raise ValueError("quarantine")
        manifest_path = paths.quarantine_manifest(manifest["attempt_id"])
        if manifest_path.exists():
            existing = store.read_json(manifest_path)
            if existing != manifest:
                raise ValueError("quarantine")


def _is_repaired_attempt(
    paths: store.StatePaths,
    decision: dict,
    report_id: str,
) -> bool:
    path = paths.attempt(decision["attempt_id"])
    if not path.exists():
        return False
    attempt = store.read_json(path)
    repair_record = attempt.get("authority_repair")
    return bool(
        attempt.get("phase") == "blocked"
        and type(repair_record) is dict
        and repair_record.get("report_id") == report_id
    )


def _validate_report(report: dict) -> None:
    if type(report) is not dict or set(report) != {
        "report_id",
        "version",
        "snapshot_id",
        "snapshot_sha256",
        "index_path",
        "index_sha256",
        "decisions",
    }:
        raise ValueError("report")
    if report["version"] != REPORT_VERSION:
        raise ValueError("report")
    if report["report_id"] != _sha256_json(
        {key: value for key, value in report.items() if key != "report_id"}
    ):
        raise ValueError("report_id")
    for field in ("report_id", "snapshot_id", "snapshot_sha256", "index_sha256"):
        if (
            type(report[field]) is not str
            or len(report[field]) != 64
            or any(character not in "0123456789abcdef" for character in report[field])
        ):
            raise ValueError(field)
    if type(report["index_path"]) is not str or not report["index_path"]:
        raise ValueError("index_path")
    decisions = report["decisions"]
    if type(decisions) is not list:
        raise ValueError("decisions")
    attempt_ids = []
    for decision in decisions:
        if type(decision) is not dict:
            raise ValueError("decisions")
        attempt_ids.append(decision.get("attempt_id"))
        if decision.get("classification") not in {
            "valid-official",
            "unsupported-completion",
        }:
            raise ValueError("classification")
        if decision["classification"] == "unsupported-completion" and decision.get(
            "blocked_from"
        ) not in {"judging", "deployed", "validated", "implementing"}:
            raise ValueError("blocked_from")
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("attempt_id")


def _validate_quarantine_manifest(manifest: dict) -> None:
    if type(manifest) is not dict or set(manifest) != {
        "version",
        "attempt_id",
        "report_id",
        "files",
    }:
        raise ValueError("quarantine")
    if manifest["version"] != 1:
        raise ValueError("quarantine")
    store.validate_id(manifest["attempt_id"])
    if type(manifest["report_id"]) is not str:
        raise ValueError("quarantine")
    if type(manifest["files"]) is not list or not manifest["files"]:
        raise ValueError("quarantine")
    for entry in manifest["files"]:
        if type(entry) is not dict or set(entry) != {
            "source_path",
            "copy_path",
            "sha256",
        }:
            raise ValueError("quarantine")
        if any(
            type(entry[field]) is not str or not entry[field]
            for field in ("source_path", "copy_path", "sha256")
        ):
            raise ValueError("quarantine")


def _write_json_once(
    path: Path,
    value: dict,
    validator: store.Validator,
    field: str,
) -> None:
    validator(value)
    if path.exists():
        if store.read_json(path) != value:
            raise ValueError(field)
        return
    store.atomic_json_write(path, value, validator)


def _write_bytes_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("quarantine")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        store._fsync_directory(path.parent)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _snapshot_path(
    paths: store.StatePaths,
    snapshot_id: str,
    index: dict,
) -> Path:
    reference = index["snapshots"].get(snapshot_id)
    if reference is None:
        raise ValueError("snapshot_id")
    path = paths.index.parent / reference
    if path != paths.root / "snapshots" / f"{snapshot_id}.json":
        raise ValueError("snapshot")
    return path


def _verdict_revision(snapshot: dict) -> str:
    try:
        value = snapshot["sources"]["verdicts"]["revision"]
    except (KeyError, TypeError) as error:
        raise ValueError("verdict_revision") from error
    if type(value) is not str or not value:
        raise ValueError("verdict_revision")
    return value


def _attempt_reference(paths: store.StatePaths, attempt: dict) -> dict:
    return {
        "path": str(
            paths.attempt(attempt["attempt_id"]).relative_to(
                paths.index.parent
            )
        ),
        "paper_id": attempt["paper_id"],
        "phase": attempt["phase"],
        "updated_at": attempt["updated_at"],
    }


def _repair_validator(
    paths: store.StatePaths, path: Path
) -> store.Validator:
    if path == paths.index:
        return store.validate_index
    if path.parent == paths.root / "attempts" and path.suffix == ".json":
        return store.validate_attempt
    if path.parent == paths.root / "leases" and path.suffix == ".json":
        return leases.validate_lease
    raise ValueError("transaction")


def _released_lease_target(
    paths: store.StatePaths,
    attempt_id: str,
    observed_at: str,
) -> tuple[Path, dict, store.Validator] | None:
    resource = f"attempt:{attempt_id}"
    path = paths.resource_lease(resource)
    if not path.exists():
        return None
    value = store.read_json(path)
    leases.validate_lease(value)
    if value["resource"] != resource or value["attempt_id"] != attempt_id:
        raise ValueError("lease")
    if value["released_at"] is not None:
        return None
    released = copy.deepcopy(value)
    released["released_at"] = observed_at
    leases.validate_lease(released)
    return path, released, leases.validate_lease


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _timestamp(value: object) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError("now")
    return value.astimezone(timezone.utc).isoformat()
