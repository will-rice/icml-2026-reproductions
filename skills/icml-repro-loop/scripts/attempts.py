"""Independent fenced lifecycles for schema-v6 reproduction attempts."""

from __future__ import annotations

from collections.abc import Callable
import copy
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path, PurePosixPath
import re
import sys
from uuid import uuid4


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import attestations  # noqa: E402
import leases  # noqa: E402
import state  # noqa: E402
import store  # noqa: E402


IMMUTABLE_FIELDS = {
    "attempt_id",
    "claim_bindings",
    "legacy_reconciliation",
    "live_claims",
    "paper_id",
    "phase",
    "snapshot_id",
    "transitions",
    "updated_at",
}
DESIGN_FIELDS = {"design", "design_review"}
REVIEW_DECISIONS = {"approved", "rejected"}
ATTESTED_PHASE_KINDS = {
    "validated": "validation",
    "deployed": "deployment",
    "submitted": "submission",
    "judging": "authority-audit",
    "complete": "verdict",
}
GENERIC_PHASES = {"design-pending", "implementing", "improving", "blocked"}
ATTESTED_PROTECTED_UPDATE_FIELDS = {"improvement_attempts"}
Mutation = Callable[[dict, str], bool]


def create_attempt(
    paths: store.StatePaths,
    attempt_id: str,
    paper: dict,
    lease: leases.Lease,
    snapshot_id: str,
    now: datetime,
) -> dict:
    """Create one selected attempt under its authoritative writer fence."""
    recover_transactions(paths)
    store.validate_id(attempt_id)
    _identity(snapshot_id, "snapshot_id")
    if type(paper) is not dict:
        raise ValueError("paper")
    paper_id = _identity(paper.get("paper_id"), "paper_id")
    timestamp = _timestamp(now)
    _assert_attempt_fence(paths, attempt_id, lease, now)
    attempt = copy.deepcopy(paper)
    attempt.update(
        {
            "attempt_id": attempt_id,
            "paper_id": paper_id,
            "phase": "selected",
            "snapshot_id": snapshot_id,
            "transitions": [],
            "updated_at": timestamp,
        }
    )
    store.validate_attempt(attempt)

    with leases.hold_fence(paths, lease, now):
        with store._exclusive_lock(paths.index):
            index = store.read_json(paths.index)
            store.validate_index(index)
            if attempt_id in index["attempts"] or attempt_id in index["history"]:
                raise ValueError("attempt_id")
            if any(
                reference["paper_id"] == paper_id
                for section in ("attempts", "history")
                for reference in index[section].values()
            ) or any(
                type(rejection) is dict
                and rejection.get("paper_id") == paper_id
                for rejection in index["rejections"]
            ):
                raise ValueError("paper_id")
            runnable = sum(
                reference["phase"] in state.RUNNABLE_PHASES
                for reference in index["attempts"].values()
            )
            if runnable >= index["max_runnable_attempts"]:
                raise ValueError("max_runnable_attempts")
            updated_index = copy.deepcopy(index)
            updated_index["attempts"][attempt_id] = _reference(paths, attempt)
            _commit(paths, attempt, updated_index)
    return copy.deepcopy(attempt)


def read_attempt(paths: store.StatePaths, attempt_id: str) -> dict:
    """Read and validate one active or archived attempt shard."""
    recover_transactions(paths)
    attempt = store.read_json(paths.attempt(attempt_id))
    store.validate_attempt(attempt)
    if attempt["attempt_id"] != attempt_id:
        raise ValueError("attempt_id")
    return attempt


def update_attempt(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    now: datetime,
    **updates: object,
) -> dict:
    """Apply same-phase updates to one active attempt under its writer fence."""
    unsupported = set(updates) & (IMMUTABLE_FIELDS | DESIGN_FIELDS)
    if unsupported:
        raise ValueError(sorted(unsupported)[0])

    def update(attempt: dict, _timestamp: str) -> bool:
        attempt.update(copy.deepcopy(updates))
        return False

    return _mutate_attempt(paths, attempt_id, lease, now, update)


def transition_attempt(
    paths: store.StatePaths,
    attempt_id: str,
    phase: str,
    lease: leases.Lease,
    now: datetime,
    **updates: object,
) -> dict:
    """Advance a non-authoritative attempt edge and append provenance."""
    if phase in ATTESTED_PHASE_KINDS:
        raise ValueError("attestation")
    if phase not in GENERIC_PHASES and phase != "idle":
        raise ValueError("phase")
    unsupported = set(updates) & (IMMUTABLE_FIELDS | DESIGN_FIELDS)
    if unsupported:
        raise ValueError(sorted(unsupported)[0])
    transition_updates = copy.deepcopy(updates)

    def transition(attempt: dict, timestamp: str) -> bool:
        return _apply_transition(
            attempt, phase, lease, timestamp, transition_updates
        )

    return _mutate_attempt(paths, attempt_id, lease, now, transition)


def transition_attested(
    paths: store.StatePaths,
    attempt_id: str,
    phase: str,
    attestation_id: str,
    updates: dict,
    lease: leases.Lease,
    now: datetime,
    transaction_targets: list[tuple[Path, dict, store.Validator]] | None = None,
) -> dict:
    """Advance one authoritative phase using matching immutable evidence."""
    expected_kind = ATTESTED_PHASE_KINDS.get(phase)
    if expected_kind is None:
        raise ValueError("attestation")
    if type(updates) is not dict:
        raise ValueError("updates")
    unsupported = set(updates) & (
        IMMUTABLE_FIELDS | DESIGN_FIELDS | ATTESTED_PROTECTED_UPDATE_FIELDS
    )
    if unsupported:
        raise ValueError(sorted(unsupported)[0])
    record = attestations.read(paths, attestation_id)
    transition_updates = copy.deepcopy(updates)

    def transition(attempt: dict, timestamp: str) -> bool:
        improvement_attempts = attempt.get("improvement_attempts", 0)
        if (
            type(improvement_attempts) is not int
            or improvement_attempts not in {0, 1}
        ):
            raise ValueError("attestation")
        if (
            record["kind"] != expected_kind
            or record["attempt_id"] != attempt_id
            or record["attempt_number"] != improvement_attempts + 1
        ):
            raise ValueError("attestation")
        return _apply_transition(
            attempt,
            phase,
            lease,
            timestamp,
            transition_updates,
            attestation_id=attestation_id,
        )

    return _mutate_attempt(
        paths,
        attempt_id,
        lease,
        now,
        transition,
        attestation=record,
        transaction_targets=transaction_targets,
    )


def record_design(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    author: str,
    design_path: str,
    now: datetime,
) -> dict:
    """Record or revise the design for a design-pending attempt."""
    author = _identity(author, "author")
    design_path = _identity(design_path, "design_path")

    def record(attempt: dict, timestamp: str) -> bool:
        if attempt["phase"] != "design-pending":
            raise ValueError("phase")
        attempt["design"] = {
            "author": author,
            "path": design_path,
            "recorded_at": timestamp,
        }
        attempt["design_review"] = None
        return False

    return _mutate_attempt(paths, attempt_id, lease, now, record)


def record_design_review(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    reviewer: str,
    decision: str,
    now: datetime,
) -> dict:
    """Record an independent design decision and advance approved work."""
    reviewer = _identity(reviewer, "reviewer")
    if decision not in REVIEW_DECISIONS:
        raise ValueError("decision")

    def review(attempt: dict, timestamp: str) -> bool:
        if attempt["phase"] != "design-pending":
            raise ValueError("phase")
        design = attempt.get("design")
        if type(design) is not dict:
            raise ValueError("design")
        author = _identity(design.get("author"), "author")
        if reviewer == author:
            raise ValueError("reviewer")
        record = {
            "reviewer": reviewer,
            "decision": decision,
            "reviewed_at": timestamp,
        }
        if decision == "rejected":
            attempt["design_review"] = record
            return False
        return _apply_transition(
            attempt,
            "implementing",
            lease,
            timestamp,
            {},
            design_review=record,
        )

    return _mutate_attempt(paths, attempt_id, lease, now, review)


def reconcile_legacy_attempt(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    snapshot_id: str,
    *,
    design_author: str,
    design_path: str,
    reviewer: str,
    approval_ref: str,
    now: datetime,
) -> dict:
    """Bind one migrated active attempt to fresh claims and design provenance."""
    import migrate_v6
    import scheduler

    design_author = _identity(design_author, "design_author")
    reviewer = _identity(reviewer, "reviewer")
    approval_ref = _identity(approval_ref, "approval_ref")
    if reviewer == design_author:
        raise ValueError("reviewer")
    design_path = _relative_path(design_path, "design_path")
    snapshot = scheduler.read_fresh_snapshot(paths, snapshot_id, now)
    if snapshot.get("assessments") is None:
        raise ValueError("assessments")
    paper_id = _paper_id_for_attempt(paths, attempt_id)
    candidates = [
        candidate
        for candidate in scheduler.rank_eligible_candidates(snapshot)
        if candidate["paper_id"] == paper_id
    ]
    if len(candidates) != 1:
        raise ValueError("paper_id")
    candidate = candidates[0]
    if any(
        record.get("paper_id") == paper_id
        for field in ("queued_submissions", "tagged_spaces", "verdicts")
        for record in snapshot[field]
    ):
        raise ValueError("paper_id")

    def reconcile(attempt: dict, timestamp: str) -> bool:
        if attempt["phase"] != "implementing":
            raise ValueError("phase")
        marker = re.fullmatch(
            r"schema-v3-migration:([0-9a-f]{64})",
            attempt.get("updated_at", ""),
        )
        if marker is None or "legacy_reconciliation" in attempt:
            raise ValueError("legacy_reconciliation")
        source_state_sha256 = marker.group(1)
        backup = paths.root / "v3-backups" / f"{source_state_sha256}.json"
        if (
            not backup.is_file()
            or hashlib.sha256(backup.read_bytes()).hexdigest()
            != source_state_sha256
        ):
            raise ValueError("source_state_sha256")
        source = store.read_json(backup)
        migrate_v6.legacy_state.validate_state(source)
        legacy = source.get("current")
        if (
            type(legacy) is not dict
            or source["phase"] != attempt["phase"]
            or migrate_v6.attempt_id(
                attempt["paper_id"], "active", 1
            )
            != attempt_id
            or any(attempt.get(field) != value for field, value in legacy.items())
        ):
            raise ValueError("legacy_reconciliation")
        for field in (
            "paper_id",
            "title",
            "upstream_revision",
            "target_claims",
            "estimated_api_cost_usd",
        ):
            if candidate.get(field) != attempt.get(field):
                raise ValueError(field)
        if attempt.get("design_approved") is not True:
            raise ValueError("design_approved")
        attempt["snapshot_id"] = snapshot_id
        attempt["claim_bindings"] = copy.deepcopy(candidate["claim_bindings"])
        attempt["live_claims"] = copy.deepcopy(candidate["live_claims"])
        attempt["design"] = {
            "author": design_author,
            "path": design_path,
            "recorded_at": timestamp,
        }
        attempt["design_review"] = {
            "reviewer": reviewer,
            "decision": "approved",
            "reviewed_at": timestamp,
        }
        attempt["legacy_reconciliation"] = {
            "source_state_sha256": source_state_sha256,
            "snapshot_id": snapshot_id,
            "approval_ref": approval_ref,
            "reconciled_at": timestamp,
        }
        return False

    return _mutate_attempt(paths, attempt_id, lease, now, reconcile)


def runnable_attempt_ids(paths: store.StatePaths) -> list[str]:
    """Return active attempt IDs whose current phases consume scheduler lanes."""
    recover_transactions(paths)
    index = store.read_json(paths.index)
    store.validate_index(index)
    return [
        attempt_id
        for attempt_id, reference in sorted(index["attempts"].items())
        if reference["phase"] in state.RUNNABLE_PHASES
    ]


def recover_transactions(paths: store.StatePaths) -> None:
    """Replay interrupted attempt/index mutations before reading scheduler state."""
    with store._exclusive_lock(paths.index):
        store.recover_json_transactions(
            paths.root / "transactions" / "attempts",
            paths.index.parent,
            lambda path: _validator_for(paths, path),
        )


def _mutate_attempt(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    now: datetime,
    mutation: Mutation,
    attestation: dict | None = None,
    transaction_targets: list[tuple[Path, dict, store.Validator]] | None = None,
) -> dict:
    recover_transactions(paths)
    timestamp = _timestamp(now)
    _assert_attempt_fence(paths, attempt_id, lease, now)
    with leases.hold_fence(paths, lease, now):
        with store._exclusive_lock(paths.index):
            index = store.read_json(paths.index)
            store.validate_index(index)
            _require_active(index, attempt_id)
            attempt = store.read_json(paths.attempt(attempt_id))
            store.validate_attempt(attempt)
            _require_attempt(attempt, attempt_id)
            archive = mutation(attempt, timestamp)
            attempt["updated_at"] = timestamp
            store.validate_attempt(attempt)
            updated_index = copy.deepcopy(index)
            reference = _reference(paths, attempt)
            if archive:
                updated_index["attempts"].pop(attempt_id)
                updated_index["history"][attempt_id] = reference
            else:
                updated_index["attempts"][attempt_id] = reference
            _commit(
                paths,
                attempt,
                updated_index,
                attestation,
                transaction_targets,
            )
    return copy.deepcopy(attempt)


def _apply_transition(
    attempt: dict,
    phase: str,
    lease: leases.Lease,
    timestamp: str,
    updates: dict,
    design_review: dict | None = None,
    attestation_id: str | None = None,
) -> bool:
    source = attempt["phase"]
    abandon = updates.pop("abandon", None)
    is_resume = source == "blocked" and phase == attempt.get("blocked_from")
    is_abandon = source == "blocked" and phase == "idle"
    if (
        type(phase) is not str
        or phase not in state.PHASES
        or (
            phase not in state.ALLOWED[source]
            and not is_resume
            and not is_abandon
        )
    ):
        raise ValueError("phase")
    if is_abandon:
        if abandon is not True:
            raise ValueError("abandon")
    elif abandon is not None:
        raise ValueError("abandon")
    if phase == "blocked":
        if type(updates.get("blocker")) is not str or not updates["blocker"]:
            raise ValueError("blocker")
    elif "blocker" in updates:
        raise ValueError("blocker")
    if phase == "implementing" and not is_resume:
        if design_review is None:
            raise ValueError("design_review")
        attempt["design_review"] = copy.deepcopy(design_review)

    attempt.update(copy.deepcopy(updates))
    if phase == "blocked":
        attempt["blocked_from"] = source
    elif is_resume:
        attempt.pop("blocked_from")
        attempt.pop("blocker")
    attempt["phase"] = phase
    transition = {
        "from": source,
        "to": phase,
        "at": timestamp,
        "owner": lease.owner,
        "fencing_token": lease.fencing_token,
        "snapshot_id": attempt.get("snapshot_id"),
    }
    if attestation_id is not None:
        transition["attestation_id"] = attestation_id
    attempt.setdefault("transitions", []).append(transition)
    return phase == "complete" or is_abandon


def _commit(
    paths: store.StatePaths,
    attempt: dict,
    index: dict,
    attestation: dict | None = None,
    transaction_targets: list[tuple[Path, dict, store.Validator]] | None = None,
) -> None:
    targets = []
    if attestation is not None:
        attestation_path = paths.attestation(
            attestation["kind"],
            attestation["attempt_id"],
            attestation["attempt_number"],
        )
        targets.append(
            (
                attestation_path,
                attestation,
                lambda record: attestations.validate_target(
                    paths, attestation_path, record
                ),
            )
        )
    targets.extend(transaction_targets or [])
    targets.extend(
        [
            (paths.attempt(attempt["attempt_id"]), attempt, store.validate_attempt),
            (paths.index, index, store.validate_index),
        ]
    )
    transaction_path = (
        paths.root / "transactions" / "attempts" / f"{uuid4()}.json"
    )
    store.commit_json_transaction(
        transaction_path, paths.index.parent, targets
    )


def _validator_for(paths: store.StatePaths, path: Path) -> store.Validator:
    if path == paths.index:
        return store.validate_index
    if path.parent == paths.root / "attempts" and path.suffix == ".json":
        return store.validate_attempt
    if (
        path.parent.parent == paths.root / "attestations"
        and path.suffix == ".json"
    ):
        return lambda record: attestations.validate_target(paths, path, record)
    if (
        path.parent == paths.root / "judgments"
        and path.suffix == ".json"
    ) or (
        path.parent == paths.root / "judgments" / "archive"
        and path.suffix == ".json"
    ):
        import scheduler

        return scheduler.validate_judgment_record
    raise ValueError("transaction")


def _reference(paths: store.StatePaths, attempt: dict) -> dict:
    return {
        "path": str(paths.attempt(attempt["attempt_id"]).relative_to(paths.index.parent)),
        "paper_id": attempt["paper_id"],
        "phase": attempt["phase"],
        "updated_at": attempt["updated_at"],
    }


def _assert_attempt_fence(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    now: datetime,
) -> None:
    if lease.resource != f"attempt:{attempt_id}" or lease.attempt_id != attempt_id:
        raise leases.StaleFence(f"attempt:{attempt_id}")
    leases.assert_fence(paths, lease, now)


def _require_active(index: dict, attempt_id: str) -> None:
    if attempt_id not in index["attempts"]:
        raise ValueError("attempt_id")


def _require_attempt(attempt: dict, attempt_id: str) -> None:
    if attempt.get("attempt_id") != attempt_id:
        raise ValueError("attempt_id")


def _identity(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(field)
    return value


def _relative_path(value: object, field: str) -> str:
    value = _identity(value, field)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ValueError(field)
    return value


def _paper_id_for_attempt(paths: store.StatePaths, attempt_id: str) -> str:
    attempt = read_attempt(paths, attempt_id)
    return _identity(attempt.get("paper_id"), "paper_id")


def _timestamp(value: object) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError("now")
    return value.astimezone(timezone.utc).isoformat()
