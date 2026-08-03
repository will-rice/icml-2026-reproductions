"""Independent fenced lifecycles for schema-v6 reproduction attempts."""

from collections.abc import Callable
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
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
IMPROVEMENT_FIELDS = {"improvement_attempts", "improvement_reason"}
ATTESTED_PROTECTED_UPDATE_FIELDS = IMPROVEMENT_FIELDS
DEPLOYMENT_FIELDS = {"deployed_sha", "space_id"}
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
    unsupported = set(updates) & (
        IMMUTABLE_FIELDS | DESIGN_FIELDS | IMPROVEMENT_FIELDS
    )
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
        if attempt["phase"] == "judging" and phase == "improving":
            raise ValueError("attestation")
        if attempt["phase"] == "validated" and phase == "improving":
            _require_no_deployment(paths, attempt, transition_updates)
        if attempt["phase"] == "deployed" and phase == "improving":
            _validate_postverdict_correction(paths, attempt, transition_updates)
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
    attestation_record: dict | None = None,
) -> dict:
    """Advance one authoritative phase using matching immutable evidence."""
    expected_kind = (
        "verdict" if phase == "improving" else ATTESTED_PHASE_KINDS.get(phase)
    )
    if expected_kind is None:
        raise ValueError("attestation")
    if type(updates) is not dict:
        raise ValueError("updates")
    protected = (
        IMMUTABLE_FIELDS | DESIGN_FIELDS | ATTESTED_PROTECTED_UPDATE_FIELDS
    )
    if phase == "improving":
        protected -= IMPROVEMENT_FIELDS
    unsupported = set(updates) & protected
    if unsupported:
        raise ValueError(sorted(unsupported)[0])
    if attestation_record is None:
        record = attestations.read(paths, attestation_id)
    else:
        record = copy.deepcopy(attestation_record)
        attestations.validate_record(record)
        if record["attestation_id"] != attestation_id:
            raise ValueError("attestation_id")
    transition_updates = copy.deepcopy(updates)

    def transition(attempt: dict, timestamp: str) -> bool:
        improvement_attempts = attempt.get("improvement_attempts", 0)
        if (
            type(improvement_attempts) is not int
            or improvement_attempts < 0
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
    import refresh
    import scheduler

    design_author = _identity(design_author, "design_author")
    reviewer = _identity(reviewer, "reviewer")
    approval_ref = _identity(approval_ref, "approval_ref")
    if reviewer == design_author:
        raise ValueError("reviewer")
    design_path = _relative_path(design_path, "design_path")
    snapshot = scheduler.read_fresh_snapshot(paths, snapshot_id, now)
    paper_id = _paper_id_for_attempt(paths, attempt_id)
    assessment = refresh.assessment_record_for_snapshot(snapshot, paper_id)
    candidates = [
        candidate
        for candidate in scheduler.legacy_reconciliation_candidates(snapshot)
        if candidate["paper_id"] == paper_id
    ]
    if len(candidates) != 1:
        raise ValueError("paper_id")
    candidate = candidates[0]
    for field in refresh.ASSESSMENT_KEYS:
        if candidate.get(field) != assessment[field]:
            raise ValueError(field)
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
        design_approval = _resolve_design_approval(
            paths,
            design_path,
            approval_ref,
            attempt,
        )
        if design_author != design_approval["design_author"]:
            raise ValueError("design_author")
        if reviewer != design_approval["reviewer"]:
            raise ValueError("reviewer")
        attempt["snapshot_id"] = snapshot_id
        attempt["claim_bindings"] = copy.deepcopy(candidate["claim_bindings"])
        attempt["live_claims"] = copy.deepcopy(candidate["live_claims"])
        attempt["design"] = {
            "author": design_approval["design_author"],
            "approval_commit": design_approval["commit_sha"],
            "content_sha256": design_approval["content_sha256"],
            "paper_id": paper_id,
            "path": design_path,
            "recorded_at": timestamp,
        }
        attempt["design_review"] = {
            "reviewer": design_approval["reviewer"],
            "decision": "approved",
            "approval_ref": approval_ref,
            "design_content_sha256": design_approval["content_sha256"],
            "reviewed_at": timestamp,
        }
        attempt["legacy_reconciliation"] = {
            "source_state_sha256": source_state_sha256,
            "snapshot_id": snapshot_id,
            "approval_ref": approval_ref,
            "design_content_sha256": design_approval["content_sha256"],
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
    is_resume = source == "blocked" and phase == attempt.get("blocked_from")
    is_abandon = source == "blocked" and phase == "idle"
    is_predeployment_correction = source == "validated" and phase == "improving"
    is_postverdict_correction = source == "deployed" and phase == "improving"
    is_improving_entry = phase == "improving" and not is_resume
    if is_predeployment_correction:
        _validate_predeployment_correction(attempt, updates)
    elif is_improving_entry:
        _validate_improving_entry(attempt, updates)
    else:
        protected = set(updates) & IMPROVEMENT_FIELDS
        if protected:
            raise ValueError(sorted(protected)[0])
    abandon = updates.pop("abandon", None)
    if (
        type(phase) is not str
        or phase not in state.PHASES
        or (
            phase not in state.ALLOWED[source]
            and not is_resume
            and not is_abandon
            and not is_predeployment_correction
            and not is_postverdict_correction
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


def _validate_postverdict_correction(
    paths: store.StatePaths, attempt: dict, updates: dict
) -> None:
    """Allow deployed→improving only for an officially judged deployment.

    The official judge crawls tagged Spaces independently of local
    attestations, so a verdict can exist for a deployed revision whose
    submission was never locally attested. The correction requires an
    immutable snapshot proving the feed judged this attempt's exact
    space and deployed revision.
    """
    extra = set(updates) - IMPROVEMENT_FIELDS - {"verdict_snapshot_id"}
    if extra:
        raise ValueError(sorted(extra)[0])
    snapshot_id = updates.pop("verdict_snapshot_id", None)
    if type(snapshot_id) is not str or not snapshot_id:
        raise ValueError("verdict_snapshot_id")
    space_id = attempt.get("space_id")
    deployed_sha = attempt.get("deployed_sha")
    if (
        type(space_id) is not str
        or not space_id
        or type(deployed_sha) is not str
        or not deployed_sha
    ):
        raise ValueError("deployment")
    import refresh

    snapshot = refresh.read_snapshot(paths, snapshot_id)
    matches = [
        record
        for record in snapshot["verdicts"]
        if record.get("paper_id") == attempt["paper_id"]
        and record.get("space_id") == space_id
        and record.get("sha") == deployed_sha
    ]
    if len(matches) != 1:
        raise ValueError("verdict")


def _validate_predeployment_correction(attempt: dict, updates: dict) -> None:
    """Require the sole counted correction before an attempt is deployed."""
    if set(updates) - IMPROVEMENT_FIELDS:
        raise ValueError("updates")
    _validate_improving_entry(attempt, updates)


def _validate_improving_entry(attempt: dict, updates: dict) -> None:
    """Require each fresh correction to advance its sequence exactly once."""
    current_attempts = attempt.get("improvement_attempts", 0)
    if type(current_attempts) is not int or current_attempts < 0:
        raise ValueError("improvement_attempts")
    requested_attempts = updates.get("improvement_attempts")
    if (
        type(requested_attempts) is not int
        or requested_attempts != current_attempts + 1
    ):
        raise ValueError("improvement_attempts")
    reason = updates.get("improvement_reason")
    if type(reason) is not str or not reason.strip():
        raise ValueError("improvement_reason")


def _require_no_deployment(
    paths: store.StatePaths, attempt: dict, updates: dict
) -> None:
    """Reject correction when any authoritative deployment state exists."""
    if DEPLOYMENT_FIELDS & (set(attempt) | set(updates)):
        raise ValueError("deployment")
    if any(
        transition.get("to") in {"deployed", "submitted", "judging", "complete"}
        for transition in attempt.get("transitions", [])
    ):
        raise ValueError("deployment")
    deployment_root = paths.root / "attestations" / "deployment"
    for path in sorted(deployment_root.glob("*.json")):
        record = store.read_json(path)
        attestations.validate_target(paths, path, record)
        if record["attempt_id"] == attempt["attempt_id"]:
            raise ValueError("deployment")


def _commit(
    paths: store.StatePaths,
    attempt: dict,
    index: dict,
    attestation: dict | None = None,
    transaction_targets: list[tuple[Path, dict, store.Validator]] | None = None,
) -> None:
    targets = []
    if attestation is not None:
        object_path = attestations.object_path(
            paths, attestation["attestation_id"]
        )
        attestation_path = paths.attestation(
            attestation["kind"],
            attestation["attempt_id"],
            attestation["attempt_number"],
        )
        targets.append(
            (object_path, attestation, attestations.validate_record)
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
    if path.parent == paths.root / "attestation-objects" and path.suffix == ".json":
        return attestations.validate_record
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


def _resolve_design_approval(
    paths: store.StatePaths,
    design_path: str,
    approval_ref: str,
    attempt: dict,
) -> dict:
    match = re.fullmatch(r"git:([0-9a-f]{40})", approval_ref)
    if match is None:
        raise ValueError("approval_ref")
    commit_sha = match.group(1)
    try:
        repository_root = Path(
            _git_output(
                paths.index.parent,
                "rev-parse",
                "--show-toplevel",
            ).decode("utf-8").strip()
        ).resolve()
        state_path = paths.index.resolve().relative_to(repository_root).as_posix()
    except (UnicodeDecodeError, ValueError):
        raise ValueError("approval_ref") from None
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "merge-base",
            "--is-ancestor",
            commit_sha,
            "HEAD",
        ],
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("approval_ref")
    approved_content = _git_output(
        repository_root,
        "show",
        f"{commit_sha}:{design_path}",
    )
    tracked_content = _git_output(
        repository_root,
        "show",
        f"HEAD:{design_path}",
    )
    design_file = repository_root / design_path
    if (
        not design_file.is_file()
        or design_file.read_bytes() != tracked_content
        or tracked_content != approved_content
    ):
        raise ValueError("design_path")
    approval_state_bytes = _git_output(
        repository_root,
        "show",
        f"{commit_sha}:{state_path}",
    )
    revision = _git_output(
        repository_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        commit_sha,
    ).decode("utf-8").strip().split()
    if len(revision) != 2 or revision[0] != commit_sha:
        raise ValueError("approval_ref")
    parent_sha = revision[1]
    parent_state_bytes = _git_output(
        repository_root,
        "show",
        f"{parent_sha}:{state_path}",
    )
    try:
        approval_state = json.loads(approval_state_bytes)
        parent_state = json.loads(parent_state_bytes)
        state.validate_state(approval_state)
        state.validate_state(parent_state)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        raise ValueError("approval_ref") from None
    approved_attempt = approval_state.get("current")
    if (
        type(approved_attempt) is not dict
        or approval_state.get("phase") != "implementing"
        or approved_attempt.get("design_approved") is not True
    ):
        raise ValueError("design_approved")
    for field in (
        "paper_id",
        "title",
        "upstream_revision",
        "target_claims",
        "estimated_api_cost_usd",
    ):
        if approved_attempt.get(field) != attempt.get(field):
            raise ValueError(field)
    parent_attempt = parent_state.get("current")
    if (
        type(parent_attempt) is dict
        and parent_attempt.get("paper_id") == attempt["paper_id"]
        and parent_attempt.get("design_approved") is True
    ):
        raise ValueError("approval_ref")
    identity_fields = (
        _git_output(
            repository_root,
            "show",
            "-s",
            "--format=%an%x00%ae%x00%cn%x00%ce",
            commit_sha,
        )
        .rstrip(b"\n")
        .split(b"\x00")
    )
    if len(identity_fields) != 4:
        raise ValueError("approval_ref")
    try:
        author_name, author_email, committer_name, committer_email = (
            _identity(field.decode("utf-8"), "approval_ref")
            for field in identity_fields
        )
    except UnicodeDecodeError:
        raise ValueError("approval_ref") from None
    return {
        "commit_sha": commit_sha,
        "content_sha256": hashlib.sha256(approved_content).hexdigest(),
        "design_author": f"git-author:{author_name} <{author_email}>",
        "reviewer": f"git-committer:{committer_name} <{committer_email}>",
    }


def _git_output(repository: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise ValueError("approval_ref") from None
    return result.stdout


def _timestamp(value: object) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError("now")
    return value.astimezone(timezone.utc).isoformat()
