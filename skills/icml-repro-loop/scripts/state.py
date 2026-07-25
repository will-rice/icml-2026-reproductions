"""Persistent state for the ICML reproduction loop."""

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
import re
import tempfile
from pathlib import Path


PHASES = {
    "idle",
    "selected",
    "design-pending",
    "implementing",
    "validated",
    "deployed",
    "submitted",
    "judging",
    "improving",
    "complete",
    "blocked",
}
# Retained as the migration source schema; new scheduler state uses store.py.
SCHEMA_V3_VERSION = 3
STATE_VERSION = SCHEMA_V3_VERSION
STATE_KEYS = {
    "version",
    "phase",
    "current",
    "history",
    "rejections",
    "total_api_cost_usd",
}
IMMUTABLE_PAPER_FIELDS = {
    "paper_id",
    "title",
    "slug",
    "project_path",
    "upstream_revision",
    "target_claims",
}
IMMUTABLE_ONCE_SET_FIELDS = {"design_approved", "space_id"}
PAPER_COST_FIELDS = {"estimated_api_cost_usd", "actual_api_cost_usd"}
REJECTION_FIELDS = {"paper_id", "title", "reason", "checked_at"}
CURRENT_UPDATE_FIELDS = {
    "actual_api_cost_usd",
    "last_poll_at",
    "last_poll_status",
    "external_ids",
}
OPERATIONAL_FIELDS = {
    "polls",
    "poll_round_start",
    "last_poll_at",
    "last_poll_status",
    "external_ids",
    "verdicts",
}
BLOCKABLE_PHASES = {
    "selected",
    "design-pending",
    "implementing",
    "validated",
    "deployed",
    "submitted",
    "judging",
    "improving",
}
RUNNABLE_PHASES = BLOCKABLE_PHASES
JUDGMENT_PHASES = {"submitted", "judging"}
VERDICT_STATUSES = {
    "verified",
    "partial",
    "inconclusive",
    "contradicted",
    "unavailable",
}
DESIGN_APPROVED_PHASES = {
    "implementing",
    "validated",
    "deployed",
    "submitted",
    "judging",
    "improving",
    "complete",
}
DEPLOYED_PHASES = {"deployed", "submitted", "judging", "improving", "complete"}
SUBMITTED_PHASES = {"submitted", "judging", "improving", "complete"}
SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
ALLOWED = {
    "idle": {"selected"},
    "selected": {"design-pending", "blocked"},
    "design-pending": {"implementing", "blocked"},
    "implementing": {"validated", "blocked"},
    "validated": {"deployed", "blocked"},
    "deployed": {"submitted", "blocked"},
    "submitted": {"judging", "blocked"},
    "judging": {"improving", "complete", "blocked"},
    "improving": {"validated", "blocked"},
    "complete": {"idle"},
    "blocked": {"idle"},
}


def main() -> None:
    """Run the state management command-line interface."""
    parser = argparse.ArgumentParser(
        description="Manage explicit fenced schema-v6 reproduction attempts."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    migrate_parser = commands.add_parser(
        "migrate-v6", help="migrate schema-v3 state to the sharded schema-v6 store"
    )
    migrate_parser.add_argument("path", type=Path)
    migrate_parser.add_argument("--dry-run", action="store_true")
    refresh_parser = commands.add_parser(
        "refresh-live", help="fetch and persist one immutable live Hub snapshot"
    )
    refresh_parser.add_argument("path", type=Path)
    refresh_parser.add_argument("--assessments-json", type=Path)
    list_parser = commands.add_parser("list-attempts", help="list explicit attempts")
    list_parser.add_argument("path", type=Path)
    list_parser.add_argument("--phase", choices=sorted(PHASES))
    list_parser.add_argument("--runnable", action="store_true")
    show_attempt_parser = commands.add_parser(
        "show-attempt", help="show one explicitly identified attempt"
    )
    show_attempt_parser.add_argument("path", type=Path)
    show_attempt_parser.add_argument("--attempt-id", required=True)
    show_snapshot_parser = commands.add_parser(
        "show-snapshot", help="show one immutable content-verified snapshot"
    )
    show_snapshot_parser.add_argument("path", type=Path)
    show_snapshot_parser.add_argument("--snapshot-id", required=True)
    scheduler_parser = commands.add_parser(
        "scheduler-pass", help="refill runnable lanes from one immutable snapshot"
    )
    scheduler_parser.add_argument("path", type=Path)
    scheduler_parser.add_argument("--snapshot-id", required=True)
    scheduler_parser.add_argument("--now")
    claim_parser = commands.add_parser(
        "claim-attempt", help="claim an active attempt from an expected predecessor"
    )
    claim_parser.add_argument("path", type=Path)
    _add_fence_arguments(claim_parser)
    claim_parser.add_argument("--now")
    renew_parser = commands.add_parser(
        "renew-attempt", help="renew one exact live attempt writer"
    )
    renew_parser.add_argument("path", type=Path)
    _add_fence_arguments(renew_parser)
    renew_parser.add_argument("--now")
    transition_attempt_parser = commands.add_parser(
        "transition-attempt",
        help="transition one fenced non-authoritative attempt edge",
    )
    transition_attempt_parser.add_argument("path", type=Path)
    transition_attempt_parser.add_argument("phase", choices=sorted(PHASES))
    _add_fence_arguments(transition_attempt_parser)
    transition_attempt_parser.add_argument("--updates-json", default="{}")
    transition_attempt_parser.add_argument("--now")
    design_parser = commands.add_parser(
        "record-design", help="record one fenced paper-specific design"
    )
    design_parser.add_argument("path", type=Path)
    _add_fence_arguments(design_parser)
    design_parser.add_argument("--author", required=True)
    design_parser.add_argument("--design-path", required=True)
    design_parser.add_argument("--now")
    review_parser = commands.add_parser(
        "review-design", help="record an independent fenced design review"
    )
    review_parser.add_argument("path", type=Path)
    _add_fence_arguments(review_parser)
    review_parser.add_argument("--reviewer", required=True)
    review_parser.add_argument(
        "--decision", choices=("approved", "rejected"), required=True
    )
    review_parser.add_argument("--now")
    watch_parser = commands.add_parser(
        "watch-attempt", help="create a bounded fenced judgment record"
    )
    watch_parser.add_argument("path", type=Path)
    _add_fence_arguments(watch_parser)
    watch_parser.add_argument("--poll-limit", type=int, required=True)
    watch_parser.add_argument("--poll-deadline", required=True)
    watch_parser.add_argument("--now")
    poll_parser = commands.add_parser(
        "record-poll", help="append one fenced judgment poll"
    )
    poll_parser.add_argument("path", type=Path)
    _add_fence_arguments(poll_parser)
    poll_parser.add_argument("--status", required=True)
    poll_parser.add_argument("--now")
    verdict_parser = commands.add_parser(
        "record-verdict", help="persist one fenced judgment verdict"
    )
    verdict_parser.add_argument("path", type=Path)
    _add_fence_arguments(verdict_parser)
    verdict_parser.add_argument("--raw-verdict", required=True)
    verdict_parser.add_argument("--normalized-verdict", required=True)
    verdict_parser.add_argument("--source-revision", required=True)
    verdict_parser.add_argument("--now")
    validation_parser = commands.add_parser(
        "attest-validation",
        help="run and attest one fenced paper validation",
    )
    validation_parser.add_argument("path", type=Path)
    _add_fence_arguments(validation_parser)
    validation_parser.add_argument("--manifest", type=Path, required=True)
    validation_parser.add_argument("--now")
    deployment_parser = commands.add_parser(
        "publish-deployment",
        help="publish and attest one fenced validated Space",
    )
    deployment_parser.add_argument("path", type=Path)
    _add_fence_arguments(deployment_parser)
    deployment_parser.add_argument("--space-id", required=True)
    deployment_parser.add_argument("--source-dir", type=Path, required=True)
    deployment_parser.add_argument("--now")
    submission_parser = commands.add_parser(
        "attest-submission",
        help="attest one exact fenced live submission observation",
    )
    submission_parser.add_argument("path", type=Path)
    _add_fence_arguments(submission_parser)
    submission_parser.add_argument("--snapshot-id", required=True)
    submission_parser.add_argument("--now")
    arguments = parser.parse_args()

    if arguments.command in {
        "refresh-live",
        "list-attempts",
        "show-attempt",
        "show-snapshot",
        "scheduler-pass",
        "claim-attempt",
        "renew-attempt",
        "transition-attempt",
        "record-design",
        "review-design",
        "watch-attempt",
        "record-poll",
        "record-verdict",
        "attest-validation",
        "publish-deployment",
        "attest-submission",
    }:
        state = _run_v6_command(arguments)
    elif arguments.command == "migrate-v6":
        import migrate_v6
        import store

        paths = store.StatePaths(arguments.path)
        plan = migrate_v6.plan_for_existing_migration(paths)
        legacy = plan.source
        if arguments.dry_run:
            state = {
                "active_attempts": len(plan.index["attempts"]),
                "archived_attempts": len(plan.index["history"]),
                "rejections": len(plan.index["rejections"]),
                "max_runnable_attempts": plan.index["max_runnable_attempts"],
                "total_api_cost_usd": plan.index["total_api_cost_usd"],
            }
        else:
            migrate_v6.recover_transactions(paths)
            state = migrate_v6.verify_semantic_equivalence(legacy, paths)
    print(json.dumps(state, indent=2, sort_keys=True))


def _add_fence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--fencing-token", type=int, required=True)


def _run_v6_command(arguments: argparse.Namespace) -> object:
    import attempts
    import leases
    import scheduler
    import store

    paths = store.StatePaths(arguments.path)
    if arguments.command == "refresh-live":
        from huggingface_hub import HfApi
        import refresh

        observed_at = datetime.now(timezone.utc).isoformat()
        assessment_input = (
            None
            if arguments.assessments_json is None
            else refresh.load_assessments(arguments.assessments_json)
        )
        snapshot = refresh.fetch_live_snapshot(
            HfApi(), observed_at, assessment_input
        )
        return {"snapshot_id": refresh.persist_snapshot(paths, snapshot)}
    if arguments.command == "list-attempts":
        index = store.read_json(paths.index)
        store.validate_index(index)
        records = [
            attempts.read_attempt(paths, attempt_id)
            for attempt_id in sorted(index["attempts"])
        ]
        if arguments.phase is not None:
            records = [
                record for record in records if record["phase"] == arguments.phase
            ]
        if arguments.runnable:
            records = [
                record for record in records if record["phase"] in RUNNABLE_PHASES
            ]
        return records
    if arguments.command == "show-attempt":
        return attempts.read_attempt(paths, arguments.attempt_id)
    if arguments.command == "show-snapshot":
        import refresh

        return refresh.read_snapshot(paths, arguments.snapshot_id)
    now = _cli_datetime(getattr(arguments, "now", None))
    if arguments.command == "scheduler-pass":
        report = scheduler.scheduler_pass(paths, arguments.snapshot_id, now)
        return {
            "assignments": [
                {
                    "attempt_id": assignment.attempt_id,
                    "paper_id": assignment.paper_id,
                    "owner": assignment.writer_lease.owner,
                    "fencing_token": assignment.writer_lease.fencing_token,
                }
                for assignment in report.assignments
            ]
        }
    if arguments.command == "claim-attempt":
        lease = leases.claim_attempt(
            paths,
            arguments.attempt_id,
            arguments.owner,
            arguments.fencing_token,
            now,
        )
        return _lease_identity(lease)
    lease = _reconstruct_attempt_lease(
        paths,
        arguments.attempt_id,
        arguments.owner,
        arguments.fencing_token,
        leases,
        store,
    )
    if arguments.command == "renew-attempt":
        return _lease_identity(leases.renew_attempt(paths, lease, now))
    if arguments.command == "transition-attempt":
        return attempts.transition_attempt(
            paths,
            arguments.attempt_id,
            arguments.phase,
            lease,
            now,
            **json.loads(arguments.updates_json),
        )
    if arguments.command == "record-design":
        return attempts.record_design(
            paths,
            arguments.attempt_id,
            lease,
            arguments.author,
            arguments.design_path,
            now,
        )
    if arguments.command == "review-design":
        return attempts.record_design_review(
            paths,
            arguments.attempt_id,
            lease,
            arguments.reviewer,
            arguments.decision,
            now,
        )
    if arguments.command == "watch-attempt":
        return scheduler.watch_attempt(
            paths,
            arguments.attempt_id,
            lease,
            arguments.poll_limit,
            _cli_datetime(arguments.poll_deadline),
            now,
        )
    if arguments.command == "record-poll":
        return scheduler.record_poll(
            paths, arguments.attempt_id, lease, arguments.status, now
        )
    if arguments.command == "attest-validation":
        import controller

        manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
        return controller.attest_validation(
            paths,
            arguments.attempt_id,
            lease,
            manifest,
            controller.run_command,
            now,
        )
    if arguments.command == "publish-deployment":
        from huggingface_hub import HfApi
        import controller

        return controller.publish_and_attest_deployment(
            paths,
            arguments.attempt_id,
            lease,
            arguments.space_id,
            arguments.source_dir,
            HfApi(),
            now,
        )
    if arguments.command == "attest-submission":
        import controller

        return controller.attest_submission(
            paths,
            arguments.attempt_id,
            lease,
            arguments.snapshot_id,
            now,
        )
    return scheduler.record_verdict(
        paths,
        arguments.attempt_id,
        lease,
        json.loads(arguments.raw_verdict),
        json.loads(arguments.normalized_verdict),
        arguments.source_revision,
        now,
    )


def _reconstruct_attempt_lease(paths, attempt_id, owner, fencing_token, leases, store):
    resource = f"attempt:{attempt_id}"
    value = store.read_json(paths.resource_lease(resource))
    leases.validate_lease(value)
    if value["resource"] != resource:
        raise ValueError("resource")
    if value["attempt_id"] != attempt_id:
        raise ValueError("attempt_id")
    if value["owner"] != owner:
        raise ValueError("owner")
    if value["fencing_token"] != fencing_token:
        raise ValueError("fencing_token")
    return leases.Lease(**value)


def _lease_identity(lease) -> dict:
    return {
        "attempt_id": lease.attempt_id,
        "owner": lease.owner,
        "fencing_token": lease.fencing_token,
        "acquired_at": lease.acquired_at,
        "expires_at": lease.expires_at,
    }


def _cli_datetime(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = parse_aware_datetime(value, "now")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("now")
    return parsed.astimezone(timezone.utc)


def new_state() -> dict:
    """Return an empty reproduction loop state."""
    return {
        "version": STATE_VERSION,
        "phase": "idle",
        "current": None,
        "history": [],
        "rejections": [],
        "total_api_cost_usd": 0.0,
    }


def load_state(path: Path) -> dict:
    """Load and validate state from a JSON file."""
    with path.open(encoding="utf-8") as file:
        state = json.load(file)
    validate_state(state)
    return state


def save_state(path: Path, state: dict) -> None:
    """Validate and atomically save state as JSON."""
    validate_state(state)
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
            json.dump(state, file, allow_nan=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def select_paper(state: dict, paper: dict) -> dict:
    """Select a previously uncompleted paper for the loop."""
    validate_state(state)
    if state["phase"] != "idle":
        raise ValueError("phase")
    if not isinstance(paper, dict):
        raise ValueError("paper")
    for field in (
        "paper_id",
        "title",
        "slug",
        "estimated_api_cost_usd",
        "upstream_revision",
        "target_claims",
    ):
        if field not in paper:
            raise ValueError(field)
    if type(paper["upstream_revision"]) is not str or not paper["upstream_revision"]:
        raise ValueError("upstream_revision")
    validate_target_claims(paper["target_claims"])
    if any(
        isinstance(completed, dict)
        and completed.get("paper_id") == paper["paper_id"]
        for completed in state["history"]
    ):
        raise ValueError("paper_id")
    if any(
        rejected["paper_id"] == paper["paper_id"]
        for rejected in state["rejections"]
    ):
        raise ValueError("paper_id")

    current = copy.deepcopy(paper)
    current["project_path"] = f"submissions/{current['slug']}"
    current["polls"] = []
    current["improvement_attempts"] = 0
    current["verdicts"] = []
    validate_paper_costs(current)
    if any(
        completed.get("project_path") == current["project_path"]
        for completed in state["history"]
    ):
        raise ValueError("project_path")

    selected = copy.deepcopy(state)
    selected["phase"] = "selected"
    selected["current"] = current
    validate_state(selected)
    return selected


def reject_candidate(state: dict, candidate: dict) -> dict:
    """Record an ineligible candidate without leaving the idle phase."""
    validate_state(state)
    if state["phase"] != "idle":
        raise ValueError("phase")
    paper_ids = {
        record["paper_id"]
        for record in state["rejections"] + state["history"]
    }
    if state["current"] is not None:
        paper_ids.add(state["current"]["paper_id"])
    validate_rejection_record(candidate, paper_ids)

    rejected = copy.deepcopy(state)
    rejected["rejections"].append(copy.deepcopy(candidate))
    validate_state(rejected)
    return rejected


def transition(state: dict, phase: str, **updates: object) -> dict:
    """Return a copied state after a valid phase transition."""
    validate_state(state)
    source_phase = state["phase"]
    is_blocked_resume = (
        source_phase == "blocked"
        and phase == state["current"].get("blocked_from")
    )
    if type(phase) is not str or phase not in PHASES or (
        phase not in ALLOWED[source_phase] and not is_blocked_resume
    ):
        raise ValueError("phase")
    operational_updates = set(updates) & OPERATIONAL_FIELDS
    if operational_updates:
        raise ValueError(sorted(operational_updates)[0])
    if source_phase == "idle":
        return select_paper(state, updates)

    abandon = updates.pop("abandon", None)
    has_verdict = "verdict" in updates
    transition_verdict = updates.pop("verdict", None)
    if "improvement_attempts" in updates:
        raise ValueError("improvement_attempts")
    for field, required_phase in (
        ("poll_limit", "judging"),
        ("poll_deadline", "judging"),
        ("improvement_reason", "improving"),
    ):
        if field in updates and phase != required_phase:
            raise ValueError(field)
    if has_verdict and phase not in {"improving", "complete"}:
        raise ValueError("verdict")
    if "blocked_from" in updates:
        raise ValueError("blocked_from")
    if "blocker" in updates and phase != "blocked":
        raise ValueError("blocker")
    if source_phase == "blocked" and phase == "idle":
        if abandon is not True:
            raise ValueError("abandon")
    elif abandon is not None:
        raise ValueError("abandon")
    if phase == "blocked":
        if type(updates.get("blocker")) is not str or not updates["blocker"]:
            raise ValueError("blocker")

    transitioned = copy.deepcopy(state)
    current = transitioned["current"]
    if (
        phase == "improving"
        and not is_blocked_resume
        and current["improvement_attempts"] >= 1
    ):
        raise ValueError("improvement_attempts")
    for field in IMMUTABLE_PAPER_FIELDS:
        if field in updates and updates[field] != current.get(field):
            raise ValueError(field)
    if (
        "estimated_api_cost_usd" in updates
        and updates["estimated_api_cost_usd"] != current.get("estimated_api_cost_usd")
    ):
        raise ValueError("estimated_api_cost_usd")
    for field in IMMUTABLE_ONCE_SET_FIELDS:
        if field in updates and field in current and updates[field] != current[field]:
            raise ValueError(field)
    if (
        "deployed_sha" in updates
        and "deployed_sha" in current
        and updates["deployed_sha"] != current["deployed_sha"]
        and not (
            source_phase == "validated"
            and phase == "deployed"
            and current["improvement_attempts"] == 1
        )
    ):
        raise ValueError("deployed_sha")
    if (
        "improvement_reason" in updates
        and "improvement_reason" in current
        and updates["improvement_reason"] != current["improvement_reason"]
    ):
        raise ValueError("improvement_reason")
    current.update(updates)
    if phase == "blocked":
        current["blocked_from"] = source_phase
    elif is_blocked_resume:
        current.pop("blocked_from")
        current.pop("blocker")
    validate_paper_costs(current)
    if (
        "actual_api_cost_usd" in updates
        and "actual_api_cost_usd" in state["current"]
        and current["actual_api_cost_usd"] < state["current"]["actual_api_cost_usd"]
    ):
        raise ValueError("actual_api_cost_usd")

    if (
        phase == "implementing"
        and not is_blocked_resume
        and updates.get("design_approved") is not True
    ):
        raise ValueError("design_approved")
    if phase == "deployed" and not is_blocked_resume and not updates.get(
        "deployed_sha"
    ):
        raise ValueError("deployed_sha")
    if phase == "submitted" and not is_blocked_resume and not updates.get("space_id"):
        raise ValueError("space_id")
    if phase == "judging":
        if not is_blocked_resume:
            for field in ("poll_limit", "poll_deadline"):
                if field not in updates:
                    raise ValueError(field)
            current["poll_round_start"] = len(current["polls"])
        validate_poll_configuration(current)
    if phase == "complete":
        if not has_verdict:
            raise ValueError("verdict")
        validate_verdict(transition_verdict, current["target_claims"])
        verdict_record = copy.deepcopy(transition_verdict)
        verdict_record["improvement_attempt"] = current["improvement_attempts"]
        current["verdicts"].append(verdict_record)
        current["verdict"] = copy.deepcopy(transition_verdict)
    if phase == "improving" and not is_blocked_resume:
        if (
            type(updates.get("improvement_reason")) is not str
            or not updates["improvement_reason"]
        ):
            raise ValueError("improvement_reason")
        if not has_verdict:
            raise ValueError("verdict")
        validate_verdict(transition_verdict, current["target_claims"])
        current["improvement_attempts"] += 1
        verdict_record = copy.deepcopy(transition_verdict)
        verdict_record["improvement_attempt"] = current["improvement_attempts"]
        verdict_record["improvement_reason"] = current["improvement_reason"]
        current["verdicts"].append(verdict_record)
    if phase == "submitted" and any(
        completed.get("space_id") == current["space_id"]
        for completed in transitioned["history"]
    ):
        raise ValueError("space_id")

    if phase == "idle":
        transitioned["history"].append(copy.deepcopy(current))
        transitioned["total_api_cost_usd"] += current.get("actual_api_cost_usd", 0.0)
        transitioned["current"] = None
    transitioned["phase"] = phase
    validate_state(transitioned)
    return transitioned


def update_current(state: dict, **updates: object) -> dict:
    """Return a copied state with allowed same-phase persistence updates."""
    validate_state(state)
    if state["current"] is None:
        raise ValueError("current")
    unsupported_fields = set(updates) - CURRENT_UPDATE_FIELDS
    if unsupported_fields:
        raise ValueError(sorted(unsupported_fields)[0])
    has_poll_at = "last_poll_at" in updates
    has_poll_status = "last_poll_status" in updates
    if has_poll_at != has_poll_status:
        missing_field = "last_poll_status" if has_poll_at else "last_poll_at"
        raise ValueError(missing_field)
    if has_poll_at:
        if state["phase"] != "judging":
            raise ValueError("phase")
        for field in ("last_poll_at", "last_poll_status"):
            if type(updates[field]) is not str or not updates[field]:
                raise ValueError(field)
        poll_at = parse_aware_datetime(updates["last_poll_at"], "last_poll_at")
        round_poll_count = (
            len(state["current"]["polls"])
            - state["current"]["poll_round_start"]
        )
        if round_poll_count >= state["current"]["poll_limit"]:
            raise ValueError("poll_limit")
        if poll_at > parse_aware_datetime(
            state["current"]["poll_deadline"], "poll_deadline"
        ):
            raise ValueError("poll_deadline")

    updated = copy.deepcopy(state)
    current = updated["current"]
    if "actual_api_cost_usd" in updates:
        current["actual_api_cost_usd"] = copy.deepcopy(
            updates["actual_api_cost_usd"]
        )
    if has_poll_at:
        poll = {
            "at": updates["last_poll_at"],
            "status": updates["last_poll_status"],
        }
        validate_polls([poll])
        current.setdefault("polls", []).append(copy.deepcopy(poll))
        current["last_poll_at"] = poll["at"]
        current["last_poll_status"] = poll["status"]
    if "external_ids" in updates:
        external_ids = updates["external_ids"]
        validate_external_ids(external_ids)
        persisted_ids = current.setdefault("external_ids", {})
        if any(
            key in persisted_ids and persisted_ids[key] != value
            for key, value in external_ids.items()
        ):
            raise ValueError("external_ids")
        persisted_ids.update(copy.deepcopy(external_ids))

    validate_state(updated)
    if (
        "actual_api_cost_usd" in updates
        and "actual_api_cost_usd" in state["current"]
        and updated["current"]["actual_api_cost_usd"]
        < state["current"]["actual_api_cost_usd"]
    ):
        raise ValueError("actual_api_cost_usd")
    return updated


def validate_state(state: dict) -> None:
    """Raise ValueError when state does not satisfy the persisted schema."""
    if not isinstance(state, dict) or set(state) != STATE_KEYS:
        raise ValueError("keys")
    if type(state["version"]) is not int or state["version"] != STATE_VERSION:
        raise ValueError("version")
    if type(state["phase"]) is not str or state["phase"] not in PHASES:
        raise ValueError("phase")
    validate_cost(state["total_api_cost_usd"], "total_api_cost_usd")
    if not isinstance(state["history"], list):
        raise ValueError("history")
    if not isinstance(state["rejections"], list):
        raise ValueError("rejections")
    paper_ids = set()
    for rejected in state["rejections"]:
        validate_rejection_record(rejected, paper_ids)
    project_paths = set()
    space_ids = set()
    for completed in state["history"]:
        if not isinstance(completed, dict):
            raise ValueError("history")
        validate_paper_record(completed, paper_ids, project_paths, space_ids)
    if state["phase"] == "idle" and state["current"] is not None:
        raise ValueError("current")
    if state["phase"] != "idle" and not isinstance(state["current"], dict):
        raise ValueError("current")
    if state["current"] is not None:
        validate_paper_record(
            state["current"], paper_ids, project_paths, space_ids, state["phase"]
        )


def validate_cost(value: object, field: str) -> None:
    """Raise ValueError when a cost is not a nonnegative number."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(field)


def validate_paper_costs(paper: dict) -> None:
    """Raise ValueError when a per-paper cost is outside its allowed range."""
    for field in PAPER_COST_FIELDS:
        if field in paper:
            validate_cost(paper[field], field)
            if paper[field] > 10.0:
                raise ValueError(field)


def validate_paper_record(
    paper: dict,
    paper_ids: set[str],
    project_paths: set[str],
    space_ids: set[str],
    active_phase: str | None = None,
) -> None:
    """Validate persistent paper identity and cost invariants."""
    validate_paper_costs(paper)
    for field in ("paper_id", "title"):
        if type(paper.get(field)) is not str or not paper[field]:
            raise ValueError(field)
    if paper["paper_id"] in paper_ids:
        raise ValueError("paper_id")
    paper_ids.add(paper["paper_id"])
    if "estimated_api_cost_usd" not in paper:
        raise ValueError("estimated_api_cost_usd")
    if type(paper.get("upstream_revision")) is not str or not paper[
        "upstream_revision"
    ]:
        raise ValueError("upstream_revision")
    validate_target_claims(paper.get("target_claims"))
    if type(paper.get("verdicts")) is not list:
        raise ValueError("verdicts")
    if (
        type(paper.get("improvement_attempts")) is not int
        or paper["improvement_attempts"] not in {0, 1}
    ):
        raise ValueError("improvement_attempts")
    has_improvement_reason = "improvement_reason" in paper
    if paper["improvement_attempts"] == 1 and (
        type(paper.get("improvement_reason")) is not str
        or not paper["improvement_reason"]
    ):
        raise ValueError("improvement_reason")
    if paper["improvement_attempts"] == 0 and has_improvement_reason:
        raise ValueError("improvement_reason")
    if type(paper.get("slug")) is not str or not SLUG_PATTERN.fullmatch(paper["slug"]):
        raise ValueError("slug")
    if (
        type(paper.get("project_path")) is not str
        or paper["project_path"] != f"submissions/{paper['slug']}"
    ):
        raise ValueError("project_path")
    if paper["project_path"] in project_paths:
        raise ValueError("project_path")
    project_paths.add(paper["project_path"])
    if "space_id" in paper:
        if type(paper["space_id"]) is not str or not paper["space_id"]:
            raise ValueError("space_id")
        if paper["space_id"] in space_ids:
            raise ValueError("space_id")
        space_ids.add(paper["space_id"])
    polls = paper.get("polls", [])
    validate_polls(polls)
    if any(
        field in paper for field in ("poll_limit", "poll_deadline", "poll_round_start")
    ):
        validate_poll_configuration(paper)
    has_poll_at = "last_poll_at" in paper
    has_poll_status = "last_poll_status" in paper
    if has_poll_at != has_poll_status:
        missing_field = "last_poll_status" if has_poll_at else "last_poll_at"
        raise ValueError(missing_field)
    for field in ("last_poll_at", "last_poll_status"):
        if field in paper and (type(paper[field]) is not str or not paper[field]):
            raise ValueError(field)
    if has_poll_at and not polls:
        raise ValueError("last_poll_at")
    if has_poll_at and paper["last_poll_at"] != polls[-1]["at"]:
        raise ValueError("last_poll_at")
    if has_poll_status and paper["last_poll_status"] != polls[-1]["status"]:
        raise ValueError("last_poll_status")
    if "external_ids" in paper:
        validate_external_ids(paper["external_ids"])
    has_blocked_from = "blocked_from" in paper
    has_blocker = "blocker" in paper
    if has_blocked_from != has_blocker:
        raise ValueError("blocker" if has_blocked_from else "blocked_from")
    if has_blocked_from and paper["blocked_from"] not in BLOCKABLE_PHASES:
        raise ValueError("blocked_from")
    if has_blocker and (
        type(paper["blocker"]) is not str or not paper["blocker"]
    ):
        raise ValueError("blocker")
    if active_phase == "blocked" and not has_blocked_from:
        raise ValueError("blocked_from")

    effective_phase = active_phase
    if active_phase == "blocked":
        effective_phase = paper["blocked_from"]
    elif active_phase is None:
        effective_phase = paper.get("blocked_from", "complete")
    validate_phase_prerequisites(paper, effective_phase)
    validate_verdict_history(paper, effective_phase)


def validate_rejection_record(candidate: object, paper_ids: set[str]) -> None:
    """Validate a persisted candidate rejection and its unique paper ID."""
    if type(candidate) is not dict or set(candidate) != REJECTION_FIELDS:
        raise ValueError("rejections")
    for field in REJECTION_FIELDS:
        if type(candidate[field]) is not str or not candidate[field]:
            raise ValueError(field)
    if candidate["paper_id"] in paper_ids:
        raise ValueError("paper_id")
    paper_ids.add(candidate["paper_id"])


def validate_polls(polls: object) -> None:
    """Raise ValueError unless polls are exact nonempty string records."""
    if type(polls) is not list or any(
        type(poll) is not dict
        or set(poll) != {"at", "status"}
        or type(poll["at"]) is not str
        or not poll["at"]
        or type(poll["status"]) is not str
        or not poll["status"]
        for poll in polls
    ):
        raise ValueError("polls")


def validate_poll_configuration(paper: dict) -> None:
    """Validate the finite judging budget and every persisted observation."""
    poll_limit = paper.get("poll_limit")
    if type(poll_limit) is not int or poll_limit <= 0:
        raise ValueError("poll_limit")
    deadline = parse_aware_datetime(paper.get("poll_deadline"), "poll_deadline")
    polls = paper.get("polls", [])
    poll_round_start = paper.get("poll_round_start")
    if (
        type(poll_round_start) is not int
        or poll_round_start < 0
        or poll_round_start > len(polls)
    ):
        raise ValueError("poll_round_start")
    if len(polls) - poll_round_start > poll_limit:
        raise ValueError("poll_limit")
    for poll in polls[poll_round_start:]:
        if parse_aware_datetime(poll["at"], "polls") > deadline:
            raise ValueError("poll_deadline")


def validate_target_claims(target_claims: object) -> None:
    """Validate immutable claim names selected for reproduction."""
    if (
        type(target_claims) is not list
        or len(target_claims) < 2
        or any(type(claim) is not str or not claim for claim in target_claims)
        or len(set(target_claims)) != len(target_claims)
    ):
        raise ValueError("target_claims")


def validate_verdict(verdict: object, target_claims: list[str]) -> None:
    """Validate claim-level completion outcomes."""
    if type(verdict) is not dict or type(verdict.get("claims")) is not list or not verdict[
        "claims"
    ]:
        raise ValueError("verdict")
    if any(
        type(claim) is not dict
        or set(claim) != {"claim", "status"}
        or type(claim["claim"]) is not str
        or not claim["claim"]
        or type(claim["status"]) is not str
        or claim["status"] not in VERDICT_STATUSES
        for claim in verdict["claims"]
    ):
        raise ValueError("verdict")
    claim_names = [claim["claim"] for claim in verdict["claims"]]
    if len(claim_names) != len(set(claim_names)) or set(claim_names) != set(
        target_claims
    ):
        raise ValueError("verdict")


def validate_phase_prerequisites(paper: dict, phase: str | None) -> None:
    """Validate artifacts required by the effective persisted phase."""
    if phase in DESIGN_APPROVED_PHASES and paper.get("design_approved") is not True:
        raise ValueError("design_approved")
    if phase in DEPLOYED_PHASES and (
        type(paper.get("deployed_sha")) is not str or not paper["deployed_sha"]
    ):
        raise ValueError("deployed_sha")
    if phase in SUBMITTED_PHASES and (
        type(paper.get("space_id")) is not str or not paper["space_id"]
    ):
        raise ValueError("space_id")
    if phase == "judging":
        for field in ("poll_limit", "poll_deadline", "poll_round_start"):
            if field not in paper:
                raise ValueError(field)
        validate_poll_configuration(paper)
    if phase == "improving" and paper["improvement_attempts"] != 1:
        raise ValueError("improvement_attempts")


def validate_verdict_history(paper: dict, phase: str | None) -> None:
    """Validate authoritative verdict records and the final verdict alias."""
    verdicts = paper["verdicts"]
    for verdict_record in verdicts:
        validate_verdict(verdict_record, paper["target_claims"])
        if (
            type(verdict_record.get("improvement_attempt")) is not int
            or verdict_record["improvement_attempt"] not in {0, 1}
        ):
            raise ValueError("verdicts")
        if "improvement_reason" in verdict_record and (
            verdict_record["improvement_attempt"] != 1
            or type(verdict_record["improvement_reason"]) is not str
            or not verdict_record["improvement_reason"]
        ):
            raise ValueError("verdicts")

    attempts = paper["improvement_attempts"]
    if any(
        verdict_record["improvement_attempt"] != attempts
        for verdict_record in verdicts
    ):
        raise ValueError("verdicts")
    if attempts == 1 and (
        not verdicts
        or verdicts[0].get("improvement_reason") != paper["improvement_reason"]
    ):
        raise ValueError("verdicts")
    if attempts == 0 and any(
        "improvement_reason" in verdict_record for verdict_record in verdicts
    ):
        raise ValueError("verdicts")
    if attempts == 1 and any(
        "improvement_reason" in verdict_record for verdict_record in verdicts[1:]
    ):
        raise ValueError("verdicts")
    expected_before_completion = attempts
    if phase == "complete":
        if len(verdicts) != expected_before_completion + 1:
            raise ValueError("verdicts")
        if "verdict" not in paper:
            raise ValueError("verdict")
    elif len(verdicts) != expected_before_completion:
        raise ValueError("verdicts")

    if "verdict" in paper:
        validate_verdict(paper["verdict"], paper["target_claims"])
        if not verdicts:
            raise ValueError("verdict")
        final_record = {
            key: value
            for key, value in verdicts[-1].items()
            if key not in {"improvement_attempt", "improvement_reason"}
        }
        if paper["verdict"] != final_record:
            raise ValueError("verdict")
    elif phase == "complete":
        raise ValueError("verdict")


def parse_aware_datetime(value: object, field: str) -> datetime:
    """Parse a timezone-aware ISO-8601 timestamp."""
    if type(value) is not str or not value:
        raise ValueError(field)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(field) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(field)
    return parsed


def validate_external_ids(external_ids: object) -> None:
    """Raise ValueError unless external IDs are nonempty strings."""
    if type(external_ids) is not dict or not external_ids or any(
        type(key) is not str
        or not key
        or type(value) is not str
        or not value
        for key, value in external_ids.items()
    ):
        raise ValueError("external_ids")


if __name__ == "__main__":
    main()
