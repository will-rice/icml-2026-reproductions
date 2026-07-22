"""Persistent state for the ICML reproduction loop."""

import argparse
import copy
from datetime import datetime
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
STATE_VERSION = 3
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
        description=(
            "Manage schema-v3 reproduction state. Selection JSON requires "
            "estimated_api_cost_usd, upstream_revision, and target_claims; "
            "judging transitions "
            "require poll_limit and poll_deadline; blocked transitions require "
            "blocker and archival requires abandon=true; improving requires "
            "improvement_reason and claim-level verdicts; verdict transitions "
            "append authoritative history and judging budgets are round-scoped."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "show"):
        subparser = commands.add_parser(command, help=f"{command} state")
        subparser.add_argument("path", type=Path)
    select_parser = commands.add_parser(
        "select",
        help="select a paper with explicit cost, revision, and target claims",
    )
    select_parser.add_argument("path", type=Path)
    select_parser.add_argument("paper_json")
    reject_parser = commands.add_parser("reject", help="record an idle rejection")
    reject_parser.add_argument("path", type=Path)
    reject_parser.add_argument("candidate_json")
    update_parser = commands.add_parser(
        "update", help="persist judging polls, cost, or external IDs"
    )
    update_parser.add_argument("path", type=Path)
    update_parser.add_argument("updates_json")
    transition_parser = commands.add_parser(
        "transition", help="change phase with required phase metadata"
    )
    transition_parser.add_argument("path", type=Path)
    transition_parser.add_argument("phase", choices=sorted(PHASES))
    transition_parser.add_argument("updates_json")
    arguments = parser.parse_args()

    if arguments.command == "init":
        if arguments.path.exists():
            raise FileExistsError(arguments.path)
        state = new_state()
        save_state(arguments.path, state)
    elif arguments.command == "show":
        state = load_state(arguments.path)
    elif arguments.command == "select":
        state = select_paper(load_state(arguments.path), json.loads(arguments.paper_json))
        save_state(arguments.path, state)
    elif arguments.command == "reject":
        state = reject_candidate(
            load_state(arguments.path), json.loads(arguments.candidate_json)
        )
        save_state(arguments.path, state)
    elif arguments.command == "update":
        state = update_current(
            load_state(arguments.path), **json.loads(arguments.updates_json)
        )
        save_state(arguments.path, state)
    else:
        state = transition(
            load_state(arguments.path),
            arguments.phase,
            **json.loads(arguments.updates_json),
        )
        save_state(arguments.path, state)
    print(json.dumps(state, indent=2, sort_keys=True))


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
