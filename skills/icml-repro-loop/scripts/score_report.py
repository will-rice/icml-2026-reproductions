"""Read-only official score, capacity, queue, and telemetry reporting."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import math
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import score_rate  # noqa: E402
import state  # noqa: E402
import store  # noqa: E402
import telemetry  # noqa: E402


RANK_OBSERVATION_KEYS = {
    "observed_at",
    "source_url",
    "username",
    "points",
    "rank",
}
PENDING_PHASES = {"submitted", "judging"}
AWAITING_VALIDATION_PHASES = {"implementing", "improving"}
AWAITING_DEPLOYMENT_PHASES = {"validated"}


def official_points(snapshot: dict, username: str) -> dict:
    """Score the earliest canonical judged logbook for each user/paper pair."""
    return _official_summary(_official_points_by_paper(snapshot, username))


def _official_summary(by_paper: dict[str, dict]) -> dict:
    return {
        "points": sum(record["points"] for record in by_paper.values()),
        "max_points": sum(
            record["max_points"] for record in by_paper.values()
        ),
        "judged_papers": len(by_paper),
    }


def candidate_queue(snapshot: dict) -> list[dict]:
    """Return assessed candidates in deterministic score-rate priority order."""
    if type(snapshot) is not dict or type(snapshot.get("candidates")) is not list:
        raise ValueError("candidates")
    candidates = [
        candidate
        for candidate in snapshot["candidates"]
        if type(candidate) is dict
        and type(candidate.get("score_rate")) is dict
    ]
    ranked = sorted(candidates, key=score_rate.ranking_key)
    return [
        {
            "authority": "estimate",
            "paper_id": _identity(candidate.get("paper_id"), "paper_id"),
            "title": candidate.get("title"),
            "expected_points": score_rate.expected_points(
                candidate["score_rate"]
            ),
            "priority": score_rate.priority(candidate["score_rate"]),
            "remaining_hours_p90": candidate["score_rate"][
                "remaining_hours_p90"
            ],
            "judged_before_deadline_probability": candidate["score_rate"][
                "judged_before_deadline_probability"
            ],
            "primary_risk": candidate["score_rate"]["primary_risk"],
        }
        for candidate in ranked
    ]


def read_git_head(worktree: Path) -> str:
    """Return the exact checked-out revision for one registered worktree."""
    result = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    if not head or "\n" in head:
        raise ValueError("git_head")
    return head


def candidate_census(
    paths: store.StatePaths,
    snapshot: dict,
    worktree_roots: list[Path],
    *,
    git_head=read_git_head,
) -> list[dict]:
    """List unclaimed live candidates without assessing feasibility or score."""
    if type(snapshot) is not dict or type(snapshot.get("candidates")) is not list:
        raise ValueError("candidates")
    import scheduler

    index = store.read_json(paths.index)
    store.validate_index(index)
    claimed = scheduler._claimed_paper_ids(
        paths, index, snapshot, datetime.now(timezone.utc)
    )
    roots = sorted({Path(root).resolve() for root in worktree_roots}, key=str)
    rows = []
    for candidate in snapshot["candidates"]:
        if type(candidate) is not dict:
            raise ValueError("candidates")
        paper_id = _identity(candidate.get("paper_id"), "paper_id")
        slug = _identity(candidate.get("slug"), "slug")
        title = _identity(candidate.get("title"), "title")
        live_claims = candidate.get("live_claims")
        if type(live_claims) is not list:
            raise ValueError("live_claims")
        if paper_id in claimed or len(live_claims) < 2:
            continue
        existing_projects = _existing_projects(slug, roots, git_head)
        rows.append(
            {
                "paper_id": paper_id,
                "title": title,
                "claim_count": len(live_claims),
                "existing_projects": existing_projects,
                "authority": "research-required",
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            not row["existing_projects"],
            -row["claim_count"],
            row["paper_id"],
        ),
    )


def _existing_projects(slug: str, roots: list[Path], git_head) -> list[dict]:
    projects = []
    for worktree in roots:
        project = worktree / "submissions" / slug
        if not project.is_dir() or not (project / "pyproject.toml").is_file():
            continue
        try:
            head = git_head(worktree)
        except (OSError, subprocess.CalledProcessError, ValueError):
            continue
        if type(head) is not str or not head:
            continue
        projects.append({"path": str(project), "git_head": head})
    return sorted(projects, key=lambda record: record["path"])


def build_report(
    paths: store.StatePaths,
    snapshot: dict,
    username: str,
    rank_observation: dict | None = None,
) -> dict:
    """Build one offline report without repairing or mutating coordinator state."""
    snapshot_id = _identity(snapshot.get("snapshot_id"), "snapshot_id")
    index, active_attempts, all_attempts = _read_attempts(paths)
    official_by_paper = _official_points_by_paper(snapshot, username)
    official = _official_summary(official_by_paper)
    validated_rank = _validate_rank_observation(
        rank_observation, username, official["points"]
    )

    pending_papers = []
    for attempt in active_attempts:
        if (
            attempt["phase"] not in PENDING_PHASES
            or attempt["paper_id"] in official_by_paper
        ):
            continue
        envelope = attempt.get("score_rate")
        live_claims = attempt.get("live_claims")
        try:
            score_rate.validate_envelope(envelope, live_claims)
        except ValueError:
            continue
        pending_papers.append(
            {
                "attempt_id": attempt["attempt_id"],
                "paper_id": attempt["paper_id"],
                "phase": attempt["phase"],
                "expected_points": score_rate.expected_points(envelope),
                "authority": "estimate",
            }
        )
    pending_papers.sort(key=lambda record: record["attempt_id"])

    phase_counts = {}
    for attempt in active_attempts:
        phase_counts[attempt["phase"]] = phase_counts.get(attempt["phase"], 0) + 1
    phase_counts = dict(sorted(phase_counts.items()))

    max_runnable = index["max_runnable_attempts"]
    runnable = sum(
        attempt["phase"] in state.RUNNABLE_PHASES
        for attempt in active_attempts
    )
    attempt_rows = [
        {
            "attempt_id": attempt["attempt_id"],
            "paper_id": attempt["paper_id"],
            "phase": attempt["phase"],
        }
        for attempt in active_attempts
    ]
    awaiting_validation = [
        row
        for row in attempt_rows
        if row["phase"] in AWAITING_VALIDATION_PHASES
    ]
    awaiting_deployment = [
        row
        for row in attempt_rows
        if row["phase"] in AWAITING_DEPLOYMENT_PHASES
    ]
    blockers = [
        {
            "attempt_id": attempt["attempt_id"],
            "paper_id": attempt["paper_id"],
            "blocked_from": attempt.get("blocked_from"),
            "blocker": attempt["blocker"],
        }
        for attempt in active_attempts
        if attempt["phase"] == "blocked"
        and type(attempt.get("blocker")) is str
        and attempt["blocker"]
    ]

    return {
        "official": {
            "authority": "official-verdict-snapshot",
            "snapshot_id": snapshot_id,
            "username": username,
            **official,
            "rank_observation": validated_rank,
        },
        "pending_judgment": {
            "authority": "estimate",
            "papers": pending_papers,
            "expected_points": sum(
                record["expected_points"] for record in pending_papers
            ),
        },
        "capacity": {
            "max_runnable": max_runnable,
            "runnable": runnable,
            "idle": max_runnable - runnable,
        },
        "phases": phase_counts,
        "awaiting_validation": awaiting_validation,
        "awaiting_deployment": awaiting_deployment,
        "blockers": blockers,
        "candidate_queue": candidate_queue(snapshot),
        "telemetry": _telemetry_report(
            paths,
            all_attempts,
            official_by_paper,
        ),
    }


def _official_points_by_paper(snapshot: dict, username: str) -> dict[str, dict]:
    _identity(username, "username")
    if type(snapshot) is not dict or type(snapshot.get("verdicts")) is not list:
        raise ValueError("verdicts")
    canonical: dict[str, tuple[datetime, str, dict]] = {}
    for verdict in snapshot["verdicts"]:
        if type(verdict) is not dict:
            raise ValueError("verdicts")
        space_id = verdict.get("space_id")
        if type(space_id) is not str:
            continue
        owner, separator, name = space_id.partition("/")
        if not separator or not name or owner != username:
            continue
        paper_id = verdict.get("paper_id")
        if type(paper_id) is not str or not paper_id:
            continue
        judged_at = _aware_datetime(verdict.get("judged_at"))
        if judged_at is None:
            continue
        key = (judged_at, space_id)
        existing = canonical.get(paper_id)
        if existing is None or key < existing[:2]:
            canonical[paper_id] = (judged_at, space_id, verdict)

    scored = {}
    for paper_id, (_judged_at, _space_id, verdict) in canonical.items():
        claims = verdict.get("claims")
        if type(claims) is not list:
            raise ValueError("claims")
        points = 0
        for claim in claims:
            if type(claim) is not dict:
                raise ValueError("claims")
            points += score_rate.claim_points(claim.get("verdict"))
        scored[paper_id] = {
            "points": points,
            "max_points": 2 * len(claims),
        }
    return scored


def _validate_rank_observation(
    observation: dict | None, username: str, points: int
) -> dict | None:
    if observation is None:
        return None
    if type(observation) is not dict or set(observation) != RANK_OBSERVATION_KEYS:
        raise ValueError("rank_observation")
    source_url = observation["source_url"]
    parsed_url = urlparse(source_url) if type(source_url) is str else None
    if (
        _aware_datetime(observation["observed_at"]) is None
        or parsed_url is None
        or parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
        or observation["username"] != username
        or type(observation["points"]) is not int
        or isinstance(observation["points"], bool)
        or observation["points"] != points
        or type(observation["rank"]) is not int
        or isinstance(observation["rank"], bool)
        or observation["rank"] < 1
    ):
        raise ValueError("rank_observation")
    return copy.deepcopy(observation)


def _read_attempts(
    paths: store.StatePaths,
) -> tuple[dict, list[dict], list[dict]]:
    """Read validated shards directly; reporting must never trigger recovery."""
    index = store.read_json(paths.index)
    store.validate_index(index)
    records_by_section = {}
    for section in ("attempts", "history"):
        records = []
        for attempt_id, reference in sorted(index[section].items()):
            expected = paths.attempt(attempt_id)
            actual = paths.index.parent / reference["path"]
            if actual != expected:
                raise ValueError("attempt")
            attempt = store.read_json(actual)
            store.validate_attempt(attempt)
            if (
                attempt["attempt_id"] != attempt_id
                or attempt["paper_id"] != reference["paper_id"]
                or attempt["phase"] != reference["phase"]
                or attempt["updated_at"] != reference["updated_at"]
            ):
                raise ValueError("attempt")
            records.append(attempt)
        records_by_section[section] = records
    all_attempts = records_by_section["attempts"] + records_by_section["history"]
    paper_ids = [attempt["paper_id"] for attempt in all_attempts]
    if len(paper_ids) != len(set(paper_ids)):
        raise ValueError("paper_id")
    return index, records_by_section["attempts"], all_attempts


def _telemetry_report(
    paths: store.StatePaths,
    all_attempts: list[dict],
    official_by_paper: dict[str, dict],
) -> dict:
    worker_queue: list[float] = []
    worker_process: list[float] = []
    worker_by_attempt: dict[str, list[float]] = {}
    launches_by_attempt: dict[str, list[datetime]] = {}
    submissions_by_attempt: dict[str, list[datetime]] = {}
    stages = {"validation": [], "deployment": []}
    validation_by_attempt: dict[str, list[tuple[datetime, str]]] = {}
    implementation_sessions = 0
    correction_sessions = 0
    open_sessions = 0
    attempt_papers = {
        attempt["attempt_id"]: attempt["paper_id"] for attempt in all_attempts
    }

    for events in telemetry.iter_sessions(paths):
        event_names = {event["event"] for event in events}
        if event_names & {"worker-queued", "worker-launched", "worker-exited"}:
            worker_identity = _worker_identity(events, attempt_papers)
            if worker_identity is None:
                continue
            worker_attempt_id, _worker_paper_id, work_kind = worker_identity
            summary = telemetry.summarize_worker_session(events)
            if summary["status"] == "open":
                open_sessions += 1
            queued = _first_event(events, "worker-queued")
            launched = _first_event(events, "worker-launched")
            if queued is not None and launched is not None:
                queued_at = _aware_datetime(queued.get("observed_at"))
                launched_at = _aware_datetime(launched.get("observed_at"))
                if (
                    queued_at is not None
                    and launched_at is not None
                    and launched_at >= queued_at
                ):
                    worker_queue.append((launched_at - queued_at).total_seconds())
            if launched is not None:
                launched_at = _aware_datetime(launched.get("observed_at"))
                if launched_at is not None:
                    launches_by_attempt.setdefault(
                        worker_attempt_id, []
                    ).append(
                        launched_at,
                    )
                if work_kind == "implementation":
                    implementation_sessions += 1
                elif work_kind == "correction":
                    correction_sessions += 1
            elapsed = summary["elapsed_seconds"]
            if elapsed is not None:
                worker_process.append(elapsed)
                worker_by_attempt.setdefault(worker_attempt_id, []).append(
                    elapsed
                )

        started = _first_event(events, "stage-started")
        finished = _first_event(events, "stage-finished")
        if started is not None and finished is not None:
            stage = started.get("stage")
            attempt_id = started.get("attempt_id")
            elapsed = finished.get("elapsed_seconds")
            if (
                stage in stages
                and attempt_id in attempt_papers
                and _nonnegative_finite_number(elapsed)
            ):
                stages[stage].append(float(elapsed))
                if stage == "validation":
                    started_at = _aware_datetime(started.get("observed_at"))
                    outcome = finished.get("outcome")
                    if (
                        started_at is not None
                        and outcome in {"passed", "failed"}
                    ):
                        validation_by_attempt.setdefault(attempt_id, []).append(
                            (started_at, outcome)
                        )

        for event in events:
            if (
                event["event"] == "observation"
                and event.get("name") == "submission-observed"
            ):
                attempt_id = event.get("attempt_id")
                observed_at = _aware_datetime(event.get("observed_at"))
                if attempt_id in attempt_papers and observed_at is not None:
                    submissions_by_attempt.setdefault(attempt_id, []).append(
                        observed_at
                    )

    end_to_end_by_attempt = {}
    for attempt_id, launches in launches_by_attempt.items():
        submissions = submissions_by_attempt.get(attempt_id, [])
        if not submissions:
            continue
        first_launch = min(launches)
        later_submissions = [
            observed_at
            for observed_at in submissions
            if observed_at >= first_launch
        ]
        if later_submissions:
            end_to_end_by_attempt[attempt_id] = (
                min(later_submissions) - first_launch
            ).total_seconds()

    first_validation_outcomes = [
        min(records, key=lambda record: record[0])[1]
        for records in validation_by_attempt.values()
    ]
    attempt_id_by_paper = {
        attempt["paper_id"]: attempt["attempt_id"] for attempt in all_attempts
    }
    worker_rate = _points_rate(
        official_by_paper,
        attempt_id_by_paper,
        {
            attempt_id: sum(durations)
            for attempt_id, durations in worker_by_attempt.items()
        },
    )
    end_to_end_rate = _points_rate(
        official_by_paper,
        attempt_id_by_paper,
        end_to_end_by_attempt,
    )

    return {
        "worker_queue_seconds": _sum_or_none(worker_queue),
        "worker_process_seconds": _sum_or_none(worker_process),
        "implementation_sessions": implementation_sessions,
        "correction_sessions": correction_sessions,
        "validation_seconds": _sum_or_none(stages["validation"]),
        "deployment_seconds": _sum_or_none(stages["deployment"]),
        "first_launch_to_submission_seconds": _sum_or_none(
            list(end_to_end_by_attempt.values())
        ),
        "first_pass_validation_rate": (
            None
            if not first_validation_outcomes
            else first_validation_outcomes.count("passed")
            / len(first_validation_outcomes)
        ),
        "judged_points_per_worker_hour": worker_rate,
        "judged_points_per_end_to_end_hour": end_to_end_rate,
        "open_sessions": open_sessions,
    }


def _points_rate(
    official_by_paper: dict[str, dict],
    attempt_id_by_paper: dict[str, str],
    seconds_by_attempt: dict[str, float],
) -> float | None:
    points = 0
    seconds = 0.0
    for paper_id, score in official_by_paper.items():
        attempt_id = attempt_id_by_paper.get(paper_id)
        duration = seconds_by_attempt.get(attempt_id) if attempt_id else None
        if duration is None:
            continue
        points += score["points"]
        seconds += duration
    if seconds <= 0:
        return None
    return points / (seconds / 3600)


def _first_event(events: list[dict], name: str) -> dict | None:
    return next((event for event in events if event["event"] == name), None)


def _worker_identity(
    events: list[dict], attempt_papers: dict[str, str]
) -> tuple[str, str, str] | None:
    endpoints = [
        event
        for event in events
        if event["event"]
        in {"worker-queued", "worker-launched", "worker-exited"}
    ]
    identities = {
        (
            event.get("attempt_id"),
            event.get("paper_id"),
            event.get("work_kind"),
        )
        for event in endpoints
    }
    if len(identities) != 1:
        return None
    attempt_id, paper_id, work_kind = identities.pop()
    if (
        type(attempt_id) is not str
        or type(paper_id) is not str
        or work_kind not in {"implementation", "correction"}
        or attempt_papers.get(attempt_id) != paper_id
    ):
        return None
    return attempt_id, paper_id, work_kind


def _sum_or_none(values: list[float]) -> float | None:
    return sum(values) if values else None


def _aware_datetime(value: object) -> datetime | None:
    if type(value) is not str:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _identity(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(field)
    return value


def _nonnegative_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )
