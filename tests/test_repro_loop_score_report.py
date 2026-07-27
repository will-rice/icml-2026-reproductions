"""Behavioral tests for the offline official score and queue report."""

import hashlib
import importlib
import sys
from pathlib import Path

import pytest


SCRIPTS = (
    Path(__file__).resolve().parents[1] / "skills" / "icml-repro-loop" / "scripts"
)
sys.path.insert(0, str(SCRIPTS))
score_report = importlib.import_module("score_report")
leases = importlib.import_module("leases")
store = importlib.import_module("store")
telemetry = importlib.import_module("telemetry")


def envelope(*, expected: float = 1.0, hours: float = 2.0) -> dict:
    return {
        "claim_expectations": [
            {
                "challenge_claim_sha256": hashlib.sha256(b"Claim A").hexdigest(),
                "p_verified": expected / 2,
                "p_falsified": 0.0,
                "p_toy": 0.0,
            }
        ],
        "judged_before_deadline_probability": 0.8,
        "remaining_hours_p90": hours,
        "reusable_implementation": False,
        "direct_artifact_score": 4,
        "full_score_claim_paths": 1,
        "remaining_time_variance_hours2": 0.25,
        "primary_risk": "Artifact schema may have drifted.",
    }


def candidate(paper_id: str, *, expected: float = 1.0, hours: float = 2.0) -> dict:
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "estimated_api_cost_usd": 0.0,
        "score_rate": envelope(expected=expected, hours=hours),
    }


def attempt(
    attempt_id: str,
    paper_id: str,
    phase: str,
    *,
    expected: float | None = None,
    **updates,
) -> dict:
    value = {
        "attempt_id": attempt_id,
        "paper_id": paper_id,
        "phase": phase,
        "updated_at": "2026-07-27T00:00:00+00:00",
    }
    if expected is not None:
        value.update(
            {
                "live_claims": [{"text": "Claim A", "status": "extracted"}],
                "score_rate": envelope(expected=expected),
            }
        )
    value.update(updates)
    return value


def report_paths(tmp_path: Path, records: list[dict]):
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    index = store.new_index()
    for record in records:
        store.atomic_json_write(
            paths.attempt(record["attempt_id"]),
            record,
            store.validate_attempt,
        )
        index["attempts"][record["attempt_id"]] = {
            "path": str(
                paths.attempt(record["attempt_id"]).relative_to(paths.index.parent)
            ),
            "paper_id": record["paper_id"],
            "phase": record["phase"],
            "updated_at": record["updated_at"],
        }
    store.atomic_json_write(paths.index, index, store.validate_index)
    return paths


def live_candidate(paper_id: str, *, claims: int) -> dict:
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "live_claims": [
            {"text": f"{paper_id} claim {number}", "status": "extracted"}
            for number in range(claims)
        ],
    }


def raw_snapshot(*, candidates: list[dict], **records: list[dict]) -> dict:
    return {
        "candidates": candidates,
        "queued_submissions": records.get("queued_submissions", []),
        "tagged_spaces": records.get("tagged_spaces", []),
        "verdicts": records.get("verdicts", []),
    }


def test_census_excludes_claimed_and_finds_existing_project(tmp_path: Path):
    paths = report_paths(tmp_path, [])
    worktree = tmp_path / "paper-worktree"
    project = worktree / "submissions" / "paper-fast"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='paper-fast'\n")
    snapshot = raw_snapshot(
        candidates=[
            live_candidate("paper-fast", claims=5),
            live_candidate("paper-claimed", claims=6),
            live_candidate("paper-one-claim", claims=1),
        ],
        tagged_spaces=[{"paper_id": "paper-claimed", "space_id": "org/claimed"}],
    )

    rows = score_report.candidate_census(
        paths,
        snapshot,
        [worktree],
        git_head=lambda _worktree: "a" * 40,
    )

    assert rows == [
        {
            "paper_id": "paper-fast",
            "title": "Paper paper-fast",
            "claim_count": 5,
            "existing_projects": [
                {"path": str(project), "git_head": "a" * 40}
            ],
            "authority": "research-required",
        }
    ]


def test_census_excludes_every_durable_and_live_claim_source(tmp_path: Path):
    paths = report_paths(
        tmp_path,
        [
            attempt("active", "attempt-paper", "implementing"),
            attempt("history", "history-paper", "complete"),
        ],
    )
    index = store.read_json(paths.index)
    index["attempts"].pop("history")
    index["history"]["history"] = {
        "path": str(paths.attempt("history").relative_to(paths.index.parent)),
        "paper_id": "history-paper",
        "phase": "complete",
        "updated_at": "2026-07-27T00:00:00+00:00",
    }
    index["rejections"] = [
        {"paper_id": "rejected-paper", "reason": "Already assessed."}
    ]
    store.atomic_json_write(paths.index, index, store.validate_index)
    lease = {
        "resource": "candidate:leased-paper",
        "owner": "scheduler",
        "attempt_id": "lease-attempt",
        "acquired_at": "2026-07-27T00:00:00+00:00",
        "expires_at": "2099-07-27T00:00:00+00:00",
        "fencing_token": 1,
        "released_at": None,
    }
    store.atomic_json_write(
        paths.resource_lease(lease["resource"]), lease, leases.validate_lease
    )
    snapshot = raw_snapshot(
        candidates=[
            live_candidate(paper_id, claims=2)
            for paper_id in (
                "attempt-paper",
                "history-paper",
                "leased-paper",
                "queued-paper",
                "space-paper",
                "verdict-paper",
                "rejected-paper",
                "available-paper",
            )
        ],
        queued_submissions=[{"paper_id": "queued-paper"}],
        tagged_spaces=[{"paper_id": "space-paper"}],
        verdicts=[{"paper_id": "verdict-paper"}],
    )

    rows = score_report.candidate_census(
        paths,
        snapshot,
        [],
        git_head=lambda _worktree: "a" * 40,
    )

    assert rows == [
        {
            "paper_id": "available-paper",
            "title": "Paper available-paper",
            "claim_count": 2,
            "existing_projects": [],
            "authority": "research-required",
        }
    ]


def append_worker(
    paths,
    session_id: str,
    attempt_id: str,
    paper_id: str,
    work_kind: str,
    *,
    queued_at: str,
    launched_at: str,
    exited_at: str,
    launched_ns: int,
    exited_ns: int,
) -> None:
    common = {
        "attempt_id": attempt_id,
        "paper_id": paper_id,
        "work_kind": work_kind,
    }
    telemetry.append_event(
        paths,
        session_id,
        0,
        "worker-queued",
        {**common, "observed_at": queued_at},
    )
    telemetry.append_event(
        paths,
        session_id,
        1,
        "worker-launched",
        {**common, "observed_at": launched_at, "monotonic_ns": launched_ns},
    )
    telemetry.append_event(
        paths,
        session_id,
        2,
        "worker-exited",
        {**common, "observed_at": exited_at, "monotonic_ns": exited_ns},
    )


def append_stage(
    paths,
    session_id: str,
    attempt_id: str,
    stage: str,
    *,
    started_at: str,
    elapsed_seconds: float,
    outcome: str,
) -> None:
    telemetry.append_event(
        paths,
        session_id,
        0,
        "stage-started",
        {
            "attempt_id": attempt_id,
            "stage": stage,
            "observed_at": started_at,
            "monotonic_ns": 1,
        },
    )
    telemetry.append_event(
        paths,
        session_id,
        1,
        "stage-finished",
        {
            "observed_at": started_at,
            "monotonic_ns": 2,
            "elapsed_seconds": elapsed_seconds,
            "outcome": outcome,
        },
    )


def append_submission(
    paths, session_id: str, attempt_id: str, observed_at: str
) -> None:
    telemetry.append_event(
        paths,
        session_id,
        0,
        "observation",
        {
            "name": "submission-observed",
            "attempt_id": attempt_id,
            "snapshot_id": "snapshot-a",
            "attestation_id": "a" * 64,
            "observed_at": observed_at,
        },
    )


def test_official_points_use_first_judged_logbook_per_user_and_paper():
    snapshot = {
        "verdicts": [
            {
                "space_id": "wrice/first",
                "paper_id": "paper-a",
                "judged_at": "2026-07-27T01:00:00+00:00",
                "claims": [
                    {"verdict": "verified"},
                    {"verdict": "toy"},
                    {"verdict": "inconclusive"},
                ],
            },
            {
                "space_id": "wrice/later",
                "paper_id": "paper-a",
                "judged_at": "2026-07-27T02:00:00+00:00",
                "claims": [{"verdict": "verified"}, {"verdict": "verified"}],
            },
        ]
    }

    assert score_report.official_points(snapshot, "wrice") == {
        "points": 3,
        "max_points": 6,
        "judged_papers": 1,
    }


def test_official_points_reject_unknown_canonical_verdict_status():
    snapshot = {
        "verdicts": [
            {
                "space_id": "wrice/repro",
                "paper_id": "paper-a",
                "judged_at": "2026-07-27T01:00:00+00:00",
                "claims": [{"verdict": "partial"}],
            }
        ]
    }

    with pytest.raises(ValueError, match="status"):
        score_report.official_points(snapshot, "wrice")


def test_queue_labels_estimates_and_orders_by_priority():
    snapshot = {"candidates": [candidate("slow"), candidate("fast")]}
    snapshot["candidates"][0]["score_rate"]["remaining_hours_p90"] = 4.0
    snapshot["candidates"][1]["score_rate"]["remaining_hours_p90"] = 1.0

    queue = score_report.candidate_queue(snapshot)

    assert [row["paper_id"] for row in queue] == ["fast", "slow"]
    assert all(row["authority"] == "estimate" for row in queue)
    assert queue[0]["primary_risk"]


@pytest.mark.parametrize(
    "observation",
    [
        {
            "observed_at": "2026-07-27T03:00:00+00:00",
            "source_url": "https://huggingface.co/spaces/leaderboard",
            "username": "someone-else",
            "points": 2,
            "rank": 1,
        },
        {
            "observed_at": "2026-07-27T03:00:00+00:00",
            "source_url": "https://huggingface.co/spaces/leaderboard",
            "username": "wrice",
            "points": 99,
            "rank": 1,
        },
    ],
    ids=["username", "points"],
)
def test_rank_observation_must_match_snapshot_identity_and_points(
    tmp_path: Path, observation: dict
):
    paths = report_paths(tmp_path, [])
    snapshot = {
        "snapshot_id": "snapshot-a",
        "candidates": [],
        "verdicts": [
            {
                "space_id": "wrice/repro",
                "paper_id": "paper-a",
                "judged_at": "2026-07-27T01:00:00+00:00",
                "claims": [{"verdict": "verified"}],
            }
        ],
    }

    with pytest.raises(ValueError, match="rank_observation"):
        score_report.build_report(paths, snapshot, "wrice", observation)


def test_build_report_uses_only_explicit_attempt_and_telemetry_evidence(
    tmp_path: Path,
):
    records = [
        attempt("a-submitted", "paper-a", "submitted", expected=1.0),
        attempt("a-judging", "paper-b", "judging", expected=0.5),
        attempt("a-implementing", "paper-c", "implementing"),
        attempt("a-validated", "paper-d", "validated"),
        attempt(
            "a-blocked",
            "paper-e",
            "blocked",
            blocker="Upstream artifact unavailable.",
            blocked_from="implementing",
        ),
    ]
    paths = report_paths(tmp_path, records)
    append_worker(
        paths,
        "worker-implementation",
        "a-submitted",
        "paper-a",
        "implementation",
        queued_at="2026-07-27T01:00:00+00:00",
        launched_at="2026-07-27T01:00:02+00:00",
        exited_at="2026-07-27T01:00:12+00:00",
        launched_ns=11_000_000_000,
        exited_ns=11_000_000_000,
    )
    append_worker(
        paths,
        "worker-correction",
        "a-judging",
        "paper-b",
        "correction",
        queued_at="2026-07-27T02:00:00+00:00",
        launched_at="2026-07-27T02:00:03+00:00",
        exited_at="2026-07-27T02:00:09+00:00",
        launched_ns=2_000_000_000,
        exited_ns=8_000_000_000,
    )
    telemetry.append_event(
        paths,
        "worker-open",
        0,
        "worker-queued",
        {
            "attempt_id": "a-implementing",
            "paper_id": "paper-c",
            "work_kind": "implementation",
            "observed_at": "2026-07-27T03:00:00+00:00",
        },
    )
    telemetry.append_event(
        paths,
        "worker-open",
        1,
        "worker-launched",
        {
            "attempt_id": "a-implementing",
            "paper_id": "paper-c",
            "work_kind": "implementation",
            "observed_at": "2026-07-27T03:00:00+00:00",
            "monotonic_ns": 1_000_000_000,
        },
    )
    append_stage(
        paths,
        "validation-first-failed",
        "a-implementing",
        "validation",
        started_at="2026-07-27T03:10:00+00:00",
        elapsed_seconds=4.0,
        outcome="failed",
    )
    append_stage(
        paths,
        "validation-second-passed",
        "a-implementing",
        "validation",
        started_at="2026-07-27T03:20:00+00:00",
        elapsed_seconds=6.0,
        outcome="passed",
    )
    append_stage(
        paths,
        "validation-other-passed",
        "a-validated",
        "validation",
        started_at="2026-07-27T03:15:00+00:00",
        elapsed_seconds=2.0,
        outcome="passed",
    )
    append_stage(
        paths,
        "deployment-passed",
        "a-validated",
        "deployment",
        started_at="2026-07-27T03:30:00+00:00",
        elapsed_seconds=5.0,
        outcome="passed",
    )
    append_submission(
        paths,
        "submission-a",
        "a-submitted",
        "2026-07-27T01:00:22+00:00",
    )
    append_submission(
        paths,
        "submission-b",
        "a-judging",
        "2026-07-27T02:00:33+00:00",
    )
    snapshot = {
        "snapshot_id": "snapshot-a",
        "candidates": [candidate("candidate-a", expected=1.0, hours=1.0)],
        "verdicts": [
            {
                "space_id": "wrice/repro-a",
                "paper_id": "paper-a",
                "judged_at": "2026-07-27T04:00:00+00:00",
                "claims": [{"verdict": "verified"}],
            },
            {
                "space_id": "wrice/repro-b",
                "paper_id": "paper-b",
                "judged_at": "2026-07-27T04:01:00+00:00",
                "claims": [{"verdict": "toy"}],
            },
        ],
    }
    rank = {
        "observed_at": "2026-07-27T05:00:00+00:00",
        "source_url": "https://huggingface.co/spaces/leaderboard",
        "username": "wrice",
        "points": 3,
        "rank": 7,
    }

    report = score_report.build_report(paths, snapshot, "wrice", rank)

    assert set(report) == {
        "official",
        "pending_judgment",
        "capacity",
        "phases",
        "awaiting_validation",
        "awaiting_deployment",
        "blockers",
        "candidate_queue",
        "telemetry",
    }
    assert report["official"] == {
        "authority": "official-verdict-snapshot",
        "snapshot_id": "snapshot-a",
        "username": "wrice",
        "points": 3,
        "max_points": 4,
        "judged_papers": 2,
        "rank_observation": rank,
    }
    assert report["pending_judgment"] == {
        "authority": "estimate",
        "papers": [],
        "expected_points": 0,
    }
    assert report["capacity"] == {"max_runnable": 20, "runnable": 4, "idle": 16}
    assert report["phases"] == {
        "blocked": 1,
        "implementing": 1,
        "judging": 1,
        "submitted": 1,
        "validated": 1,
    }
    assert report["awaiting_validation"] == [
        {
            "attempt_id": "a-implementing",
            "paper_id": "paper-c",
            "phase": "implementing",
        }
    ]
    assert report["awaiting_deployment"] == [
        {
            "attempt_id": "a-validated",
            "paper_id": "paper-d",
            "phase": "validated",
        }
    ]
    assert report["blockers"] == [
        {
            "attempt_id": "a-blocked",
            "paper_id": "paper-e",
            "blocked_from": "implementing",
            "blocker": "Upstream artifact unavailable.",
        }
    ]
    assert report["telemetry"] == {
        "worker_queue_seconds": 5.0,
        "worker_process_seconds": 6.0,
        "implementation_sessions": 2,
        "correction_sessions": 1,
        "validation_seconds": 12.0,
        "deployment_seconds": 5.0,
        "first_launch_to_submission_seconds": 50.0,
        "first_pass_validation_rate": 0.5,
        "judged_points_per_worker_hour": pytest.approx(1800.0),
        "judged_points_per_end_to_end_hour": pytest.approx(216.0),
        "open_sessions": 1,
    }


def test_missing_complete_intervals_return_none_instead_of_phase_or_git_guesses(
    tmp_path: Path,
):
    paths = report_paths(
        tmp_path,
        [attempt("a-complete", "paper-a", "complete")],
    )
    snapshot = {
        "snapshot_id": "snapshot-a",
        "candidates": [],
        "verdicts": [
            {
                "space_id": "wrice/repro",
                "paper_id": "paper-a",
                "judged_at": "2026-07-27T01:00:00+00:00",
                "claims": [{"verdict": "verified"}],
            }
        ],
    }

    metrics = score_report.build_report(paths, snapshot, "wrice")["telemetry"]

    assert metrics == {
        "worker_queue_seconds": None,
        "worker_process_seconds": None,
        "implementation_sessions": 0,
        "correction_sessions": 0,
        "validation_seconds": None,
        "deployment_seconds": None,
        "first_launch_to_submission_seconds": None,
        "first_pass_validation_rate": None,
        "judged_points_per_worker_hour": None,
        "judged_points_per_end_to_end_hour": None,
        "open_sessions": 0,
    }


def test_pending_estimates_exclude_judged_and_nonjudgment_phases(tmp_path: Path):
    paths = report_paths(
        tmp_path,
        [
            attempt("a-submitted", "paper-a", "submitted", expected=1.0),
            attempt("a-judging", "paper-b", "judging", expected=0.5),
            attempt("a-validated", "paper-c", "validated", expected=0.25),
        ],
    )
    snapshot = {
        "snapshot_id": "snapshot-a",
        "candidates": [],
        "verdicts": [
            {
                "space_id": "wrice/repro-a",
                "paper_id": "paper-a",
                "judged_at": "2026-07-27T04:00:00+00:00",
                "claims": [{"verdict": "verified"}],
            }
        ],
    }

    pending = score_report.build_report(paths, snapshot, "wrice")[
        "pending_judgment"
    ]

    assert pending == {
        "authority": "estimate",
        "papers": [
            {
                "attempt_id": "a-judging",
                "paper_id": "paper-b",
                "phase": "judging",
                "expected_points": 0.5,
                "authority": "estimate",
            }
        ],
        "expected_points": 0.5,
    }


def test_telemetry_requires_known_attempt_and_consistent_worker_identity(
    tmp_path: Path,
):
    paths = report_paths(
        tmp_path,
        [attempt("a-complete", "paper-a", "complete")],
    )
    telemetry.append_event(
        paths,
        "mismatched-worker",
        0,
        "worker-queued",
        {
            "attempt_id": "a-complete",
            "paper_id": "paper-b",
            "work_kind": "implementation",
            "observed_at": "2026-07-27T01:00:00+00:00",
        },
    )
    telemetry.append_event(
        paths,
        "mismatched-worker",
        1,
        "worker-launched",
        {
            "attempt_id": "a-complete",
            "paper_id": "paper-a",
            "work_kind": "implementation",
            "observed_at": "2026-07-27T01:00:01+00:00",
            "monotonic_ns": 1_000_000_000,
        },
    )
    telemetry.append_event(
        paths,
        "mismatched-worker",
        2,
        "worker-exited",
        {
            "attempt_id": "a-complete",
            "paper_id": "paper-a",
            "work_kind": "implementation",
            "observed_at": "2026-07-27T01:00:02+00:00",
            "monotonic_ns": 2_000_000_000,
        },
    )
    append_stage(
        paths,
        "unknown-validation",
        "unknown-attempt",
        "validation",
        started_at="2026-07-27T02:00:00+00:00",
        elapsed_seconds=3.0,
        outcome="passed",
    )
    snapshot = {
        "snapshot_id": "snapshot-a",
        "candidates": [],
        "verdicts": [
            {
                "space_id": "wrice/repro-a",
                "paper_id": "paper-a",
                "judged_at": "2026-07-27T04:00:00+00:00",
                "claims": [{"verdict": "verified"}],
            }
        ],
    }

    metrics = score_report.build_report(paths, snapshot, "wrice")["telemetry"]

    assert metrics["worker_queue_seconds"] is None
    assert metrics["worker_process_seconds"] is None
    assert metrics["implementation_sessions"] == 0
    assert metrics["validation_seconds"] is None
    assert metrics["first_pass_validation_rate"] is None
    assert metrics["judged_points_per_worker_hour"] is None
