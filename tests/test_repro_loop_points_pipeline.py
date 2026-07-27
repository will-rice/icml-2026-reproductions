"""End-to-end behavior for the leaderboard-points operating loop."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


SCRIPTS = (
    Path(__file__).resolve().parents[1] / "skills" / "icml-repro-loop" / "scripts"
)
sys.path.insert(0, str(SCRIPTS))
refresh = importlib.import_module("refresh")
scheduler = importlib.import_module("scheduler")
score_report = importlib.import_module("score_report")
store = importlib.import_module("store")
telemetry = importlib.import_module("telemetry")
worker_guard = importlib.import_module("worker_guard")

OBSERVED_AT = "2026-07-27T12:00:00+00:00"
NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


class LocalHub:
    """Serve complete challenge artifacts without network access."""

    def __init__(self, root: Path):
        papers = [
            {"orid": "paper-fast", "title": "Fast Paper"},
            {"orid": "paper-slow", "title": "Slow Paper"},
        ]
        documents = {
            (refresh.CHALLENGE_REPO, "index.json"): {"papers": papers},
            (refresh.CHALLENGE_REPO, "challenge.json"): {
                "papers": papers,
                "claims": {
                    paper["orid"]: [
                        {
                            "text": f"{paper['orid']} claim {number}",
                            "status": "extracted",
                        }
                        for number in (1, 2)
                    ]
                    for paper in papers
                },
            },
            (refresh.VERDICTS_REPO, "verdicts.json"): {},
        }
        self.files = {}
        for (repo_id, filename), document in documents.items():
            path = root / repo_id.rsplit("/", 1)[-1] / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document), encoding="utf-8")
            self.files[(repo_id, filename)] = path

    def dataset_info(self, repo_id, *, revision=None):
        assert revision == "main"
        revision = (
            "challenge-revision"
            if repo_id == refresh.CHALLENGE_REPO
            else "verdicts-revision"
        )
        return SimpleNamespace(id=repo_id, sha=revision)

    def hf_hub_download(self, *, repo_id, filename, repo_type, revision):
        assert repo_type == "dataset"
        return str(self.files[(repo_id, filename)])

    def list_spaces(self, **kwargs):
        assert kwargs == {"filter": "icml2026-repro", "full": True}
        return []


def _score_rate(paper_id: str, *, p_verified: float, hours: float) -> dict:
    return {
        "claim_expectations": [
            {
                "challenge_claim_sha256": hashlib.sha256(
                    f"{paper_id} claim {number}".encode()
                ).hexdigest(),
                "p_verified": p_verified,
                "p_falsified": 0.0,
                "p_toy": 0.0,
            }
            for number in (1, 2)
        ],
        "judged_before_deadline_probability": 0.9,
        "remaining_hours_p90": hours,
        "reusable_implementation": paper_id == "paper-fast",
        "direct_artifact_score": 4,
        "full_score_claim_paths": 2,
        "remaining_time_variance_hours2": 0.25,
        "primary_risk": "The released artifact schema may have drifted.",
    }


def _assessment(paper_id: str, *, p_verified: float, hours: float) -> dict:
    target_claims = [f"{paper_id}-target-{number}" for number in (1, 2)]
    claim_texts = [f"{paper_id} claim {number}" for number in (1, 2)]
    return {
        "paper_id": paper_id,
        "score": 1 if paper_id == "paper-fast" else 100,
        "target_claims": target_claims,
        "claim_bindings": [
            {
                "target_claim": target,
                "challenge_claim": claim,
                "challenge_claim_sha256": hashlib.sha256(
                    claim.encode()
                ).hexdigest(),
            }
            for target, claim in zip(target_claims, claim_texts, strict=True)
        ],
        "upstream_revision": f"revision-{paper_id}",
        "artifact_access": True,
        "cpu_only": True,
        "safety_blocker": None,
        "licensing_blocker": None,
        "estimated_api_cost_usd": 0.0,
        "score_rate": _score_rate(
            paper_id, p_verified=p_verified, hours=hours
        ),
    }


def _write_attempt(paths, index, record: dict) -> None:
    store.atomic_json_write(
        paths.attempt(record["attempt_id"]), record, store.validate_attempt
    )
    index["attempts"][record["attempt_id"]] = {
        "paper_id": record["paper_id"],
        "path": str(
            paths.attempt(record["attempt_id"]).relative_to(paths.index.parent)
        ),
        "phase": record["phase"],
        "updated_at": record["updated_at"],
    }


class CompletedProcess:
    pid = 4242

    def wait(self, timeout=None):
        assert timeout == 30
        return 0


def _next(values):
    iterator = iter(values)
    return lambda: next(iterator)


def test_points_pipeline_preserves_authority_and_releases_nonrunnable_capacity(
    tmp_path: Path,
):
    paths = store.StatePaths(tmp_path / "state" / "repro-loop.json")
    index = store.new_index()
    for number in range(18):
        _write_attempt(
            paths,
            index,
            {
                "attempt_id": f"occupied-{number:02d}",
                "paper_id": f"occupied-paper-{number:02d}",
                "phase": "selected",
                "updated_at": OBSERVED_AT,
            },
        )
    _write_attempt(
        paths,
        index,
        {
            "attempt_id": "already-submitted",
            "paper_id": "paper-submitted",
            "phase": "submitted",
            "updated_at": OBSERVED_AT,
            "live_claims": [
                {
                    "text": f"paper-submitted claim {number}",
                    "status": "extracted",
                }
                for number in (1, 2)
            ],
            "score_rate": _score_rate(
                "paper-submitted", p_verified=0.125, hours=1.0
            ),
        },
    )
    _write_attempt(
        paths,
        index,
        {
            "attempt_id": "already-judging",
            "paper_id": "paper-pending",
            "phase": "judging",
            "updated_at": OBSERVED_AT,
            "live_claims": [
                {"text": "paper-pending claim 1", "status": "extracted"}
            ],
            "score_rate": {
                **_score_rate(
                    "paper-pending", p_verified=0.25, hours=1.0
                ),
                "claim_expectations": [
                    {
                        "challenge_claim_sha256": hashlib.sha256(
                            b"paper-pending claim 1"
                        ).hexdigest(),
                        "p_verified": 0.25,
                        "p_falsified": 0.0,
                        "p_toy": 0.0,
                    }
                ],
                "full_score_claim_paths": 1,
            },
        },
    )
    _write_attempt(
        paths,
        index,
        {
            "attempt_id": "already-blocked",
            "paper_id": "paper-blocked",
            "phase": "blocked",
            "updated_at": OBSERVED_AT,
            "blocked_from": "implementing",
            "blocker": "Required artifact is temporarily unavailable.",
        },
    )
    store.atomic_json_write(paths.index, index, store.validate_index)

    client = LocalHub(tmp_path / "hub")
    raw = refresh.fetch_live_snapshot(client, OBSERVED_AT)
    raw_id = refresh.persist_snapshot(paths, raw)
    raw_snapshot = refresh.read_snapshot(paths, raw_id)
    census = score_report.candidate_census(paths, raw_snapshot, [])

    assert [row["paper_id"] for row in census] == [
        "paper-fast",
        "paper-slow",
    ]
    assert all(row["authority"] == "research-required" for row in census)

    assessment_path = tmp_path / "candidate-assessments.json"
    assessment_path.write_text(
        json.dumps(
            {
                "challenge_revision": "challenge-revision",
                "assessor": "source-reviewer",
                "assessed_at": OBSERVED_AT,
                "assessments": [
                    _assessment(
                        "paper-fast", p_verified=0.75, hours=1.0
                    ),
                    _assessment(
                        "paper-slow", p_verified=0.25, hours=4.0
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    assessed = refresh.fetch_live_snapshot(
        client, OBSERVED_AT, refresh.load_assessments(assessment_path)
    )
    assessed_id = refresh.persist_snapshot(paths, assessed)
    assessed_snapshot = refresh.read_snapshot(paths, assessed_id)

    assert assessed_snapshot["assessments"]["matched_paper_ids"] == [
        "paper-fast",
        "paper-slow",
    ]
    assert all(
        binding["challenge_claim_sha256"]
        == hashlib.sha256(binding["challenge_claim"].encode()).hexdigest()
        for candidate in assessed_snapshot["candidates"]
        for binding in candidate["claim_bindings"]
    )

    admission = scheduler.scheduler_pass(paths, assessed_id, NOW)

    # Submitted, judging, and blocked attempts do not consume implementation
    # slots.
    assert admission.paper_ids == ("paper-fast", "paper-slow")
    fast_assignment = admission.assignments[0]

    worktree = tmp_path / "worker"
    worktree.mkdir()
    contract = worktree / "worker-contract.json"
    contract.write_text('{"scope":"paper-fast"}\n', encoding="utf-8")
    plan = worktree / "paper-fast-plan.md"
    plan.write_text("# Paper Fast Plan\n", encoding="utf-8")
    worker_guard.run_worker(
        paths,
        worker_guard.LaunchSpec(
            runtime="codex",
            argv=("codex", "exec", "--model", "model-a"),
            cwd=worktree,
            env={},
            contract=contract,
            plan=plan,
            mode="implementation",
            attempt_id=fast_assignment.attempt_id,
            paper_id="paper-fast",
            project_path="submissions/fast-paper",
        ),
        timeout_seconds=30,
        process_factory=lambda *args, **kwargs: CompletedProcess(),
        utc_now=_next(
            [
                "2026-07-27T12:00:00+00:00",
                "2026-07-27T12:00:05+00:00",
                "2026-07-27T12:00:11+00:00",
            ]
        ),
        monotonic_ns=_next([1_000_000_000, 7_000_000_000]),
        session_id_factory=lambda: "worker-session",
        git_head=lambda _path: "a" * 40,
    )
    telemetry.run_stage(
        paths,
        fast_assignment.attempt_id,
        "validation",
        lambda: {"attestation_id": "b" * 64},
        utc_now=_next(
            [
                "2026-07-27T12:01:00+00:00",
                "2026-07-27T12:01:03+00:00",
            ]
        ),
        monotonic_ns=_next([10_000_000_000, 13_000_000_000]),
        session_id_factory=lambda: "validation-session",
    )
    telemetry.run_stage(
        paths,
        fast_assignment.attempt_id,
        "deployment",
        lambda: {"attestation_id": "c" * 64},
        utc_now=_next(
            [
                "2026-07-27T12:02:00+00:00",
                "2026-07-27T12:02:04+00:00",
            ]
        ),
        monotonic_ns=_next([20_000_000_000, 24_000_000_000]),
        session_id_factory=lambda: "deployment-session",
    )

    final_payload = {
        key: copy.deepcopy(value)
        for key, value in assessed_snapshot.items()
        if key != "snapshot_id"
    }
    final_payload["verdicts"] = [
        {
            "space_id": "wrice/repro-fast",
            "paper_id": "paper-fast",
            "judged_at": "2026-07-27T12:03:00+00:00",
            "claims": [
                {"verdict": "verified"},
                {"verdict": "toy"},
            ],
        }
    ]
    final_id = refresh.persist_snapshot(paths, final_payload)
    report_snapshot = refresh.read_snapshot(paths, final_id)
    assert final_id == refresh.canonical_snapshot_id(final_payload)
    assert final_id != assessed_id
    rank = {
        "observed_at": "2026-07-27T12:04:00+00:00",
        "source_url": "https://example.test/leaderboard",
        "username": "wrice",
        "points": 3,
        "rank": 7,
    }
    before_report = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    report = score_report.build_report(
        paths, report_snapshot, "wrice", rank
    )
    after_report = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert before_report == after_report
    assert report["official"] == {
        "authority": "official-verdict-snapshot",
        "snapshot_id": final_id,
        "username": "wrice",
        "points": 3,
        "max_points": 4,
        "judged_papers": 1,
        "rank_observation": rank,
    }
    assert report["pending_judgment"] == {
        "authority": "estimate",
        "papers": [
            {
                "attempt_id": "already-judging",
                "paper_id": "paper-pending",
                "phase": "judging",
                "expected_points": 0.5,
                "authority": "estimate",
            },
            {
                "attempt_id": "already-submitted",
                "paper_id": "paper-submitted",
                "phase": "submitted",
                "expected_points": 0.5,
                "authority": "estimate",
            },
        ],
        "expected_points": 1.0,
    }
    assert [row["authority"] for row in report["candidate_queue"]] == [
        "estimate",
        "estimate",
    ]
    assert report["candidate_queue"][0]["paper_id"] == "paper-fast"
    assert (
        report["candidate_queue"][0]["authority"]
        != report["official"]["authority"]
    )
    assert report["capacity"] == {
        "max_runnable": 20,
        "runnable": 20,
        "idle": 0,
    }
    assert report["telemetry"]["worker_queue_seconds"] == 5.0
    assert report["telemetry"]["worker_process_seconds"] == 6.0
    assert report["telemetry"]["implementation_sessions"] == 1
    assert report["telemetry"]["correction_sessions"] == 0
    assert report["telemetry"]["validation_seconds"] == 3.0
    assert report["telemetry"]["deployment_seconds"] == 4.0
    assert report["telemetry"]["first_launch_to_submission_seconds"] is None
    assert score_report.build_report(paths, report_snapshot, "wrice")[
        "official"
    ]["rank_observation"] is None
