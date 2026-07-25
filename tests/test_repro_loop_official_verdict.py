"""Snapshot-only official-verdict import and atomic completion tests."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "icml-repro-loop" / "scripts"
STATE = SCRIPTS / "state.py"
sys.path.insert(0, str(SCRIPTS))

import attestations
import attempts
import controller
import leases
import refresh
import scheduler
import store


NOW = datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc)
PAPER_ID = "paper-a"
SPACE_ID = "wrice/repro-paper-a"
SPACE_SHA = "a" * 40
SOURCE_COMMIT = "b" * 40
VERDICT_REVISION = "verdict-revision-2"
CLAIM_BINDINGS = [
    {
        "target_claim": "claim-a-one",
        "challenge_claim": "Claim A1",
        "challenge_claim_sha256": hashlib.sha256(b"Claim A1").hexdigest(),
    },
    {
        "target_claim": "claim-a-two",
        "challenge_claim": "Claim A2",
        "challenge_claim_sha256": hashlib.sha256(b"Claim A2").hexdigest(),
    },
]


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def persist_attestation(paths: store.StatePaths, kind: str, payload: dict) -> str:
    record = {
        "kind": kind,
        "attempt_id": "a1",
        "attempt_number": 1,
        "observed_at": payload.pop("observed_at"),
        "source_commit": SOURCE_COMMIT,
        "payload_sha256": canonical_sha256(payload),
        **payload,
    }
    return attestations.persist(paths, record)


def validation_payload() -> dict:
    return {
        "observed_at": NOW.isoformat(),
        "worktree": "/tmp/paper-a",
        "branch": "paper-a",
        "base_sha": "c" * 40,
        "project_path": "submissions/paper-a",
        "design_path": "docs/designs/paper-a.md",
        "commands": [],
        "checks": [],
        "environment": [],
        "source_tree": "d" * 40,
        "source_tree_sha256": "e" * 64,
        "environment_sha256": "f" * 64,
    }


def deployment_payload(validation_id: str) -> dict:
    return {
        "observed_at": (NOW + timedelta(minutes=1)).isoformat(),
        "space_id": SPACE_ID,
        "space_sha": SPACE_SHA,
        "owner": "wrice",
        "tags": ["icml2026-repro", f"paper-{PAPER_ID}"],
        "runtime_stage": "RUNNING",
        "validation_attestation_id": validation_id,
        "source_tree_sha256": "e" * 64,
    }


def submission_payload(deployment_id: str) -> dict:
    return {
        "observed_at": (NOW + timedelta(minutes=2)).isoformat(),
        "snapshot_id": "1" * 64,
        "verdict_revision": "verdict-revision-1",
        "space_id": SPACE_ID,
        "space_sha": SPACE_SHA,
        "paper_id": PAPER_ID,
        "queue_status": "pending",
        "deployment_attestation_id": deployment_id,
    }


@pytest.fixture
def submitted_case(tmp_path: Path):
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    store.atomic_json_write(paths.index, store.new_index(), store.validate_index)
    lease = leases.acquire_lease(
        paths,
        "attempt:a1",
        "controller-1",
        "a1",
        NOW,
        timedelta(hours=24),
    )
    attempts.create_attempt(
        paths,
        "a1",
        {
            "paper_id": PAPER_ID,
            "title": "Paper A",
            "slug": "paper-a",
            "project_path": "submissions/paper-a",
            "upstream_revision": "upstream-a",
            "target_claims": ["claim-a-one", "claim-a-two"],
            "claim_bindings": copy.deepcopy(CLAIM_BINDINGS),
            "estimated_api_cost_usd": 0.0,
        },
        lease,
        "selection-snapshot",
        NOW,
    )
    attempts.transition_attempt(paths, "a1", "design-pending", lease, NOW)
    attempts.record_design(
        paths, "a1", lease, "author-a", "docs/designs/paper-a.md", NOW
    )
    attempts.record_design_review(
        paths, "a1", lease, "reviewer-a", "approved", NOW
    )
    validation_id = persist_attestation(paths, "validation", validation_payload())
    attempts.transition_attested(
        paths, "a1", "validated", validation_id, {}, lease, NOW
    )
    deployment_id = persist_attestation(
        paths, "deployment", deployment_payload(validation_id)
    )
    attempts.transition_attested(
        paths,
        "a1",
        "deployed",
        deployment_id,
        {"space_id": SPACE_ID, "deployed_sha": SPACE_SHA},
        lease,
        NOW + timedelta(minutes=1),
    )
    submission_id = persist_attestation(
        paths, "submission", submission_payload(deployment_id)
    )
    attempts.transition_attested(
        paths,
        "a1",
        "submitted",
        submission_id,
        {},
        lease,
        NOW + timedelta(minutes=2),
    )
    return {
        "paths": paths,
        "lease": lease,
        "submission_id": submission_id,
    }


def enter_judging(case: dict) -> dict:
    return scheduler.watch_attempt(
        case["paths"],
        "a1",
        case["lease"],
        12,
        NOW + timedelta(hours=12),
        NOW + timedelta(minutes=3),
    )


def official_verdict() -> dict:
    return {
        "space_id": SPACE_ID,
        "paper_id": PAPER_ID,
        "source_revision": VERDICT_REVISION,
        "orid": PAPER_ID,
        "paper_title": "Paper A",
        "sha": SPACE_SHA,
        "judged_at": (NOW + timedelta(minutes=4)).isoformat(),
        "model": "judge/model",
        "claims": [
            {
                "claim": "Claim A1",
                "verdict": "toy",
                "evidence": "bounded run",
            },
            {
                "claim": "Claim A2",
                "verdict": "inconclusive",
                "evidence": "missing artifact",
            },
        ],
        "overall": "Bounded evidence only.",
        "quality": "low",
    }


def verdict_snapshot(verdict: dict | None = None) -> dict:
    return {
        "fetched_at": (NOW + timedelta(minutes=5)).isoformat(),
        "source_revision": "2" * 64,
        "sources": {
            "challenge": {"revision": "challenge-revision"},
            "verdicts": {"revision": VERDICT_REVISION},
        },
        "assessments": None,
        "candidates": [],
        "queued_submissions": [],
        "tagged_spaces": [
            {
                "paper_id": PAPER_ID,
                "space_id": SPACE_ID,
                "revision": SPACE_SHA,
            }
        ],
        "verdicts": [official_verdict() if verdict is None else verdict],
        "spaces": [
            {
                "paper_ids": [PAPER_ID],
                "space_id": SPACE_ID,
                "revision": SPACE_SHA,
                "tags": ["icml2026-repro", f"paper-{PAPER_ID}"],
            }
        ],
    }


def persist_verdict_snapshot(case: dict, verdict: dict | None = None) -> str:
    return refresh.persist_snapshot(case["paths"], verdict_snapshot(verdict))


def test_public_caller_authored_verdict_interface_is_removed():
    assert not hasattr(scheduler, "record_verdict")

    top_level = subprocess.run(
        [sys.executable, str(STATE), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SCRIPTS)},
    )
    command = subprocess.run(
        [sys.executable, str(STATE), "sync-verdict", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SCRIPTS)},
    )

    assert top_level.returncode == 0
    assert "record-verdict" not in top_level.stdout
    assert "sync-verdict" in top_level.stdout
    assert command.returncode == 0
    assert "--snapshot-id" in command.stdout
    assert "--raw-verdict" not in command.stdout
    assert "--normalized-verdict" not in command.stdout
    assert "--source-revision" not in command.stdout


def test_watch_attempt_atomically_enters_judging_with_bounded_attestation(
    submitted_case,
):
    judgment = enter_judging(submitted_case)
    attempt = attempts.read_attempt(submitted_case["paths"], "a1")
    record = attestations.read(
        submitted_case["paths"], attempt["transitions"][-1]["attestation_id"]
    )

    assert attempt["phase"] == "judging"
    assert judgment["poll_limit"] == 12
    assert judgment["poll_deadline"] == (
        NOW + timedelta(hours=12)
    ).isoformat()
    assert record["kind"] == "authority-audit"
    assert record["submission_attestation_id"] == submitted_case["submission_id"]
    assert record["poll_limit"] == 12
    assert record["poll_deadline"] == judgment["poll_deadline"]
    assert record["space_id"] == SPACE_ID
    assert record["space_sha"] == SPACE_SHA


def test_sync_verdict_copies_exact_official_claims_and_completes_atomically(
    submitted_case,
):
    enter_judging(submitted_case)
    snapshot_id = persist_verdict_snapshot(submitted_case)

    completed = controller.sync_verdict(
        submitted_case["paths"],
        "a1",
        submitted_case["lease"],
        snapshot_id,
        NOW + timedelta(minutes=5),
    )

    expected = {
        "claims": [
            {
                "target_claim": "claim-a-one",
                "claim": "Claim A1",
                "status": "toy",
                "evidence": "bounded run",
            },
            {
                "target_claim": "claim-a-two",
                "claim": "Claim A2",
                "status": "inconclusive",
                "evidence": "missing artifact",
            },
        ]
    }
    index = store.read_json(submitted_case["paths"].index)
    judgment = store.read_json(submitted_case["paths"].judgment("a1"))
    verdict_attestation = attestations.read(
        submitted_case["paths"], completed["transitions"][-1]["attestation_id"]
    )

    assert completed["phase"] == "complete"
    assert completed["verdict"] == expected
    assert "a1" not in index["attempts"]
    assert index["history"]["a1"]["phase"] == "complete"
    assert judgment["raw_verdict"] == official_verdict()
    assert judgment["normalized_verdict"] == expected
    assert judgment["source_revision"] == VERDICT_REVISION
    assert verdict_attestation["kind"] == "verdict"
    assert verdict_attestation["snapshot_id"] == snapshot_id
    assert verdict_attestation["verdict_revision"] == VERDICT_REVISION
    assert verdict_attestation["space_id"] == SPACE_ID
    assert verdict_attestation["space_sha"] == SPACE_SHA
    assert verdict_attestation["paper_id"] == PAPER_ID
    assert verdict_attestation["judged_at"] == official_verdict()["judged_at"]
    assert verdict_attestation["claims"] == expected["claims"]


def test_sync_verdict_selects_only_bound_targets_from_larger_official_verdict(
    submitted_case,
):
    enter_judging(submitted_case)
    verdict = official_verdict()
    verdict["claims"].append(
        {
            "claim": "Unselected challenge claim",
            "verdict": "verified",
            "evidence": "official but outside this reproduction target set",
        }
    )
    snapshot_id = persist_verdict_snapshot(submitted_case, verdict)

    completed = controller.sync_verdict(
        submitted_case["paths"],
        "a1",
        submitted_case["lease"],
        snapshot_id,
        NOW + timedelta(minutes=5),
    )

    assert [
        claim["target_claim"] for claim in completed["verdict"]["claims"]
    ] == ["claim-a-one", "claim-a-two"]


def test_sync_verdict_preserves_empty_official_evidence(submitted_case):
    enter_judging(submitted_case)
    verdict = official_verdict()
    verdict["claims"][0]["evidence"] = ""
    snapshot_id = persist_verdict_snapshot(submitted_case, verdict)

    completed = controller.sync_verdict(
        submitted_case["paths"],
        "a1",
        submitted_case["lease"],
        snapshot_id,
        NOW + timedelta(minutes=5),
    )

    assert completed["verdict"]["claims"][0]["evidence"] == ""


def test_sync_verdict_cannot_precede_latest_poll(submitted_case):
    enter_judging(submitted_case)
    scheduler.record_poll(
        submitted_case["paths"],
        "a1",
        submitted_case["lease"],
        "pending",
        NOW + timedelta(minutes=6),
    )
    snapshot_id = persist_verdict_snapshot(submitted_case)

    with pytest.raises(ValueError, match="now"):
        controller.sync_verdict(
            submitted_case["paths"],
            "a1",
            submitted_case["lease"],
            snapshot_id,
            NOW + timedelta(minutes=5),
        )


def test_poll_cannot_follow_official_verdict(submitted_case):
    enter_judging(submitted_case)
    snapshot_id = persist_verdict_snapshot(submitted_case)
    controller.sync_verdict(
        submitted_case["paths"],
        "a1",
        submitted_case["lease"],
        snapshot_id,
        NOW + timedelta(minutes=5),
    )

    with pytest.raises(ValueError, match="verdict"):
        scheduler.record_poll(
            submitted_case["paths"],
            "a1",
            submitted_case["lease"],
            "late",
            NOW + timedelta(minutes=6),
        )


@pytest.mark.parametrize(
    ("variant", "error"),
    [
        ("missing", "official_verdict"),
        ("space", "space"),
        ("paper", "paper"),
        ("sha", "sha"),
        ("source", "source_revision"),
        ("time", "judged_at"),
        ("claim", "claim"),
        ("binding-hash", "claim"),
        ("status", "verdict"),
    ],
)
def test_sync_verdict_rejects_nonexact_official_source_without_mutation(
    submitted_case,
    variant,
    error,
):
    enter_judging(submitted_case)
    verdict = official_verdict()
    if variant == "missing":
        snapshot = verdict_snapshot()
        snapshot["verdicts"] = []
        snapshot_id = refresh.persist_snapshot(submitted_case["paths"], snapshot)
    else:
        if variant == "space":
            verdict["space_id"] = "wrice/repro-other"
        elif variant == "paper":
            verdict["paper_id"] = "paper-b"
        elif variant == "sha":
            verdict["sha"] = "wrong-sha"
        elif variant == "source":
            verdict["source_revision"] = "wrong-revision"
        elif variant == "time":
            verdict["judged_at"] = (
                NOW + timedelta(minutes=1)
            ).isoformat()
        elif variant == "claim":
            verdict["claims"][0]["claim"] = "Different claim"
        elif variant == "binding-hash":
            attempt = attempts.read_attempt(submitted_case["paths"], "a1")
            attempt["claim_bindings"][0]["challenge_claim_sha256"] = "0" * 64
            store.atomic_json_write(
                submitted_case["paths"].attempt("a1"),
                attempt,
                store.validate_attempt,
            )
        elif variant == "status":
            verdict["claims"][0]["verdict"] = "partial"
        snapshot_id = persist_verdict_snapshot(submitted_case, verdict)
    attempt_before = submitted_case["paths"].attempt("a1").read_bytes()
    judgment_before = submitted_case["paths"].judgment("a1").read_bytes()
    index_before = submitted_case["paths"].index.read_bytes()

    with pytest.raises(ValueError, match=error):
        controller.sync_verdict(
            submitted_case["paths"],
            "a1",
            submitted_case["lease"],
            snapshot_id,
            NOW + timedelta(minutes=5),
        )

    assert submitted_case["paths"].attempt("a1").read_bytes() == attempt_before
    assert submitted_case["paths"].judgment("a1").read_bytes() == judgment_before
    assert submitted_case["paths"].index.read_bytes() == index_before


def test_watch_attempt_transaction_recovers_all_authority_targets(
    submitted_case,
    monkeypatch,
):
    real_write = store._transaction_write
    writes = 0

    def interrupted(path, value, validator):
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("simulated interruption")
        real_write(path, value, validator)

    monkeypatch.setattr(store, "_transaction_write", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        enter_judging(submitted_case)
    monkeypatch.setattr(store, "_transaction_write", real_write)

    attempts.recover_transactions(submitted_case["paths"])

    attempt = attempts.read_attempt(submitted_case["paths"], "a1")
    judgment = store.read_json(submitted_case["paths"].judgment("a1"))
    record = attestations.read(
        submitted_case["paths"], attempt["transitions"][-1]["attestation_id"]
    )
    assert attempt["phase"] == "judging"
    assert judgment["attempt_id"] == "a1"
    assert record["kind"] == "authority-audit"


def test_sync_verdict_transaction_recovers_judgment_attempt_and_history(
    submitted_case,
    monkeypatch,
):
    enter_judging(submitted_case)
    snapshot_id = persist_verdict_snapshot(submitted_case)
    real_write = store._transaction_write
    writes = 0

    def interrupted(path, value, validator):
        nonlocal writes
        writes += 1
        if writes == 4:
            raise OSError("simulated interruption")
        real_write(path, value, validator)

    monkeypatch.setattr(store, "_transaction_write", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        controller.sync_verdict(
            submitted_case["paths"],
            "a1",
            submitted_case["lease"],
            snapshot_id,
            NOW + timedelta(minutes=5),
        )
    monkeypatch.setattr(store, "_transaction_write", real_write)

    attempts.recover_transactions(submitted_case["paths"])

    attempt = attempts.read_attempt(submitted_case["paths"], "a1")
    index = store.read_json(submitted_case["paths"].index)
    judgment = store.read_json(submitted_case["paths"].judgment("a1"))
    assert attempt["phase"] == "complete"
    assert index["history"]["a1"]["phase"] == "complete"
    assert judgment["normalized_verdict"]["claims"][0]["status"] == "toy"
