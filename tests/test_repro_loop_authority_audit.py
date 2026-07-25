"""Authority classification, quarantine preservation, and repair tests."""

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
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(SCRIPTS))

import attestations
import authority_audit
import leases
import refresh
import store


NOW = datetime(2026, 7, 25, 20, 0, tzinfo=timezone.utc)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(name: str) -> dict:
    return json.loads(
        (FIXTURES / name).read_text(encoding="utf-8")
    )


def clone_attempt(template: dict, attempt_id: str, paper_id: str) -> dict:
    value = copy.deepcopy(template)
    value["attempt_id"] = attempt_id
    value["paper_id"] = paper_id
    value["space_id"] = f"wrice/repro-{paper_id}"
    value["deployed_sha"] = hashlib.sha1(
        attempt_id.encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return value


def record_payload(kind: str, attempt: dict, **updates: object) -> dict:
    if kind == "validation":
        payload = {
            "worktree": f"/tmp/{attempt['paper_id']}",
            "branch": attempt["paper_id"],
            "base_sha": "1" * 40,
            "project_path": f"submissions/{attempt['paper_id']}",
            "design_path": f"docs/designs/{attempt['paper_id']}.md",
            "commands": [],
            "checks": [],
            "environment": [],
            "source_tree": "2" * 40,
            "source_tree_sha256": "3" * 64,
            "environment_sha256": "4" * 64,
        }
    elif kind == "deployment":
        payload = {
            "space_id": attempt["space_id"],
            "space_sha": attempt["deployed_sha"],
            "owner": "wrice",
            "tags": [
                "icml2026-repro",
                f"paper-{attempt['paper_id']}",
            ],
            "runtime_stage": "RUNNING",
            "validation_attestation_id": "5" * 64,
            "source_tree_sha256": "3" * 64,
        }
    else:
        raise AssertionError(kind)
    payload.update(updates)
    return {
        "kind": kind,
        "attempt_id": attempt["attempt_id"],
        "attempt_number": 1,
        "observed_at": (NOW - timedelta(hours=1)).isoformat(),
        "source_commit": "6" * 40,
        "payload_sha256": canonical_sha256(payload),
        **payload,
    }


def persist_slot(
    paths: store.StatePaths,
    record: dict,
) -> str:
    attestation_id = attestations.persist(paths, record)
    persisted = attestations.read(paths, attestation_id)
    path = paths.attestation(
        record["kind"],
        record["attempt_id"],
        record["attempt_number"],
    )
    store.atomic_json_write(
        path,
        persisted,
        lambda value: attestations.validate_target(paths, path, value),
    )
    return attestation_id


def transition_record(
    source: str,
    target: str,
    *,
    attestation_id: str | None = None,
) -> dict:
    record = {
        "from": source,
        "to": target,
        "at": (NOW - timedelta(hours=1)).isoformat(),
        "owner": "controller",
        "fencing_token": 1,
        "snapshot_id": "selection-snapshot",
    }
    if attestation_id is not None:
        record["attestation_id"] = attestation_id
    return record


def write_attempt(
    paths: store.StatePaths,
    index: dict,
    attempt: dict,
) -> None:
    store.atomic_json_write(
        paths.attempt(attempt["attempt_id"]),
        attempt,
        store.validate_attempt,
    )
    index["history"][attempt["attempt_id"]] = {
        "path": str(
            paths.attempt(attempt["attempt_id"]).relative_to(
                paths.index.parent
            )
        ),
        "paper_id": attempt["paper_id"],
        "phase": attempt["phase"],
        "updated_at": attempt["updated_at"],
    }
    judgment = {
        "attempt_id": attempt["attempt_id"],
        "paper_id": attempt["paper_id"],
        "created_at": attempt["updated_at"],
        "worker_authored_verdict": copy.deepcopy(attempt.get("verdict")),
    }
    store.atomic_json_write(
        paths.judgment(attempt["attempt_id"]),
        judgment,
        store.validate_judgment,
    )


def official_verdict(attempt: dict) -> dict:
    return {
        "space_id": attempt["space_id"],
        "paper_id": attempt["paper_id"],
        "source_revision": "official-revision",
        "orid": attempt["paper_id"],
        "sha": attempt["deployed_sha"],
        "judged_at": (NOW - timedelta(minutes=10)).isoformat(),
        "claims": [
            {
                "claim": "Challenge claim 1",
                "verdict": "toy",
                "evidence": "bounded run",
            },
            {
                "claim": "Challenge claim 2",
                "verdict": "inconclusive",
                "evidence": "missing artifact",
            },
        ],
    }


@pytest.fixture
def authority_case(tmp_path: Path):
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    index = store.new_index()
    index["total_api_cost_usd"] = 1.25
    forged = fixture("forged-completion.json")
    official = fixture("official-completion.json")
    live = clone_attempt(forged, "a-live", "paper-live")
    deployed = clone_attempt(forged, "a-deployed", "paper-deployed")
    validated = clone_attempt(forged, "a-validated", "paper-validated")
    fake_validated = clone_attempt(
        forged, "a-fake-validated", "paper-fake-validated"
    )
    wrong_sha = clone_attempt(official, "a-wrong-sha", "paper-wrong-sha")

    validation_id = persist_slot(
        paths, record_payload("validation", validated)
    )
    validated["transitions"].insert(
        0,
        transition_record(
            "implementing", "validated", attestation_id=validation_id
        ),
    )
    deployed_validation_id = persist_slot(
        paths, record_payload("validation", deployed)
    )
    deployment_record = record_payload(
        "deployment",
        deployed,
        validation_attestation_id=deployed_validation_id,
    )
    deployment_id = persist_slot(paths, deployment_record)
    deployed["transitions"] = [
        transition_record(
            "implementing",
            "validated",
            attestation_id=deployed_validation_id,
        ),
        transition_record(
            "validated",
            "deployed",
            attestation_id=deployment_id,
        ),
        *deployed["transitions"],
    ]
    fake_validated["transitions"].insert(
        0,
        transition_record(
            "implementing",
            "validated",
            attestation_id="7" * 64,
        ),
    )

    for attempt in (
        forged,
        official,
        live,
        deployed,
        validated,
        fake_validated,
        wrong_sha,
    ):
        write_attempt(paths, index, attempt)
    store.atomic_json_write(paths.index, index, store.validate_index)

    snapshot = {
        "fetched_at": NOW.isoformat(),
        "source_revision": "8" * 64,
        "sources": {
            "challenge": {"revision": "challenge-revision"},
            "verdicts": {"revision": "official-revision"},
        },
        "assessments": None,
        "candidates": [],
        "queued_submissions": [],
        "tagged_spaces": [
            {
                "paper_id": live["paper_id"],
                "space_id": live["space_id"],
                "revision": live["deployed_sha"],
            }
        ],
        "verdicts": [
            official_verdict(official),
            {
                **official_verdict(wrong_sha),
                "sha": "0" * 40,
            },
        ],
        "spaces": [
            {
                "paper_ids": [live["paper_id"]],
                "space_id": live["space_id"],
                "revision": live["deployed_sha"],
                "tags": [
                    "icml2026-repro",
                    f"paper-{live['paper_id']}",
                ],
            }
        ],
    }
    snapshot_id = refresh.persist_snapshot(paths, snapshot)
    return {
        "paths": paths,
        "snapshot_id": snapshot_id,
        "attempts": {
            attempt["attempt_id"]: attempt
            for attempt in (
                forged,
                official,
                live,
                deployed,
                validated,
                fake_validated,
                wrong_sha,
            )
        },
    }


def decisions_by_id(report: dict) -> dict[str, dict]:
    return {
        decision["attempt_id"]: decision
        for decision in report["decisions"]
    }


def test_audit_classifies_only_exact_official_completion_and_proven_phases(
    authority_case,
):
    paths = authority_case["paths"]
    index_before = paths.index.read_bytes()

    report = authority_audit.audit(
        paths,
        authority_case["snapshot_id"],
    )

    decisions = decisions_by_id(report)
    assert decisions["a-official"]["classification"] == "valid-official"
    assert decisions["a-forged"]["classification"] == "unsupported-completion"
    assert decisions["a-forged"]["blocked_from"] == "implementing"
    assert decisions["a-live"]["blocked_from"] == "judging"
    assert decisions["a-deployed"]["blocked_from"] == "deployed"
    assert decisions["a-validated"]["blocked_from"] == "validated"
    assert decisions["a-fake-validated"]["blocked_from"] == "implementing"
    assert decisions["a-wrong-sha"]["blocked_from"] == "implementing"
    assert report["snapshot_id"] == authority_case["snapshot_id"]
    assert report["index_sha256"] == hashlib.sha256(index_before).hexdigest()
    assert report["report_id"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "report_id"}
    )
    assert paths.index.read_bytes() == index_before
    assert not paths.authority_audit(report["report_id"]).exists()


def test_repair_preserves_original_bytes_and_restores_blocked_attempts(
    authority_case,
):
    paths = authority_case["paths"]
    report = authority_audit.audit(paths, authority_case["snapshot_id"])
    invalid_ids = {
        decision["attempt_id"]
        for decision in report["decisions"]
        if decision["classification"] == "unsupported-completion"
    }
    original_attempts = {
        attempt_id: paths.attempt(attempt_id).read_bytes()
        for attempt_id in invalid_ids
    }
    original_judgments = {
        attempt_id: paths.judgment(attempt_id).read_bytes()
        for attempt_id in invalid_ids
    }

    result = authority_audit.repair(paths, report, NOW + timedelta(minutes=1))

    repaired_index = store.read_json(paths.index)
    assert result["mutations"] == len(invalid_ids)
    assert repaired_index["total_api_cost_usd"] == 1.25
    assert set(repaired_index["history"]) == {"a-official"}
    assert invalid_ids <= set(repaired_index["attempts"])
    assert paths.authority_audit(report["report_id"]).exists()
    for attempt_id in invalid_ids:
        repaired = store.read_json(paths.attempt(attempt_id))
        decision = decisions_by_id(report)[attempt_id]
        assert repaired["phase"] == "blocked"
        assert repaired["blocked_from"] == decision["blocked_from"]
        assert repaired["blocker"]
        assert repaired["authority_repair"]["report_id"] == report["report_id"]
        assert "verdict" not in repaired
        assert "verdict_at" not in repaired
        assert "verdict_source_revision" not in repaired
        if attempt_id == "a-forged":
            assert repaired["external_ids"] == {
                "claimed_submission_id": "synthetic-submission"
            }
        manifest = store.read_json(paths.quarantine_manifest(attempt_id))
        copies = {
            entry["source_path"]: entry for entry in manifest["files"]
        }
        attempt_source = str(
            paths.attempt(attempt_id).relative_to(paths.index.parent)
        )
        judgment_source = str(
            paths.judgment(attempt_id).relative_to(paths.index.parent)
        )
        attempt_copy = paths.index.parent / copies[attempt_source]["copy_path"]
        judgment_copy = paths.index.parent / copies[judgment_source]["copy_path"]
        assert attempt_copy.read_bytes() == original_attempts[attempt_id]
        assert judgment_copy.read_bytes() == original_judgments[attempt_id]
        assert copies[attempt_source]["sha256"] == hashlib.sha256(
            original_attempts[attempt_id]
        ).hexdigest()
        assert copies[judgment_source]["sha256"] == hashlib.sha256(
            original_judgments[attempt_id]
        ).hexdigest()


def test_repair_is_byte_idempotent(authority_case):
    paths = authority_case["paths"]
    report = authority_audit.audit(paths, authority_case["snapshot_id"])
    first = authority_audit.repair(paths, report, NOW + timedelta(minutes=1))
    tracked = [
        paths.index,
        paths.authority_audit(report["report_id"]),
        *sorted((paths.root / "attempts").glob("*.json")),
        *sorted((paths.root / "quarantine").rglob("*")),
    ]
    before = {
        path: path.read_bytes() for path in tracked if path.is_file()
    }

    second = authority_audit.repair(
        paths,
        report,
        NOW + timedelta(minutes=2),
    )

    after = {
        path: path.read_bytes() for path in tracked if path.is_file()
    }
    assert first["mutations"] > 0
    assert second["mutations"] == 0
    assert after == before


def test_repair_fences_existing_writer_and_requires_a_fresh_lease(
    authority_case,
):
    paths = authority_case["paths"]
    prior = leases.acquire_lease(
        paths,
        "attempt:a-forged",
        "old-worker",
        "a-forged",
        NOW - timedelta(minutes=5),
        timedelta(hours=2),
    )
    report = authority_audit.audit(paths, authority_case["snapshot_id"])

    authority_audit.repair(paths, report, NOW + timedelta(minutes=1))

    with pytest.raises(leases.StaleFence):
        leases.renew_attempt(
            paths,
            prior,
            NOW + timedelta(minutes=2),
        )
    replacement = leases.claim_attempt(
        paths,
        "a-forged",
        "new-worker",
        prior.fencing_token,
        NOW + timedelta(minutes=2),
    )
    assert replacement.fencing_token == prior.fencing_token + 1


def test_repair_refuses_to_overwrite_different_quarantine_bytes(
    authority_case,
):
    paths = authority_case["paths"]
    report = authority_audit.audit(paths, authority_case["snapshot_id"])
    target = paths.quarantine("a-forged") / "attempt.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"different evidence\n")
    index_before = paths.index.read_bytes()

    with pytest.raises(ValueError, match="quarantine"):
        authority_audit.repair(paths, report, NOW + timedelta(minutes=1))

    assert paths.index.read_bytes() == index_before
    assert store.read_json(paths.attempt("a-forged"))["phase"] == "complete"


def test_repair_rejects_a_rehashed_caller_modified_audit(authority_case):
    paths = authority_case["paths"]
    report = authority_audit.audit(paths, authority_case["snapshot_id"])
    modified = copy.deepcopy(report)
    decision = decisions_by_id(modified)["a-official"]
    decision["classification"] = "unsupported-completion"
    decision["blocked_from"] = "implementing"
    decision["reasons"] = ["caller-authored downgrade"]
    modified["report_id"] = canonical_sha256(
        {
            key: value
            for key, value in modified.items()
            if key != "report_id"
        }
    )
    index_before = paths.index.read_bytes()

    with pytest.raises(ValueError, match="report"):
        authority_audit.repair(
            paths,
            modified,
            NOW + timedelta(minutes=1),
        )

    assert paths.index.read_bytes() == index_before
    assert not paths.authority_audit(modified["report_id"]).exists()


def test_repair_transaction_recovers_attempts_and_index(
    authority_case,
    monkeypatch,
):
    paths = authority_case["paths"]
    report = authority_audit.audit(paths, authority_case["snapshot_id"])
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
        authority_audit.repair(paths, report, NOW + timedelta(minutes=1))
    monkeypatch.setattr(store, "_transaction_write", real_write)

    authority_audit.recover_transactions(paths)

    index = store.read_json(paths.index)
    assert "a-forged" in index["attempts"]
    assert store.read_json(paths.attempt("a-forged"))["phase"] == "blocked"


def test_cli_exposes_dry_run_and_repair_authority_audit(authority_case):
    paths = authority_case["paths"]
    help_result = subprocess.run(
        [sys.executable, str(STATE), "audit-authority", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SCRIPTS)},
    )
    dry_result = subprocess.run(
        [
            sys.executable,
            str(STATE),
            "audit-authority",
            str(paths.index),
            "--snapshot-id",
            authority_case["snapshot_id"],
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SCRIPTS)},
    )

    assert help_result.returncode == 0
    assert "--snapshot-id" in help_result.stdout
    assert "--repair" in help_result.stdout
    assert dry_result.returncode == 0
    assert json.loads(dry_result.stdout)["decisions"]
    assert store.read_json(paths.attempt("a-forged"))["phase"] == "complete"
