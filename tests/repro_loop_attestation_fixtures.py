"""Shared strict validation-attestation test values."""

import hashlib
import json


def add_attestation_fields(record: dict) -> None:
    kind = record["kind"]
    if kind == "validation":
        add_validation_fields(record)
    elif kind == "deployment":
        payload = {
            "space_id": "wrice/repro-paper-1",
            "space_sha": "space-sha-1",
            "owner": "wrice",
            "tags": ["icml2026-repro", "paper-paper-1"],
            "runtime_stage": "RUNNING",
            "validation_attestation_id": "8" * 64,
            "source_tree_sha256": "9" * 64,
        }
        _add_payload(record, payload)
    elif kind == "submission":
        payload = {
            "snapshot_id": "a" * 64,
            "verdict_revision": "verdict-revision",
            "space_id": "wrice/repro-paper-1",
            "space_sha": "space-sha-1",
            "paper_id": "paper-1",
            "queue_status": "pending",
            "deployment_attestation_id": "b" * 64,
        }
        _add_payload(record, payload)
    elif kind == "authority-audit":
        payload = {
            "submission_attestation_id": "c" * 64,
            "poll_limit": 12,
            "poll_deadline": "2026-07-25T18:00:00+00:00",
            "space_id": "wrice/repro-paper-1",
            "space_sha": "space-sha-1",
        }
        _add_payload(record, payload)
    elif kind == "verdict":
        payload = {
            "snapshot_id": "d" * 64,
            "verdict_revision": "verdict-revision",
            "submission_attestation_id": "c" * 64,
            "authority_attestation_id": "e" * 64,
            "space_id": "wrice/repro-paper-1",
            "space_sha": "space-sha-1",
            "paper_id": "paper-1",
            "judged_at": "2026-07-25T17:00:00+00:00",
            "claims": [
                {
                    "target_claim": "claim-1",
                    "claim": "Challenge claim 1",
                    "status": "toy",
                    "evidence": "bounded run",
                },
                {
                    "target_claim": "claim-2",
                    "claim": "Challenge claim 2",
                    "status": "inconclusive",
                    "evidence": "missing artifact",
                },
            ],
        }
        _add_payload(record, payload)


def add_validation_fields(record: dict) -> None:
    payload = {
        "worktree": "/tmp/test-worktree",
        "branch": "test-branch",
        "base_sha": "2" * 40,
        "project_path": "submissions/paper-1",
        "design_path": "docs/designs/paper-1.md",
        "commands": [],
        "checks": [],
        "environment": [],
        "source_tree": "3" * 40,
        "source_tree_sha256": "6" * 64,
        "environment_sha256": "4" * 64,
    }
    _add_payload(record, payload)


def _add_payload(record: dict, payload: dict) -> None:
    record["source_commit"] = "5" * 40
    record["payload_sha256"] = hashlib.sha256(
        json.dumps(
            payload, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    record.update(payload)
