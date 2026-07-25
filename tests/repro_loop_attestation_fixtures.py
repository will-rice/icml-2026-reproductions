"""Shared strict validation-attestation test values."""

import hashlib
import json


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
        "environment_sha256": "4" * 64,
    }
    record["source_commit"] = "5" * 40
    record["payload_sha256"] = hashlib.sha256(
        json.dumps(
            payload, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    record.update(payload)
