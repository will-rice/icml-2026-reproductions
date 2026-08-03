"""Tests for the committed evidence bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rbench_repro.evidence import validate_evidence
from rbench_repro.model import sha256_bytes


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parents[1]


def test_committed_bundle_validates_and_matches_controller_attempt(project_root):
    results_path = project_root / "evidence" / "results.json"
    validation_path = project_root / "evidence" / "validation.json"
    schema_path = project_root / "schema" / "evidence-v1.schema.json"

    results = json.loads(results_path.read_text())
    validation = json.loads(validation_path.read_text())
    validate_evidence(results, schema_path)
    assert results["attempt_id"] == "8c21f2dc-a357-422e-9c1b-79a4d417e3dc"
    assert validation["valid"] is True
    assert validation["results_sha256"] == sha256_bytes(
        results_path.read_bytes()
    )


def test_commands_json_records_audit_environment(project_root):
    commands_path = project_root / "evidence" / "commands.json"
    commands = json.loads(commands_path.read_text())
    assert "commands" in commands
    assert "python_version" in commands
    assert "platform" in commands
    assert len(commands["commands"]) >= 1
