"""Tests for the CLI commands and offline audit isolation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rbench_repro.evidence import AuditInputs, build_evidence, validate_evidence
from rbench_repro.model import canonical_json, sha256_bytes


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def offline_environment() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()}
    env["ALL_PROXY"] = "http://127.0.0.1:9"
    env["HTTPS_PROXY"] = "http://127.0.0.1:9"
    env["HTTP_PROXY"] = "http://127.0.0.1:9"
    env["NO_PROXY"] = ""
    env["UV_CACHE_DIR"] = str(project_root() / ".cache" / "uv")
    return env


def test_audit_is_offline_schema_valid_and_byte_identical(
    complete_audit_inputs,
):
    """Evidence builds are deterministic and schema-valid when run offline."""
    schema_path = project_root() / "schema" / "evidence-v1.schema.json"
    generated_at = "2026-07-27T00:00:00+00:00"
    tool_revision = "a" * 40
    outputs = []
    for _index in range(2):
        evidence = build_evidence(complete_audit_inputs, generated_at, tool_revision)
        validate_evidence(evidence, schema_path)
        outputs.append(canonical_json(evidence))
    assert outputs[0] == outputs[1]


def test_invalid_input_preserves_existing_output(tmp_path):
    """Audit with missing manifest exits nonzero and preserves existing output."""
    output = tmp_path / "results.json"
    output.write_bytes(b"preserve\n")
    result = subprocess.run(
        [
            sys.executable, "-m", "rbench_repro.cli",
            "audit",
            "--manifest", str(tmp_path / "missing.json"),
            "--cache-dir", str(tmp_path / "cache"),
            "--schema", str(project_root() / "schema" / "evidence-v1.schema.json"),
            "--output", str(output),
            "--generated-at", "2026-07-27T00:00:00+00:00",
            "--tool-revision", "a" * 40,
        ],
        capture_output=True,
        env=offline_environment(),
        text=True,
    )
    assert result.returncode != 0
    assert output.read_bytes() == b"preserve\n"
    assert "Traceback" not in result.stderr


def test_cli_validate_rejects_invalid_evidence(tmp_path):
    """The validate subcommand exits nonzero for invalid evidence."""
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}")
    result = subprocess.run(
        [
            sys.executable, "-m", "rbench_repro.cli",
            "validate",
            str(invalid),
            "--schema", str(project_root() / "schema" / "evidence-v1.schema.json"),
        ],
        capture_output=True,
        env=offline_environment(),
        text=True,
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
