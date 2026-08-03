"""Tests for the closed evidence renderer."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest


from rbench_repro.evidence import (
    AuditInputs,
    build_evidence,
    resolve_json_pointer,
    validate_evidence,
)
from rbench_repro.model import canonical_json, sha256_bytes

GENERATED_AT = "2026-07-27T00:00:00+00:00"
TOOL_REVISION = "abc123"


@pytest.fixture
def schema_path():
    return (
        Path(__file__).resolve().parents[1]
        / "schema"
        / "evidence-v1.schema.json"
    )




def test_evidence_has_controller_identity_and_honest_claim_statuses(
    complete_audit_inputs, schema_path
):
    evidence = build_evidence(
        complete_audit_inputs, "2026-07-27T00:00:00+00:00", "abc123"
    )
    validate_evidence(evidence, schema_path)
    assert evidence["paper_id"] == "p5QSlnwume"
    assert evidence["attempt_id"] == "8c21f2dc-a357-422e-9c1b-79a4d417e3dc"
    assert [item["status"] for item in evidence["claims"]] == [
        "verified", "verified", "inconclusive"
    ]
    assert "real-video" in evidence["claims"][2]["limitations"][0]


def test_evidence_is_canonical_and_every_pointer_resolves(complete_audit_inputs):
    first = build_evidence(complete_audit_inputs, GENERATED_AT, TOOL_REVISION)
    second = build_evidence(complete_audit_inputs, GENERATED_AT, TOOL_REVISION)
    assert canonical_json(first) == canonical_json(second)
    for artifact in first["artifacts"]:
        pointed = resolve_json_pointer(first, artifact["json_pointer"])
        assert artifact["sha256"] == sha256_bytes(canonical_json(pointed))


def test_claims_downgrade_when_required_artifact_routes_are_missing(
    complete_audit_inputs,
):
    complete_audit_inputs.metrics = ()
    assert build_evidence(
        complete_audit_inputs, GENERATED_AT, TOOL_REVISION
    )["claims"][0]["status"] == "partial"

    complete_audit_inputs.category_evidence = {}
    assert build_evidence(
        complete_audit_inputs, GENERATED_AT, TOOL_REVISION
    )["claims"][1]["status"] == "partial"

    complete_audit_inputs.failure_modes = ()
    assert build_evidence(
        complete_audit_inputs, GENERATED_AT, TOOL_REVISION
    )["claims"][2]["status"] == "inconclusive"


def test_artifact_inferred_formula_is_disclosed(complete_audit_inputs):
    complete_audit_inputs.formula_provenance = "artifact-inferred"
    evidence = build_evidence(complete_audit_inputs, GENERATED_AT, TOOL_REVISION)
    assert any(
        "not source-traced" in limitation
        for limitation in evidence["unavailable"]
    )


def test_missing_only_failure_modes_are_inconclusive(complete_audit_inputs):
    complete_audit_inputs.failure_modes = tuple(
        replace(
            item,
            status="missing",
            source_locations=(),
            parser_path=None,
            aggregation_path=None,
            invoked_by_entry_point=False,
            fixtures=(),
        )
        for item in complete_audit_inputs.failure_modes
    )
    claim = build_evidence(
        complete_audit_inputs, GENERATED_AT, TOOL_REVISION
    )["claims"][2]
    assert claim["status"] == "inconclusive"
