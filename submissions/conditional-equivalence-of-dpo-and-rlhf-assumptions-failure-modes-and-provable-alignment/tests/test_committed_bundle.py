import hashlib
import json

from conditional_dpo_repro.evidence import (
    build_evidence,
    canonical_json_bytes,
    validate_evidence,
)


def test_committed_bundle_matches_fresh_build(project_root):
    committed = (project_root / "evidence.json").read_bytes()
    fresh = canonical_json_bytes(build_evidence(project_root))
    assert committed == fresh
    validate_evidence(
        json.loads(committed),
        project_root / "schema/evidence-v1.schema.json",
    )


def test_validation_binds_evidence_hash(project_root):
    payload = (project_root / "evidence.json").read_bytes()
    validation = json.loads(
        (project_root / "evidence/validation.json").read_text("utf-8")
    )
    assert validation["evidence_sha256"] == hashlib.sha256(payload).hexdigest()
    assert validation["schema_valid"] is True
    assert validation["deterministic"] is True
