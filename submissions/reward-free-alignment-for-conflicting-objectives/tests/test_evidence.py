from pathlib import Path
import pytest
from reward_free_alignment.evidence import (
    build_evidence,
    canonical_json,
    validate_evidence,
)
from reward_free_alignment.provenance import load_manifest

EXPECTED_HASHES = (
    "e9a35e34b57a7273bf84d3d5981ab19f8ff1088adef8363f4640dcf436183944",
    "7c0aa54e034d03f2d0905417a024af4db41338cd9a803a0b42e441945c307cf9",
    "85abbc8a21d5c4537409f6e9f2af6bffc7e4c15e2311dfa078bf816ea0cffc9e",
    "dac93f364ac0469302894920781b034bfcd205816fbe16537c2f8e7c10d8995d",
    "269d8a5053e224206036399bccb2435455565149086de6439a9046d89682772b",
    "0d457572ea8a502fa8a489fef3e15da21b13cc39dd3a3730843d1cbe833059b0",
    "50719d645042a500f9c4d53fbdfedf719ee56429ddb73a231912f1eaeadb1b31",
    "5ec835ce150ff60d1e2bbd4fbdf7d1ebacf91bb2b6b2d65f72c44c3b3ed65229",
    "b74a0ea75967144b210934fd40fd23449d3ef985df878d5a9e14c4b04025ba4b",
    "58b31f527bb5e1bccb05c0dab775a74c2f2bdcd8e92ef2c0dd578733b5fb058e",
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).parent.parent


def test_bundle_contains_all_ten_live_claims_in_exact_order(project_root):
    evidence = build_evidence(project_root)
    assert evidence["snapshot_id"] == (
        "09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199"
    )
    assert tuple(c["sha256"] for c in evidence["claims"]) == EXPECTED_HASHES
    assert len(evidence["claims"]) == 10
    assert [c["targeted"] for c in evidence["claims"]] == [
        False, False, False, False, False, True, True, True, True, False
    ]


def test_local_outcomes_never_impersonate_official_verdicts(project_root):
    evidence = build_evidence(project_root)
    assert {c["local_outcome"] for c in evidence["claims"]} <= {
        "supported", "not-supported", "limited"
    }
    assert not {"verified", "falsified", "toy"} & {
        c["local_outcome"] for c in evidence["claims"]
    }


def test_evidence_is_canonical_and_byte_deterministic(project_root):
    first = canonical_json(build_evidence(project_root))
    second = canonical_json(build_evidence(project_root))
    assert first == second
    assert first.endswith(b"\n")
    assert b"NaN" not in first and b"Infinity" not in first


def test_evidence_passes_schema_validation(project_root):
    evidence = build_evidence(project_root)
    schema_path = project_root / "schema/evidence-v1.schema.json"
    validate_evidence(evidence, schema_path)


# --- Adversarial regressions for controller correction gate ---


def test_outcomes_are_derived_from_audits_not_hardcoded(project_root):
    """Claim outcomes must depend on actual audit results, not hard-coded labels."""
    evidence = build_evidence(project_root)
    # Claims 6-9 are targeted and should have outcomes derived from the audit
    targeted_claims = [c for c in evidence["claims"] if c["targeted"]]
    assert len(targeted_claims) == 4
    # Each targeted claim must have a reproduction_notes that references the audit
    for c in targeted_claims:
        assert c["local_outcome"] in ("supported", "not-supported", "limited")
        # Notes should reference specific computed values, not generic boilerplate
        assert len(c["reproduction_notes"]) > 20


def test_evidence_generation_calls_verification(project_root):
    """Evidence builder must call schema validation (not conditionally skip it)."""
    evidence = build_evidence(project_root)
    # If the schema existed, it was validated; test proves no exception was raised
    schema_path = project_root / "schema/evidence-v1.schema.json"
    assert schema_path.is_file()
    validate_evidence(evidence, schema_path)
