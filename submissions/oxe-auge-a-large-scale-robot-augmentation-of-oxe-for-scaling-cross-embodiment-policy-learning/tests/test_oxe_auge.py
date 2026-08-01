from oxe_auge_repro.pipeline import run_pipeline_audit, generate_evidence_bundle


def test_pipeline_audit():
    audit = run_pipeline_audit(seed=42)
    assert audit["verified"] is True
    assert len(audit["pipeline_stages"]) == 4
    assert audit["transfer_rates"]["seen"] > audit["transfer_rates"]["unseen"]


def test_generate_evidence_bundle():
    bundle = generate_evidence_bundle("8a560e44-d1c7-4f3b-819c-2ba8e0bfa749")
    assert bundle["attempt_id"] == "8a560e44-d1c7-4f3b-819c-2ba8e0bfa749"
    assert len(bundle["claims"]) == 5
    statuses = [c["status"] for c in bundle["claims"]]
    assert "verified" in statuses
    assert "toy" in statuses
