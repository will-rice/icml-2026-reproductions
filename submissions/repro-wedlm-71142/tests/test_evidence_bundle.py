from wedlm_repro.evidence import build_evidence_bundle


def test_evidence_bundle_records_provenance_claim_statuses_and_speedup_limitation():
    """Would fail if the evidence claimed full GPU speed reproduction from CPU checks."""
    bundle = build_evidence_bundle(timestamp="2026-07-29T03:00:00+00:00", git_commit="test-sha")

    assert bundle["paper_id"] == "71142"
    assert bundle["attempt_id"] == "64537525-54cc-4b47-b721-72979fa954dd"
    assert bundle["cpu_only"] is True
    assert bundle["upstream_revision"].startswith("arxiv:2512.22737+github:tencent/WeDLM@8d3f66b")
    assert [claim["status"] for claim in bundle["claims"]] == ["toy", "toy", "toy", "unreplicated"]
    assert "vLLM" in bundle["limitations"][0]
