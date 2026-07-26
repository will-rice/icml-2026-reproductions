from pathlib import Path
from recurrent_sampler_repro.evidence import audit_theorem


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_theorem_audit():
    project_root = get_project_root()
    res = audit_theorem(project_root)

    assert res["evidence_status"] == "unavailable"
    assert res["proof_reproduced"] is False
    assert res["challenge_citation"] == "Theorem 4.2"

    audit = res["citation_audit"]
    assert audit["citation_mismatch_detected"] is True
    assert "Prefilling" in audit["theorem_4_2_title"] or "prefill" in audit["theorem_4_2_title"].lower()
    assert "decoding" in audit["theorem_4_4_title"].lower() or "Decoding" in audit["theorem_4_4_title"]
