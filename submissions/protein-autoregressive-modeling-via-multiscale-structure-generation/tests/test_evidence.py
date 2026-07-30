import json
from pathlib import Path
from par_protein.evidence import generate_evidence


def test_evidence_generation(tmp_path: Path):
    results_path = generate_evidence(tmp_path)
    assert results_path.exists()
    assert (tmp_path / "provenance.json").exists()

    with open(results_path) as f:
        data = json.load(f)

    assert data["paper_id"] == "08tW615mgI"
    assert len(data["target_claims"]) == 3
    for claim_item in data["target_claims"]:
        assert claim_item["status"] == "verified"
