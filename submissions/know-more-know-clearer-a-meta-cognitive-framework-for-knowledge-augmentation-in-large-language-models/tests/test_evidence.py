import pytest
import json
from pathlib import Path
from know_more_know_clearer.evidence import generate_evidence

def test_generate_evidence(tmp_path):
    output_dir = tmp_path / "evidence"
    results_path = generate_evidence(output_dir)
    
    assert results_path.exists()
    with open(results_path) as f:
        data = json.load(f)
        
    assert "target_claims" in data
    assert len(data["target_claims"]) == 2
    for claim in data["target_claims"]:
        assert "claim" in claim
        assert "status" in claim
        assert claim["status"] in {"verified", "partial", "inconclusive", "contradicted", "unavailable"}
