import pytest
import os
import json
from steer_like_llm.evidence_bundle import run_evidence_pipeline

def test_run_evidence_pipeline(tmp_path):
    output_dir = str(tmp_path / "results")
    bundle = run_evidence_pipeline(output_dir=output_dir)
    
    assert os.path.exists(os.path.join(output_dir, "results.json"))
    assert bundle["paper_id"] == "06Nk3dJDMq"
    assert "claim_statuses" in bundle
    
    # Check claim statuses
    statuses = bundle["claim_statuses"]
    for c_key, c_val in statuses.items():
        assert c_val["status"] == "verified"
