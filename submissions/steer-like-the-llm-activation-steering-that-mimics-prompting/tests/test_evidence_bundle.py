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
    assert bundle["attempt_id"] == "4e292a9a-9ba5-44ac-a309-f3fe685a0643"
    assert bundle["snapshot_id"] == "d84e8174f494825bf70b4d9201754ddbb3e065bbea0b46d1f90da85949bb882b"
    assert bundle["upstream_revision"] == (
        "arxiv:2605.03907+github:Nokia-Bell-Labs/"
        "steer-like-the-llm@3d916c618d146c5d657f055e432a432b0fa493c6"
    )

    statuses = bundle["claim_statuses"]
    assert set(statuses) == {
        "claim_1_activation_subtraction",
        "claim_2_token_dependent_strengths",
        "claim_3_psr_objectives",
    }
    for c_key, c_val in statuses.items():
        assert c_val["status"] == "verified"

    non_target = bundle["non_target_claim_statuses"]
    assert set(non_target) == {
        "persona_vectors_table_1",
        "axbench_table_3",
        "accumulated_psr_rmse_figure_3",
    }
    for c_val in non_target.values():
        assert c_val["status"] == "unreplicated"
