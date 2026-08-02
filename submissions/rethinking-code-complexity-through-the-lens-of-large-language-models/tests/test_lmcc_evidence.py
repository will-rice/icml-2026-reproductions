import json
import subprocess
import sys
from pathlib import Path


def test_synthetic_lmcc_uses_depth_and_branch_formula():
    from lmcc_repro import evidence

    tree = {
        "children": [
            {"children": []},
            {"children": [{"children": []}]},
        ]
    }

    assert evidence.block_count(tree) == 4
    assert evidence.depth_sum(tree) == 8
    assert evidence.lmcc_score(tree) == 4.0


def test_bundle_records_pinned_upstream_and_claim_statuses():
    from lmcc_repro import evidence

    bundle = evidence.build_evidence_bundle()

    assert bundle["attempt_id"] == "48a537d9-3320-4f51-80f5-45c226518c38"
    assert bundle["paper_id"] == "tI5CFbRhmV"
    assert bundle["snapshot_id"] == "ae039423cdd4ba289b7bce43249640ced01810ccdf857e862820729b4f0c9800"
    assert bundle["estimated_api_cost_usd"] == 0.0
    assert bundle["upstream"]["commit"] == "c38a26afdfc29ee517d734c6b677a4d6c65ec59b"
    assert bundle["upstream"]["license"] == "not-found"

    selected = {
        claim["challenge_claim_sha256"]: claim["status"]
        for claim in bundle["claims"]
        if claim["selected"]
    }
    assert selected == {
        "0660349f28ad245cfc3aa87b991574cf807be5678852c6a16b0d83e01e665723": "verified",
        "ff509c260c341fe13c097c68328cd36dc01df72f0c91d492daa4b10e90fcf1f8": "toy",
        "0e79a2f5620552c4fd3871adb4129bc2da045163409b908b60d714f7234d5366": "toy",
    }


def test_correlation_observations_are_computed_from_upstream_files():
    from lmcc_repro import evidence

    bundle = evidence.build_evidence_bundle()
    correlation = next(
        claim for claim in bundle["claims"]
        if claim["challenge_claim_sha256"]
        == "ff509c260c341fe13c097c68328cd36dc01df72f0c91d492daa4b10e90fcf1f8"
    )

    observations = correlation["observations"]["tasks"]
    assert {item["task"] for item in observations} == {
        "program_repair",
        "code_translation",
        "execution_reasoning",
    }
    assert all(item["source_records"] > 0 for item in observations)
    assert all(item["raw_spearman_r"] < 0 for item in observations)
    assert all(item["raw_spearman_p"] < 0.05 for item in observations)
    significant_partial = [
        item for item in observations
        if item["partial_spearman_r"] is not None and item["partial_spearman_p"] < 0.05
    ]
    assert [item["task"] for item in significant_partial] == ["program_repair"]


def test_generate_evidence_script_writes_bundle():
    script = Path(__file__).resolve().parent.parent / "generate_evidence.py"

    subprocess.run([sys.executable, str(script)], check=True)

    output = script.parent / "evidence" / "bundle.json"
    assert output.is_file()
    bundle = json.loads(output.read_text())
    assert bundle["paper_id"] == "tI5CFbRhmV"
    assert len(bundle["claims"]) == 6
