from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_evidence import build_evidence, main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_served_pages_include_summary_and_evidence_detail():
    pages = sorted((PROJECT_ROOT / "pages").glob("*.md"))
    assert len(pages) >= 2
    assert (PROJECT_ROOT / "pages" / "00-summary.md").exists()
    assert (PROJECT_ROOT / "pages" / "01-claims-and-evidence.md").exists()
    summary_text = (PROJECT_ROOT / "pages" / "00-summary.md").read_text(encoding="utf-8")
    assert "toy mechanism checks" in summary_text.lower() or "toy" in summary_text.lower()
    claims_text = (PROJECT_ROOT / "pages" / "01-claims-and-evidence.md").read_text(encoding="utf-8")
    assert "9c25ef590bdbf95cd8dfa64cbaf7ce7093649e4b304868d51d028bf9eedd135d" in claims_text or "9c25ef590bdbf95cd8dfa64cbaf7ce7093649e4b304868d51d028bf9eedd135d" in summary_text


def test_served_pages_do_not_promote_unavailable_benchmark_claims():
    rendered_pages = "\n".join(
        page.read_text(encoding="utf-8") for page in (PROJECT_ROOT / "pages").glob("*.md")
    )
    bundle = build_evidence()
    unavailable_claims = [
        claim_id
        for claim_id, result in bundle["claim_results"].items()
        if result["status"] == "unavailable"
    ]
    assert unavailable_claims == ["claim-3", "claim-4", "claim-5", "claim-6"]
    for claim_id in unavailable_claims:
        display_id = claim_id.replace("claim-", "Claim ")
        assert not re.search(rf"\|\s*{re.escape(display_id)}\s*\|[^\n]*\|\s*Verified\s*\|", rendered_pages)


def test_bundle_provenance_is_pinned():
    bundle = build_evidence()
    assert bundle["paper_id"] == "7MlfE2Da2W"
    assert bundle["snapshot_id"] == "00263d6d9b331596d4be77a4cd17a4b1b6592f2ac7a72401cd62b751eaaef9bb"
    assert bundle["upstream"]["github"] == (
        "snumprlab/scale@b4ad2a69d14f91712704711e810cf9830e2b7121"
    )
    assert bundle["upstream"]["code_license"] == "MIT"


def test_scale_mechanism_indicators_are_detected():
    bundle = build_evidence()
    indicators = bundle["observations"]["source_indicators"]
    assert indicators["self_uncertainty"]["present"] is True
    assert indicators["self_uncertainty"]["path"] == "prismatic/extern/hf/modeling_prismatic.py"
    assert indicators["action_temperature"]["present"] is True
    assert indicators["visual_attention_temperature"]["present"] is True
    assert indicators["scale_config"]["values"] == {
        "T0": 1.0,
        "epsilon": 1.0e-12,
        "num_logits": 256,
        "kappa": 2.0,
        "alpha": 0.8,
        "attn_sensitivity": 0.3,
    }


def test_no_training_no_verifier_claim_is_not_overstated():
    bundle = build_evidence()
    claim = bundle["claim_results"]["claim-2"]
    assert claim["status"] in {"toy", "inconclusive"}
    assert "source-level" in claim["summary"]
    assert bundle["observations"]["runtime_constraints"]["gpu_robot_eval_required"] is True


def test_benchmark_claims_remain_unavailable_without_raw_artifacts():
    bundle = build_evidence()
    assert bundle["claim_results"]["claim-1"]["status"] == "verified"
    for claim_id in ("claim-3", "claim-4", "claim-5", "claim-6"):
        assert bundle["claim_results"][claim_id]["status"] == "unavailable"
    assert any("LIBERO" in item for item in bundle["unreplicated"])
    assert any("real-world" in item for item in bundle["unreplicated"])


def test_bundle_file_round_trips(tmp_path):
    output = tmp_path / "bundle.json"
    main(["--output", str(output)])
    data = json.loads(output.read_text(encoding="utf-8"))
    assert sorted(data["claim_results"]) == [
        "claim-1",
        "claim-2",
        "claim-3",
        "claim-4",
        "claim-5",
        "claim-6",
    ]
    assert data["observations"]["source_tree"]["commit"] == "b4ad2a69d14f91712704711e810cf9830e2b7121"
