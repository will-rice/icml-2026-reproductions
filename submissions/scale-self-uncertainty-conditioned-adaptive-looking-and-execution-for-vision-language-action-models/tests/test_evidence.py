from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_evidence import build_evidence, main


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
