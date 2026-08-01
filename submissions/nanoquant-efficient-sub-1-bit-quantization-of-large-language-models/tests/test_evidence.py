from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_evidence import build_evidence, main


def test_upstream_identity_and_license_are_pinned():
    bundle = build_evidence()
    assert bundle["paper_id"] == "qiZDlnvWTR"
    assert bundle["upstream"]["github"] == (
        "SamsungLabs/NanoQuant@a9e0a430881ff80d83b622c3129e330dc33c04f5"
    )
    assert bundle["upstream"]["code_license"] == "Apache-2.0"


def test_low_rank_binary_factorization_source_indicators_are_present():
    bundle = build_evidence()
    indicators = bundle["observations"]["source_indicators"]
    assert indicators["admm_nanoquant"]["path"] == "src/nanoquant/core/admm_nq.py"
    assert indicators["admm_nanoquant"]["has_binary_terms"] is True
    assert indicators["linear_module"]["path"] == "src/nanoquant/modules/linear.py"
    assert indicators["linear_module"]["has_learned_scales"] is True
    assert indicators["quant_config"]["supports_fractional_bits"] is True


def test_claim_results_do_not_overstate_large_benchmark_replication():
    bundle = build_evidence()
    assert bundle["claim_results"]["claim-1"]["status"] == "verified"
    assert bundle["claim_results"]["claim-2"]["status"] in {"toy", "inconclusive"}
    assert any("70B compression" in item for item in bundle["unreplicated"])
    assert any("WikiText" in item for item in bundle["unreplicated"])


def test_bundle_file_round_trips(tmp_path):
    output = tmp_path / "bundle.json"
    main(["--output", str(output)])
    data = json.loads(output.read_text(encoding="utf-8"))
    assert set(data["claim_results"]) == {"claim-1", "claim-2"}
    assert data["observations"]["project_metadata"]["supported_model_families"] == [
        "OPT",
        "Llama",
        "Qwen",
        "Gemma",
        "Rnj-1",
    ]


def test_scoring_page_is_substantive():
    page = Path(__file__).resolve().parents[1] / "pages" / "reproduction.md"
    text = page.read_text(encoding="utf-8")
    assert "NanoQuant" in text
    assert "claim-1" in text
    assert len(text.strip()) >= 200


def test_space_readme_declares_required_challenge_metadata():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front_matter = text.split("---", 2)[1]
    assert "sdk: gradio" in front_matter
    assert "icml2026-repro" in front_matter
    assert "paper-qiZDlnvWTR" in front_matter
