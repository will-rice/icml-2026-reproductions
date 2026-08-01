from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "generate_evidence.py"
APP_PATH = PROJECT_ROOT / "app.py"


def load_generator():
    assert GENERATOR_PATH.exists(), "generate_evidence.py must provide the evidence generator"
    spec = importlib.util.spec_from_file_location("guda_generate_evidence", GENERATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_records_pinned_sources_and_claim_statuses(tmp_path: Path):
    """Catches missing provenance or benchmark claims reported without evidence."""
    module = load_generator()

    output_path = tmp_path / "bundle.json"
    bundle = module.build_evidence_bundle(output_path=output_path)

    assert output_path.exists()
    assert json.loads(output_path.read_text()) == bundle
    assert bundle["paper_id"] == "5f0gw9YpZC"
    assert bundle["attempt_id"] == "3ffbc4da-8f54-4a81-b70e-8103fe8eda1d"
    assert bundle["upstream"]["repository"] == "https://github.com/sony/guda.git"
    assert bundle["upstream"]["commit"] == "9fcf10cc4362199efc4f975e4a950df826fada07"
    assert bundle["upstream"]["license"] == "MIT"
    assert bundle["estimated_paid_api_cost_usd"] == 0.0

    statuses = {claim["claim_sha256"]: claim["status"] for claim in bundle["claims"]}
    assert statuses["106c8d047410261b6f3b2038b498207ec9be867e354c567664d5f4cdd33c0917"] in {
        "verified",
        "toy",
    }
    for claim_sha in [
        "8cfe641882a49b33f0db50a94de87d4f60cbdda050fc34364c2267d024e9254d",
        "f2148792206d4cebe4304f05bcc130d9f83a77acab08a5df4e5b21d69930e619",
        "9c7f6323ad0541e5afe3f24417bbbe62c2510b3a1bd286cbae41f473122bd4ed",
        "dcd3d556206571fbe9121fac83ded756c044b9b7ff14ff48058d3c319b7d1338",
        "1821ab64dbe97bf7121de7694c3a3715844bb37300235ffb5b4451e878ba17ab",
    ]:
        assert statuses[claim_sha] in {"toy", "inconclusive", "unavailable"}


def test_unlearncanvas_metadata_and_anchor_configs_are_checked(tmp_path: Path):
    """Catches skipped prompt metadata and weighted-anchor validation."""
    module = load_generator()

    bundle = module.build_evidence_bundle(output_path=tmp_path / "bundle.json")
    metadata = bundle["observations"]["unlearncanvas_metadata"]
    assert metadata["train_prompts"]["rows"] == 24000
    assert metadata["train_prompts"]["styles"] == 60
    assert metadata["train_prompts"]["objects"] == 20
    assert metadata["eval_prompts"]["rows"] == 1200
    assert metadata["eval_prompts"]["styles"] == 60
    assert metadata["eval_prompts"]["objects"] == 20
    assert metadata["paper_faithful_style_count"] == 16
    assert metadata["paper_faithful_configs_match"] is True

    anchors = bundle["observations"]["anchor_configs"]
    assert anchors["weighted_select_config_present"] is True
    assert anchors["ablation_sampling_config_present"] is True
    assert anchors["weighted_differs_from_uniform"] is True


def test_synthetic_ranking_metrics_reward_correct_head_group(tmp_path: Path):
    """Catches inverted or unbounded ranking metric calculations."""
    module = load_generator()

    bundle = module.build_evidence_bundle(output_path=tmp_path / "bundle.json")
    ranking = bundle["observations"]["synthetic_ranking_metrics"]
    assert ranking["top1_accuracy"] == 1.0
    assert ranking["mean_reciprocal_rank"] == 1.0
    assert ranking["ndcg_at_3"] == 1.0


def test_app_summary_uses_generated_bundle(tmp_path: Path):
    """Catches a Space app that reports stale prose instead of bundle data."""
    generator = load_generator()
    bundle_path = PROJECT_ROOT / "evidence" / "bundle.json"
    generator.build_evidence_bundle(output_path=bundle_path)

    assert APP_PATH.exists(), "app.py must expose the evidence bundle"
    spec = importlib.util.spec_from_file_location("guda_app", APP_PATH)
    assert spec and spec.loader
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    summary = app.load_summary()
    assert summary["paper_id"] == "5f0gw9YpZC"
    assert summary["claim_count"] == 6
    assert summary["upstream_commit"] == "9fcf10cc4362199efc4f975e4a950df826fada07"
