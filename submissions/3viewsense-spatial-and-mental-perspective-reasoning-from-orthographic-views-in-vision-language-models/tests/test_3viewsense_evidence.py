import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_claims_module():
    module_path = PROJECT_ROOT / "src" / "viewsense_repro" / "claims.py"
    assert module_path.exists(), "expected claims module to exist"
    spec = importlib.util.spec_from_file_location("viewsense_repro.claims", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_views_preserve_hidden_block_counting_evidence():
    claims = load_claims_module()
    blocks = [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)]

    views = claims.canonical_orthographic_views(blocks)

    assert views == {
        "front": {"x=0": 2, "x=1": 1},
        "left": {"y=0": 2, "y=1": 1},
        "top": {"x=0,y=0": 2, "x=1,y=0": 1, "x=1,y=1": 1},
    }
    assert claims.count_blocks_from_top_view(views["top"]) == 4


def test_upstream_audit_detects_dataset_training_and_missing_raw_results(tmp_path):
    claims = load_claims_module()
    upstream = tmp_path / "upstream"
    for relative in [
        "orthomind-3d-synthetic/block-count-synthetic/build_cube_views_json.py",
        "orthomind-3d-synthetic/object-synthetic/blender_renderer.py",
        "orthomind-3d-synthetic/ood-image/call_api_for_aigc.py",
        "sft-stage/README.md",
        "rl-stage/README.md",
        "evaluation/eval_vlm.py",
    ]:
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    audit = claims.audit_upstream_artifacts(upstream)

    assert audit["dataset_sources"] == {
        "programmatic_block_counting": True,
        "programmatic_object_reasoning": True,
        "generative_ai_ood": True,
        "game_engine_ood": False,
    }
    assert audit["training_stages"] == {
        "stage_i_oms_sft": True,
        "stage_ii_vgr_sft": True,
        "stage_iii_grpo_rl": True,
    }
    assert audit["evaluation_code_present"] is True
    assert audit["raw_evaluation_outputs_present"] is False


def test_evidence_bundle_uses_bound_claims_and_conservative_statuses(tmp_path):
    claims = load_claims_module()
    upstream = tmp_path / "upstream"
    for relative in [
        "README.md",
        "orthomind-3d-synthetic/block-count-synthetic/build_cube_views_json.py",
        "orthomind-3d-synthetic/object-synthetic/blender_renderer.py",
        "orthomind-3d-synthetic/ood-image/call_api_for_aigc.py",
        "sft-stage/README.md",
        "rl-stage/README.md",
        "evaluation/eval_vlm.py",
    ]:
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    output = tmp_path / "evidence.json"
    bundle = claims.write_evidence_bundle(upstream, output)

    assert json.loads(output.read_text(encoding="utf-8")) == bundle
    assert bundle["paper_id"] == "Hm8OEDKpiO"
    assert bundle["upstream"]["revision"] == "9439d901829923d0541007e24d9d718320ee1e15"
    assert [claim["status"] for claim in bundle["claims"]] == [
        "toy",
        "verified",
        "inconclusive",
        "inconclusive",
        "inconclusive",
    ]
    assert all(len(claim["challenge_claim_sha256"]) == 64 for claim in bundle["claims"])


def test_report_pages_end_with_single_newline(tmp_path):
    claims = load_claims_module()
    upstream = tmp_path / "upstream"
    for relative in [
        "README.md",
        "orthomind-3d-synthetic/block-count-synthetic/build_cube_views_json.py",
        "orthomind-3d-synthetic/object-synthetic/blender_renderer.py",
        "orthomind-3d-synthetic/ood-image/call_api_for_aigc.py",
        "sft-stage/README.md",
        "rl-stage/README.md",
        "evaluation/eval_vlm.py",
    ]:
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    bundle = claims.build_evidence_bundle(upstream)
    claims.write_report_pages(bundle, tmp_path / "pages")

    report = (tmp_path / "pages" / "00-summary.md").read_text(encoding="utf-8")
    assert report.endswith("\n")
    assert not report.endswith("\n\n")
