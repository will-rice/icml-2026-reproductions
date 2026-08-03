import importlib.util
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "generate_evidence.py"


def load_module():
    assert MODULE_PATH.exists(), "generate_evidence.py must exist"
    spec = importlib.util.spec_from_file_location("dhsa_evidence", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_upstream_cards_pin_architecture_and_bucket_totals():
    module = load_module()

    audit = module.audit_upstream_artifacts(PROJECT / "upstream")

    assert audit["model_card_sha256"] == "c9d62fa46b70ed2a558def60461d6637f4744c96821389c2da7defa2f796db9f"
    assert audit["long_data_summary_sha256"] == "dbe4e4899fa6cf0afc868afe9fdd49ae5537d166439050cf1cc960f4d938ec66"
    assert audit["model_card_architecture_terms"] == {
        "shared_encoder": True,
        "feature_fusion": True,
        "mlp_classifier": True,
        "frozen_backbone": False,
    }
    assert audit["pretrain_total_from_buckets"] == 9_383_848
    assert audit["fine_tune_total_from_buckets"] == 98_557
    assert audit["has_128k_examples"] is True


def test_dynamic_hierarchical_router_improves_recall_over_static_blocks():
    module = load_module()

    result = module.compare_dynamic_router_to_block_sparse(
        sequence_length=96,
        chunk_size=12,
        token_budget=24,
        seed=251024606,
    )

    assert result["dynamic_recall"] > result["block_sparse_recall"]
    assert result["dynamic_selected_tokens"] == 24
    assert result["block_selected_tokens"] == 24
    assert result["chunk_candidates"] == 4


def test_density_curve_is_monotone_and_uses_less_work_than_full_attention():
    module = load_module()

    curve = module.compute_density_curve([0.125, 0.25, 0.5], sequence_length=4096, chunk_size=256)

    recalls = [row["estimated_recall"] for row in curve]
    work = [row["relative_attention_work"] for row in curve]
    assert recalls == sorted(recalls)
    assert work == sorted(work)
    assert all(row["relative_attention_work"] < 1.0 for row in curve)


def test_evidence_bundle_and_pages_cover_all_bound_claims(tmp_path):
    module = load_module()

    output_dir = tmp_path / "evidence"
    summary = module.generate_evidence(output_dir=output_dir, upstream_dir=PROJECT / "upstream")

    bundle = json.loads((output_dir / "bundle.json").read_text(encoding="utf-8"))
    report = (output_dir / "pages" / "report.md").read_text(encoding="utf-8")
    measurements = (output_dir / "pages" / "01-measurements.md").read_text(encoding="utf-8")

    assert bundle == summary
    assert bundle["attempt_id"] == "020e5035-01ad-40a7-9ab0-9147289ab70c"
    assert bundle["paper_id"] == "o3gN27ITWV"
    assert len(bundle["claims"]) == 6
    assert {claim["challenge_claim_sha256"] for claim in bundle["claims"]} == {
        "cc2810b3712d97dac5c4ef31ff712a89cbe6d0fd54a805726a9c64716445b062",
        "8f498ed44034578b35a389c120d292d97ef0c74f11cd43f2ab1987ce538e0496",
        "7f69a4a2195e81c970b02f20e4d11a220417d84aab77bbae91aa5cad6680e244",
        "9bafe807d3c1daa5aaa8c8ab00386d06295d876ba1751070795a56c86ccb27fe",
        "d7dfa8bb0e8f2e086edadfbc155268bade452058c8e20912ae274df00ec66f44",
        "999014e0aa1e212f55e6b1bcb32e1e5897bce7e26a9ac91a3e01b35f0a9bd98c",
    }
    assert "dynamic recall" in measurements
    assert "paper-reported LongBench" not in measurements
    assert report.count("|") > 20
