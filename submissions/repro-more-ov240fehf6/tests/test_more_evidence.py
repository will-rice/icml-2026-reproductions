from pathlib import Path

import pytest

from generate_evidence import build_bundle, main


ATTEMPT_ID = "ccbccca4-1090-462a-9fb6-2c6a8f594010"
PAPER_ID = "ov240fehF6"
GITHUB_PIN = "zimoqingfeng/MORE@f05c768e84f925fb2b14d3f8dd282d7036b46a21"
HF_DATASET_PIN = "zimoqingfeng/MORE@9395c04524a0c26dcb443a41f8655e808e18913b"
TARGET_HASHES = {
    "379b2d59a921cfdbe067f56b7f37e7381b9d54e78c235ba5e81f72564f03f51e",
    "a88362602383931a681db4d5a8c5ba8dc853bc3309ac0010c641faffdfa2f030",
    "f93ccd061ff16c49aebbe8066b7e81ac797becc2e0d3e1b61b74510875e8167b",
    "058defb5861b0e71fbb12437a97a7c1d314e156bad3eec0e0426bca61e4755a6",
    "b854f5b99d8aa58f2b4ba4fb762642a272cc806aecd9eb23e8141c392d66e4bb",
}


def test_bundle_records_attempt_pins_and_license():
    bundle = build_bundle()

    assert bundle["paper_id"] == PAPER_ID
    assert bundle["attempt_id"] == ATTEMPT_ID
    assert bundle["api_cost_usd"] == 0.0
    assert bundle["upstream_revisions"]["github"] == GITHUB_PIN
    assert bundle["upstream_revisions"]["hf_dataset"] == HF_DATASET_PIN
    assert bundle["artifact_observations"]["license"] == "apache-2.0"


def test_bundle_covers_selected_live_claims():
    bundle = build_bundle()

    claim_hashes = {claim["claim_sha256"] for claim in bundle["claims"]}
    assert TARGET_HASHES <= claim_hashes
    for claim in bundle["claims"]:
        assert claim["status"] in {"verified", "falsified", "toy", "unavailable"}
        assert claim["computed_observations"]
        assert not claim.get("reproduced_from_paper_value", False)


def test_artifact_observations_cover_more_structure():
    bundle = build_bundle()
    observations = bundle["artifact_observations"]

    assert observations["language_count"] == 149
    assert observations["script_family_count"] == 6
    assert observations["file_count"] >= 1000
    assert observations["task_names"] == [
        "catalog",
        "code",
        "formula",
        "reading_order",
        "table",
        "text",
    ]
    assert observations["readme_observations"]["annotation_pipeline_mentions_human_refinement"]
    assert observations["readme_observations"]["comparison_table_mentions_more_149_languages"]
    assert observations["readme_observations"]["six_task_coverage_claim"]
    assert observations["table_score_is_bottleneck"]


def test_main_writes_deterministic_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])

    assert main() == 0
    first = Path("evidence/bundle.json").read_bytes()
    assert main() == 0
    second = Path("evidence/bundle.json").read_bytes()

    assert first == second
    assert b"ov240fehF6" in first


def test_space_metadata_and_summary_page_are_claim_bound():
    root = Path(__file__).resolve().parents[1]
    readme = root / "README.md"
    summary = root / "pages/00-summary.md"

    if not readme.exists() or not summary.exists():
        pytest.fail("Space README and summary page must exist")

    readme_text = readme.read_text()
    summary_text = summary.read_text()
    assert "icml2026-repro" in readme_text
    assert "paper-ov240fehF6" in readme_text
    for claim_hash in TARGET_HASHES:
        assert claim_hash in summary_text
    assert "does not rerun OCR or VLM baselines" in summary_text
