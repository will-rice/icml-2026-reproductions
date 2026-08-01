from synermedgen_repro.core import (
    CLAIMS,
    ARTIFACTS,
    build_evidence_bundle,
    classify_claims,
    summarize_artifacts,
)


def test_missing_primary_project_is_not_treated_as_released_dataset():
    summary = summarize_artifacts(ARTIFACTS)

    assert summary["github_project"]["url"] == "https://github.com/Mhilab/SynerMedGen"
    assert summary["github_project"]["accessible"] is False
    assert summary["released_dataset_found"] is False


def test_classification_keeps_dataset_and_benchmark_claims_inconclusive_without_artifacts():
    statuses = {claim["sha256"]: claim["status"] for claim in classify_claims(CLAIMS, ARTIFACTS)}

    assert statuses["de5b83cda10b4e7f29dc8cd10df6f0172106ca34269883562fb6cce262b0c4c1"] == "toy"
    assert statuses["d40b38c6a084b9c6a917f0905d7e57f71f89c1ba588fa45281375562f5f36a03"] == "inconclusive"
    assert statuses["31b94e893509101e96e08423e6f869a2d9547d0d7061025e0127d22b8ccceb30"] == "inconclusive"
    assert statuses["2d08be03fa0be6353d754fd497dd1826c664110f17336abc5fcef7dbdbff59e3"] == "inconclusive"


def test_bundle_has_no_reproduced_synthesis_or_ablation_measurements():
    bundle = build_evidence_bundle()

    assert bundle["paper_id"] == "Tyv61ZKb9s"
    assert bundle["upstream"]["paper"] == "arxiv:2605.08724v1"
    assert bundle["reproduced_synthesis_measurements"] == []
    assert bundle["reproduced_ablation_measurements"] == []
    assert all("paper-reported" not in item.get("observation", "").lower() for item in bundle["claims"])


def test_bundle_generation_is_idempotent_for_validation():
    assert build_evidence_bundle() == build_evidence_bundle()
