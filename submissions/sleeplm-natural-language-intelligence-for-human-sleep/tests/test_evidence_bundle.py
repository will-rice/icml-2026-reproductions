from __future__ import annotations

import json
from pathlib import Path

from sleeplm_repro.evidence import TARGET_CLAIMS, UPSTREAM_PINS, build_bundle


PROJECT = Path(__file__).resolve().parents[1]


def test_bundle_records_attempt_and_zero_cost_contract():
    bundle = build_bundle()
    assert bundle["attempt_id"] == "ca01c0a8-f6cc-4d80-bf3a-c569ba7b4896"
    assert bundle["paper_id"] == "9wpwfSJCp9"
    assert bundle["paper_title"] == "SleepLM: Natural-Language Intelligence for Human Sleep"
    assert bundle["cpu_only"] is True
    assert bundle["estimated_api_cost_usd"] == 0.0


def test_upstream_pins_are_immutable():
    assert UPSTREAM_PINS["arxiv"] == "arxiv:2602.23605"
    assert UPSTREAM_PINS["code_repo"].endswith(
        "@f788466b926a9ed95d473c220814c912d5ce6abc"
    )
    assert UPSTREAM_PINS["hf_model"].endswith(
        "@ec0f94ff2be04fe11ff5a2b37ac38e8f40aa5c53"
    )


def test_target_claims_bind_current_challenge_hashes():
    hashes = {claim["challenge_claim_sha256"] for claim in TARGET_CLAIMS}
    assert hashes == {
        "0a9ab1e42662e1b1e40ded1370179413b554c5a8663d8f3bd293c56ea6f694f8",
        "a98161b9f57420109f2d31de27ac2b2d45960406af43b39dc86e0f0b17463d01",
        "c0760f7182d3b658fb12eb9409432891beb5e7634a49f10187cea9afcb595666",
        "142c585a4ac9c87506014c60d24333769cec610fc3b02fa9e464082325b984af",
    }


def test_bundle_records_license_observations():
    bundle = build_bundle()
    licenses = bundle["observations"]["licenses"]
    assert licenses["github_repository"]["license"] == "MIT"
    assert licenses["hf_model_card"]["license"] == "mit"
    assert bundle["claim_results"]["artifact_release"]["status"] == "verified"


def test_bundle_records_signal_input_contract():
    bundle = build_bundle()
    signal = bundle["observations"]["signal_input_contract"]
    assert signal["tensor_shape"] == "[N, 10, 1920]"
    assert signal["epoch_seconds"] == 30
    assert signal["sampling_rate_hz"] == 64
    assert signal["channel_count"] == 10
    assert signal["channels"] == [
        "ECG",
        "ABD",
        "THX",
        "AF",
        "EOG_Left",
        "EOG_Right",
        "EEG_C3_A2",
        "EEG_C4_A1",
        "EMG_Chin",
        "POS",
    ]


def test_bundle_records_caption_retrieval_and_objective_support():
    bundle = build_bundle()
    capabilities = bundle["observations"]["capabilities"]
    assert capabilities["targeted_caption_generation"]["supported"] is True
    assert capabilities["cross_modal_retrieval"]["supported"] is True
    assert capabilities["modality_tokens"] == [
        "brain",
        "cardiac",
        "respiration",
        "somatic",
    ]
    objective = bundle["observations"]["pretraining_objective"]
    assert objective["terms"] == [
        "contrastive_alignment",
        "caption_generation",
        "signal_reconstruction",
    ]
    assert "L_total" in objective["project_page_equation"]


def test_bundle_records_dataset_scale_as_documented_not_recomputed():
    bundle = build_bundle()
    dataset = bundle["observations"]["dataset_scale"]
    assert dataset["cohorts"] == 5
    assert dataset["hours"] == "100K+"
    assert dataset["individuals"] == "10,000+"
    assert dataset["raw_training_data_available"] is False
    assert bundle["claim_results"]["dataset_scale"]["status"] == "inconclusive"


def test_bundle_records_source_hashes():
    bundle = build_bundle()
    sources = bundle["source_files"]
    assert sources["github_readme"]["sha256"] == (
        "dfba50c6afa5eb7023ed03fec2a5f563b3086853b9b9d53da4226c4087e96518"
    )
    assert sources["github_license"]["sha256"] == (
        "1965530331f9dadeca7674cbd66f1d1c431ad2b7fed05a975c769d9508819f74"
    )
    assert sources["signal_model_config"]["sha256"] == (
        "907f8e149af7c2e2d8bfb2bd894f62fffb54d6038b9814d88241327522d22dc9"
    )
    assert sources["hf_model_card"]["sha256"] == (
        "6fa6e6b1ffaef77a08b106e3c7e507c6a37e1e62e3768ad74739692fbfc63924"
    )


def test_generated_bundle_has_required_claim_statuses():
    bundle = json.loads((PROJECT / "evidence" / "bundle.json").read_text())
    assert {result["status"] for result in bundle["claim_results"].values()} <= {
        "verified",
        "toy",
        "inconclusive",
        "unavailable",
    }


def test_space_readme_contains_required_metadata():
    readme = (PROJECT / "README.md").read_text()
    assert "sdk: gradio" in readme
    assert "paper-9wpwfSJCp9" in readme
    assert "icml2026-repro" in readme
    assert "license: mit" in readme


def test_summary_page_names_claim_hashes_revision_and_limitations():
    summary = (PROJECT / "pages" / "00-summary.md").read_text()
    for claim in TARGET_CLAIMS:
        assert claim["challenge_claim_sha256"] in summary
    assert (
        "arxiv:2602.23605+github:yang-ai-lab/SleepLM@"
        "f788466b926a9ed95d473c220814c912d5ce6abc+hf-model:"
        "yang-ai-lab/SleepLM-Base@ec0f94ff2be04fe11ff5a2b37ac38e8f40aa5c53"
    ) in summary
    assert "Dataset-scale evidence is primary-artifact documentation" in summary
