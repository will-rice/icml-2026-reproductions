import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path
import subprocess
import sys

import pytest

from demix.artifacts import (
    ArtifactIntegrityError,
    ArtifactValidationError,
    analyze_manifest,
    load_pinned_manifest,
    normalize_weights,
)


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = SUBMISSION_ROOT / "evidence" / "inputs" / "sampled_mixture.json"
PROVENANCE = SUBMISSION_ROOT / "evidence" / "provenance.json"
BUNDLE = SUBMISSION_ROOT / "evidence" / "bundle.json"
PINNED_SHA256 = (
    "2be00152f98c44a740bc2f8e2098be3740ea2f1cd31b7158ade9d54c8e852dc2"
)


def test_vendored_manifest_is_byte_identical_to_pinned_release():
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == PINNED_SHA256

    provenance = json.loads(PROVENANCE.read_text())
    assert provenance["dataset"]["revision"] == (
        "82a2effc58eb79bec691280a4e4fc50be0968b1e"
    )
    assert provenance["dataset"]["primary_input"] == {
        "bytes": 4017,
        "path": "DeMix_reproduce/reference_models/sampled_mixture.json",
        "sha256": PINNED_SHA256,
    }
    inventory = provenance["release_inventory"]
    assert inventory["paths_below_demix_reproduce"] == 1469
    assert inventory["reference_model_roots"] == 16
    assert inventory["component_model_roots"] == 7
    assert inventory["csv_paths"] == 0
    assert inventory["opencompass_result_paths"] == 0
    shards = inventory["component_checkpoint"]["shards"]
    assert len(shards) == 14
    assert sum(shard["bytes"] for shard in shards) == 48_176_346_736
    assert inventory["component_checkpoint"]["total_bytes"] == 48_176_346_736


def test_pinned_manifest_observations_are_recomputed():
    manifest = load_pinned_manifest(MANIFEST)
    observations = analyze_manifest(manifest)

    assert observations["mixture_count"] == 17
    assert observations["mixture_ids"] == [f"mix_{index}" for index in range(17)]
    assert observations["domain_names"] == [
        "general_target",
        "math_very_high",
        "math_high",
        "math_medium",
        "code_very_high",
        "code_high",
        "code_medium",
    ]
    assert observations["raw_weight_sums"]["mix_0"] == "2933"
    assert observations["raw_weight_sums"]["mix_1"] == "1038.5"
    assert observations["raw_weight_sums"]["mix_2"] == "0.9998"
    assert observations["raw_weight_sums"]["mix_6"] == "1.0002"
    assert (
        observations["normalized_weights"]["mix_0"]["general_target"]
        == "0.399931810433"
    )
    assert (
        observations["normalized_weights"]["mix_2"]["general_target"]
        == "0.390678135627"
    )
    assert observations["normalization_required"] == [
        "mix_0",
        "mix_1",
        "mix_2",
        "mix_6",
        "mix_11",
        "mix_12",
        "mix_13",
        "mix_14",
    ]
    assert observations["already_unit_sum"] == [
        "mix_3",
        "mix_4",
        "mix_5",
        "mix_7",
        "mix_8",
        "mix_9",
        "mix_10",
        "mix_15",
        "mix_16",
    ]
    assert observations["all_weights_nonnegative"] is True
    assert observations["all_sums_positive"] is True
    assert observations["reference_model_count"] == 16
    assert observations["manifest_reference_count_match"] is False


def test_every_serialized_normalized_vector_sums_to_one():
    observations = analyze_manifest(load_pinned_manifest(MANIFEST))

    for normalized in observations["normalized_weights"].values():
        assert sum(map(Decimal, normalized.values())) == Decimal("1")


def test_modified_manifest_is_rejected(tmp_path):
    modified = tmp_path / "sampled_mixture.json"
    modified.write_bytes(MANIFEST.read_bytes() + b"\n")

    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        load_pinned_manifest(modified)


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"x": Decimal("0")},
        {"x": Decimal("-1")},
        {"x": Decimal("NaN")},
        {"x": Decimal("Infinity")},
        {"x": True},
        {"x": "1"},
    ],
)
def test_normalization_rejects_invalid_weights(weights):
    with pytest.raises(ArtifactValidationError):
        normalize_weights(weights)


def test_manifest_requires_matching_ordered_domains():
    manifest = load_pinned_manifest(MANIFEST)
    manifest["mix_16"] = dict(reversed(manifest["mix_16"].items()))

    with pytest.raises(ArtifactValidationError, match="ordered domains"):
        analyze_manifest(manifest)


def test_committed_bundle_contains_no_synthetic_or_verified_evidence():
    encoded = json.dumps(json.loads(BUNDLE.read_text()), sort_keys=True)

    for forbidden in (
        '"verified"',
        '"macro_spearman"',
        '"ground_truth"',
        '"proxy_scores"',
        '"multi_seed_stability"',
        '"domain_correlations"',
        '"top25_correlations"',
    ):
        assert forbidden not in encoded


def test_bundle_has_conservative_per_claim_statuses():
    from demix.pipeline import build_bundle

    bundle = build_bundle(MANIFEST, PROVENANCE)
    assert bundle["reproduction_status"] == "partial"

    claims = {claim["id"]: claim for claim in bundle["claims"]}
    assert claims["weighted-linear-model-merging"]["status"] == "partial"
    assert claims["spearman-proxy-accuracy"]["status"] == "unavailable"
    assert claims["mixture-optimization-benchmarking"]["status"] == "unavailable"
    assert claims["weighted-linear-model-merging"]["input_artifacts"] == [
        "evidence/inputs/sampled_mixture.json",
        "upstream:model_merge/generate_merge_yaml.py",
    ]
    assert claims["weighted-linear-model-merging"]["observation"] == {
        "manifest_mixture_count": 17,
        "reference_model_directory_count": 16,
        "counts_match": False,
        "normalization_required_count": 8,
    }
    assert claims["spearman-proxy-accuracy"]["observation"] is None
    assert claims["mixture-optimization-benchmarking"]["observation"] is None
    assert all(claim["limitations"] for claim in claims.values())


def test_generated_bundle_contains_exact_provenance_and_environment():
    from demix.pipeline import build_bundle

    bundle = build_bundle(MANIFEST, PROVENANCE)
    provenance = bundle["provenance"]

    assert provenance["paper"]["arxiv_revision"] == "2602.00747v3"
    assert provenance["paper"]["pdf_sha256"] == (
        "85ea10da0925ee5bd284eeb3143c345129c74c320829dabd9d0ba4413acf55a3"
    )
    assert provenance["upstream_repository"]["commit"] == (
        "d0c945ca84d5632c6ed1bfe469337cf880757422"
    )
    assert provenance["dataset"]["revision"] == (
        "82a2effc58eb79bec691280a4e4fc50be0968b1e"
    )
    assert provenance["dataset"]["primary_input"]["sha256"] == PINNED_SHA256
    assert provenance["paper"]["acquisition_command"].startswith("curl ")
    assert provenance["upstream_repository"]["acquisition_command"].startswith(
        "git clone "
    )
    assert provenance["dataset"]["acquisition_command"].startswith("hf download ")
    assert provenance["release_inventory"]["component_checkpoint"][
        "total_bytes"
    ] == 48_176_346_736
    assert bundle["environment"] == {
        "python": "3.12.11",
        "gradio": "6.20.0",
        "pytest": "8.4.2",
    }
    assert bundle["regeneration"]["command"] == (
        "PYTHONPATH=src python -m demix.pipeline "
        "--input evidence/inputs/sampled_mixture.json "
        "--provenance evidence/provenance.json "
        "--output evidence/bundle.json"
    )
    assert bundle["paper_context"]["reproduced"] is False
    assert "context only" in bundle["paper_context"]["note"].lower()


def test_build_bundle_rejects_self_reported_provenance(tmp_path):
    from demix.pipeline import EvidenceContractError, build_bundle

    provenance = json.loads(PROVENANCE.read_text())
    provenance["dataset"]["primary_input"]["sha256"] = "0" * 64
    modified = tmp_path / "provenance.json"
    modified.write_text(json.dumps(provenance))

    with pytest.raises(EvidenceContractError, match="primary input SHA-256"):
        build_bundle(MANIFEST, modified)


def test_cli_regeneration_is_byte_identical(tmp_path):
    output = tmp_path / "bundle.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SUBMISSION_ROOT / "src")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "demix.pipeline",
            "--input",
            str(MANIFEST),
            "--provenance",
            str(PROVENANCE),
            "--output",
            str(output),
        ],
        check=True,
        cwd=SUBMISSION_ROOT,
        env=environment,
    )
    assert output.read_bytes() == BUNDLE.read_bytes()
