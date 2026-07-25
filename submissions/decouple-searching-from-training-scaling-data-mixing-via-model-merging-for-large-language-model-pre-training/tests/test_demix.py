import hashlib
import json
from decimal import Decimal
from pathlib import Path

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
