"""Build deterministic DeMix evidence from pinned released artifacts."""

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any, Mapping

from demix.artifacts import (
    PINNED_MIXTURE_SHA256,
    analyze_manifest,
    load_pinned_manifest,
)


PAPER_ID = "uyRIOjFgOn"
PAPER_TITLE = (
    "Decouple Searching from Training: Scaling Data Mixing via Model Merging "
    "for Large Language Model Pre-training"
)
PAPER_REVISION = "2602.00747v3"
PAPER_PDF_SHA256 = (
    "85ea10da0925ee5bd284eeb3143c345129c74c320829dabd9d0ba4413acf55a3"
)
UPSTREAM_COMMIT = "d0c945ca84d5632c6ed1bfe469337cf880757422"
DATASET_REVISION = "82a2effc58eb79bec691280a4e4fc50be0968b1e"
GENERATE_MERGE_YAML_SHA256 = (
    "c98210d9a9702bb648012d90f83f033514e456ad34aa209ee57a8b4efbd6ee34"
)
PROXY_EVAL_SHA256 = (
    "ba0bbd871c5e2aefca4d42f474ad02bccd98e68618d6d9efa76ec526e0931cd5"
)
COMPONENT_CHECKPOINT_BYTES = 48_176_346_736
REGENERATION_COMMAND = (
    "PYTHONPATH=src python -m demix.pipeline "
    "--input evidence/inputs/sampled_mixture.json "
    "--provenance evidence/provenance.json "
    "--output evidence/bundle.json"
)


class EvidenceContractError(ValueError):
    """Raised when checked-in provenance violates the evidence contract."""


def build_bundle(
    input_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    """Build the conservative evidence bundle from exact local artifacts."""
    manifest = load_pinned_manifest(input_path)
    observations = analyze_manifest(manifest)
    provenance = _load_and_validate_provenance(provenance_path)
    claims = _claim_records(provenance, observations)

    return {
        "schema_version": 1,
        "paper_id": PAPER_ID,
        "title": PAPER_TITLE,
        "reproduction_status": "partial",
        "provenance": provenance,
        "environment": {
            "python": "3.12.11",
            "gradio": "6.20.0",
            "pytest": "8.4.2",
        },
        "regeneration": {
            "command": REGENERATION_COMMAND,
            "serialization": (
                "UTF-8 JSON, keys sorted, two-space indentation, final newline"
            ),
        },
        "released_artifact_observations": observations,
        "claims": claims,
        "resource_constraints": {
            "component_checkpoint_bytes": COMPONENT_CHECKPOINT_BYTES,
            "component_checkpoint_gib": "44.87",
            "component_models": 7,
            "safetensors_shards": 14,
            "assessment": (
                "Full model merging and OpenCompass evaluation require material "
                "storage and accelerator resources and were not run."
            ),
        },
        "paper_context": {
            "reproduced": False,
            "note": (
                "Paper-reported values below are context only; no released "
                "benchmark outputs permit independent recomputation."
            ),
            "source": PAPER_REVISION,
            "table_2": {
                "demix_30b_x7": {
                    "reported_macro_rank_correlation": 0.81,
                    "reported_top25_rank_correlation": 0.59,
                    "reported_capability_recovery": 0.83,
                }
            },
        },
    }


def write_bundle(
    bundle: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Write canonical, byte-stable JSON."""
    encoded = (
        json.dumps(
            bundle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    output_path.write_text(encoded, encoding="utf-8")


def _load_and_validate_provenance(path: Path) -> dict[str, Any]:
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceContractError(
            "provenance must be readable UTF-8 JSON"
        ) from error

    required_values = [
        (
            ("paper", "arxiv_revision"),
            PAPER_REVISION,
            "paper revision",
        ),
        (
            ("paper", "pdf_sha256"),
            PAPER_PDF_SHA256,
            "paper PDF SHA-256",
        ),
        (
            ("upstream_repository", "commit"),
            UPSTREAM_COMMIT,
            "upstream commit",
        ),
        (
            (
                "upstream_repository",
                "files",
                "model_merge/generate_merge_yaml.py",
            ),
            GENERATE_MERGE_YAML_SHA256,
            "merge configuration SHA-256",
        ),
        (
            ("upstream_repository", "files", "eval_merged/proxy_eval.py"),
            PROXY_EVAL_SHA256,
            "proxy evaluation source SHA-256",
        ),
        (
            ("dataset", "revision"),
            DATASET_REVISION,
            "dataset revision",
        ),
        (
            ("dataset", "primary_input", "sha256"),
            PINNED_MIXTURE_SHA256,
            "primary input SHA-256",
        ),
        (
            ("release_inventory", "component_checkpoint", "total_bytes"),
            COMPONENT_CHECKPOINT_BYTES,
            "component checkpoint byte count",
        ),
        (
            ("release_inventory", "csv_paths"),
            0,
            "released CSV path count",
        ),
        (
            ("release_inventory", "opencompass_result_paths"),
            0,
            "released OpenCompass-result path count",
        ),
    ]
    for key_path, expected, label in required_values:
        actual = _nested_value(provenance, key_path, label)
        if actual != expected:
            raise EvidenceContractError(
                f"{label} {actual!r} does not match pinned {expected!r}"
            )

    shards = _nested_value(
        provenance,
        ("release_inventory", "component_checkpoint", "shards"),
        "component checkpoint shards",
    )
    if not isinstance(shards, list) or len(shards) != 14:
        raise EvidenceContractError(
            "component checkpoint provenance must contain fourteen shards"
        )
    if sum(_shard_bytes(shard) for shard in shards) != COMPONENT_CHECKPOINT_BYTES:
        raise EvidenceContractError(
            "component shard byte counts do not match pinned total"
        )

    return provenance


def _claim_records(
    provenance: Mapping[str, Any],
    observations: Mapping[str, Any],
) -> list[dict[str, Any]]:
    upstream_files = provenance["upstream_repository"]["files"]
    return [
        {
            "id": "weighted-linear-model-merging",
            "status": "partial",
            "observation": {
                "manifest_mixture_count": observations["mixture_count"],
                "reference_model_directory_count": observations[
                    "reference_model_count"
                ],
                "counts_match": observations["manifest_reference_count_match"],
                "normalization_required_count": len(
                    observations["normalization_required"]
                ),
            },
            "input_artifacts": [
                "evidence/inputs/sampled_mixture.json",
                "upstream:model_merge/generate_merge_yaml.py",
            ],
            "provenance": [
                {
                    "artifact": "evidence/inputs/sampled_mixture.json",
                    "source_revision": DATASET_REVISION,
                    "sha256": PINNED_MIXTURE_SHA256,
                },
                {
                    "artifact": "upstream:model_merge/generate_merge_yaml.py",
                    "source_revision": UPSTREAM_COMMIT,
                    "sha256": upstream_files[
                        "model_merge/generate_merge_yaml.py"
                    ],
                },
            ],
            "limitations": [
                (
                    "The released ratios and upstream merge configuration were "
                    "audited, but no component checkpoint was downloaded or merged."
                ),
                (
                    "No inference or benchmark evaluation was run, so model "
                    "behavior is not verified."
                ),
            ],
        },
        {
            "id": "spearman-proxy-accuracy",
            "status": "unavailable",
            "observation": None,
            "input_artifacts": [
                "upstream:eval_merged/proxy_eval.py",
                "dataset:release_inventory",
            ],
            "provenance": [
                {
                    "artifact": "upstream:eval_merged/proxy_eval.py",
                    "source_revision": UPSTREAM_COMMIT,
                    "sha256": upstream_files["eval_merged/proxy_eval.py"],
                },
                {
                    "artifact": "dataset:release_inventory",
                    "source_revision": DATASET_REVISION,
                },
            ],
            "limitations": [
                (
                    "The pinned release has no OpenCompass result CSVs or "
                    "equivalent per-mixture benchmark outputs."
                ),
                (
                    "The upstream proxy evaluation loader contains path "
                    "placeholders and random.random(), so it cannot supply "
                    "reproduced measurements."
                ),
            ],
        },
        {
            "id": "mixture-optimization-benchmarking",
            "status": "unavailable",
            "observation": None,
            "input_artifacts": [
                "dataset:release_inventory",
                f"paper:{PAPER_REVISION}",
            ],
            "provenance": [
                {
                    "artifact": "dataset:release_inventory",
                    "source_revision": DATASET_REVISION,
                },
                {
                    "artifact": f"paper:{PAPER_REVISION}",
                    "sha256": PAPER_PDF_SHA256,
                },
            ],
            "limitations": [
                (
                    "The final benchmark outputs needed to recompute the paper "
                    "comparison are not released."
                ),
                (
                    f"The seven component checkpoints total "
                    f"{COMPONENT_CHECKPOINT_BYTES:,} bytes before merge outputs "
                    "and evaluation caches; the full accelerator evaluation was "
                    "not run."
                ),
            ],
        },
    ]


def _nested_value(
    mapping: Mapping[str, Any],
    key_path: tuple[str, ...],
    label: str,
) -> Any:
    current: Any = mapping
    for key in key_path:
        if not isinstance(current, Mapping) or key not in current:
            raise EvidenceContractError(f"provenance is missing {label}")
        current = current[key]
    return current


def _shard_bytes(shard: Any) -> int:
    if not isinstance(shard, Mapping):
        raise EvidenceContractError("component shard record must be an object")
    value = shard.get("bytes")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvidenceContractError("component shard bytes must be positive integers")
    sha256 = shard.get("lfs_sha256")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise EvidenceContractError(
            "component shard LFS SHA-256 must be lowercase hexadecimal"
        )
    return value


def _argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Generate conservative DeMix released-artifact evidence."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    bundle = build_bundle(arguments.input, arguments.provenance)
    write_bundle(bundle, arguments.output)


if __name__ == "__main__":
    main()
