"""Canonical, deterministic AGoQ evidence composition."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

from .memory_accounting import audit_table_1, fraction_text
from .pipeline_allocator import audit_pipeline
from .provenance import (
    COMMIT,
    REPOSITORY,
    load_verified_sources,
    load_verified_transcription,
)
from .source_audit import audit_released_source

ATTEMPT_ID = "2fc3b006-3307-4fc3-8df6-c000379298c4"
PAPER_ID = "ymHDVBwmta"
LIVE_CLAIMS = (
    "AGoQ combines layer-aware activation quantization and precision-preserved gradient quantization within Megatron-LM-style distributed training (Section 3).",
    "Layer-aware activation quantization reduces cached activation memory for a transformer layer from 28U in BF16 and 16.5U in COAT to 7.75U (Table 1).",
    "Dynamic Bit-width Compensation for Pipeline Parallelism assigns higher activation bit-widths to underutilized pipeline stages while maintaining near-4-bit activation storage (Section 4.2).",
    "Kernel fusion combines quantization/dequantization with adjacent GEMM operations to reduce activation-quantization overhead (Figure 4).",
    "On LLaMA2-13B sequence lengths from 32K to 80K, AGoQ reports faster training than Megatron-LM and ZeRO-1 while avoiding activation recomputation in the listed settings (Table 2).",
    "AGoQ reports lower memory than COAT and comparable or faster training time on OLMo-1B at 24K and 32K sequence lengths (Table 3).",
)
CLAIM_SHA256 = (
    "0b198b87a5abf16409a547a6f5277a41a62eac4a791b71cada94b054c65a1a13",
    "89292ed940125355f402bc04bc847acbed65f01bd0718124cceb88416ec24228",
    "a5a088563e0ab1a912f212da4246d90e8df679e6312e494ec486f0c38953b5bf",
    "88c789000f385b4692435064cb66b427ecbfd05b92c0632adde7681cb7b69eaa",
    "a513e6751344f810d77db2b7cd9a2fac9cf9ceab94f2a583a0247f917e64145d",
    "7391424029d3da524d5b5dfe17c88119ee6b0b7d1808ec6d0bc80366630efd1a",
)


def _components(audits: Mapping[str, object], method: str) -> dict[str, str]:
    row = audits[method]
    return {name: fraction_text(value) for name, value in row.components_u.items()}


def _fraction_sequence(values: tuple[Fraction, ...]) -> list[str]:
    return [fraction_text(value) for value in values]


def build_evidence(project_root: Path) -> dict[str, object]:
    transcription = load_verified_transcription(project_root)
    files = load_verified_sources(project_root)
    audits = {row.method: row for row in audit_table_1(transcription)}
    pipeline = audit_pipeline(transcription, 4)
    source_rows = audit_released_source(project_root)
    table_1 = {
        f"{method}_total_u": fraction_text(audits[method].total_u)
        for method in ("bf16", "coat", "agoq")
    }
    table_1["components_u"] = {
        method: _components(audits, method) for method in ("bf16", "coat", "agoq")
    }
    pipeline_observation = {
        "printed_equation": transcription["pipeline"]["printed_equation"],
        "equation_order_counts": list(pipeline.equation_order_counts),
        "device_order_counts": list(pipeline.device_order_counts),
        "raw_bits": _fraction_sequence(
            tuple(stage.raw_bits for stage in pipeline.stages)
        ),
        "reported_bits": [stage.reported_bits for stage in pipeline.stages],
        "reported_storage_units": [
            stage.reported_storage_units for stage in pipeline.stages
        ],
        "target_storage_units": pipeline.target_storage_units,
        "maximum_reported_storage_units": pipeline.maximum_reported_storage_units,
        "maximum_reported_overshoot_units": (pipeline.maximum_reported_overshoot_units),
        "reported_rounding_rule_available": (pipeline.reported_rounding_rule_available),
    }
    limitations = {
        "single_gpu_fused_kernel_body": (
            "Call sites are present, but a fused GPU kernel implementation body "
            "is not present in the pinned selected source."
        ),
        "claim-1": (
            "The source path is verified, but no distributed training run was "
            "performed."
        ),
        "claim-2": (
            "This is exact arithmetic over paper-transcribed Table 1 components, "
            "not a measured runtime allocation."
        ),
        "claim-3": (
            "The paper does not specify the integer rounding policy; its reported "
            "allocation overshoots the nominal target by one storage unit."
        ),
        "claim-4": (
            "Call-site adjacency is present, but the fused kernel body and measured "
            "overhead reduction are unavailable."
        ),
        "claim-5": (
            "Unavailable: Table 2 requires 64 GPUs and no distributed training was run."
        ),
        "claim-6": (
            "Unavailable: Table 3 requires 16 NVIDIA Blackwell GPUs and no training "
            "was run."
        ),
    }
    bases = (
        [
            "activation_quantization_integration",
            "local_gradient_accumulation",
            "all_to_all_reduce_all_gather_path",
        ],
        ["table_1_exact_rational_arithmetic"],
        ["pipeline_equation_and_reported_allocation_audit"],
        ["single_gpu_fused_kernel_body"],
        [],
        [],
    )
    statuses = (
        "partial",
        "partial",
        "partial",
        "partial",
        "unavailable",
        "unavailable",
    )
    claims = [
        {
            "claim_id": f"claim-{index}",
            "claim": claim,
            "challenge_claim_sha256": claim_sha,
            "status": status,
            "evidence_basis": basis,
            "limitation": limitations[f"claim-{index}"],
        }
        for index, (claim, claim_sha, status, basis) in enumerate(
            zip(LIVE_CLAIMS, CLAIM_SHA256, statuses, bases, strict=True), start=1
        )
    ]
    return {
        "schema_version": 3,
        "identity": {
            "attempt_id": ATTEMPT_ID,
            "paper_id": PAPER_ID,
            "paper_title": (
                "AGoQ: Activation and Gradient Quantization for Memory-Efficient "
                "Distributed Training of LLMs"
            ),
            "submission_slug": (
                "agoq-activation-and-gradient-quantization-for-memory-efficient-"
                "distributed-training-of-llms"
            ),
        },
        "upstream": {
            "paper": transcription["paper"],
            "repository": REPOSITORY,
            "commit": COMMIT,
            "files": [
                {
                    "path": item.path,
                    "size_bytes": item.size_bytes,
                    "git_blob": item.git_blob,
                    "sha256": item.sha256,
                }
                for item in files
            ],
        },
        "paper_context": {
            "label": "Paper-transcribed context; not a reproduced measurement.",
            "table_1": table_1,
            "pipeline": {
                "stored_batches_device_order_n4": transcription["pipeline"][
                    "stored_batches_device_order_n4"
                ],
                "reported_bits_device_order_n4": transcription["pipeline"][
                    "reported_bits_device_order_n4"
                ],
            },
            "training_tables": transcription["training_tables"],
        },
        "reproduced_observations": {
            "label": "Deterministic arithmetic and pinned-source observations.",
            "table_1": table_1,
            "pipeline": pipeline_observation,
            "source_audit": [
                {
                    "observation_id": row.observation_id,
                    "disposition": row.disposition,
                    "files": list(row.files),
                    "symbol_names": list(row.symbol_names),
                    "detail": row.detail,
                }
                for row in source_rows
            ],
        },
        "claims": claims,
        "limitations": limitations,
    }


def canonical_json_bytes(evidence: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_evidence(project_root: Path, output: Path) -> None:
    payload = canonical_json_bytes(build_evidence(project_root))
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def evidence_sha256(evidence: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
