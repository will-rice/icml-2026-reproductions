"""Exact audit of AGoQ's printed pipeline compensation arithmetic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class PipelineStageAudit:
    device_index: int
    stored_batches: int
    raw_bits: Fraction
    reported_bits: int | None
    reported_storage_units: int | None


@dataclass(frozen=True)
class PipelineAudit:
    stage_count: int
    equation_order_counts: tuple[int, ...]
    device_order_counts: tuple[int, ...]
    stages: tuple[PipelineStageAudit, ...]
    target_storage_units: int
    maximum_reported_storage_units: int | None
    maximum_reported_overshoot_units: int | None
    reported_rounding_rule_available: bool


def audit_pipeline(
    transcription: Mapping[str, object], stage_count: int
) -> PipelineAudit:
    if type(stage_count) is not int:
        raise TypeError("stage_count must be an integer")
    if stage_count <= 0:
        raise ValueError("stage_count must be positive")
    pipeline = transcription.get("pipeline")
    if type(pipeline) is not dict:
        raise ValueError("pipeline transcription is missing")
    minimum_bits = Fraction(str(pipeline.get("minimum_bits")))
    if minimum_bits <= 0 or minimum_bits.denominator != 1:
        raise ValueError("minimum_bits must be a positive integer")

    equation_order = tuple(
        stage_count + 2 * index - 1 for index in range(1, stage_count + 1)
    )
    device_order = tuple(reversed(equation_order))
    target = int(minimum_bits) * device_order[0]

    reported: tuple[int, ...] | None = None
    if stage_count == 4:
        transcribed_counts = pipeline.get("stored_batches_device_order_n4")
        transcribed_bits = pipeline.get("reported_bits_device_order_n4")
        if transcribed_counts != list(device_order):
            raise ValueError("four-stage stored-batch transcription mismatch")
        if (
            type(transcribed_bits) is not list
            or len(transcribed_bits) != 4
            or any(type(value) is not int or value <= 0 for value in transcribed_bits)
        ):
            raise ValueError("four-stage reported-bit transcription")
        reported = tuple(transcribed_bits)

    stages: list[PipelineStageAudit] = []
    for offset, stored_batches in enumerate(device_order):
        reported_bits = reported[offset] if reported else None
        stages.append(
            PipelineStageAudit(
                device_index=offset + 1,
                stored_batches=stored_batches,
                raw_bits=minimum_bits * device_order[0] / stored_batches,
                reported_bits=reported_bits,
                reported_storage_units=(
                    reported_bits * stored_batches
                    if reported_bits is not None
                    else None
                ),
            )
        )
    products = tuple(
        stage.reported_storage_units
        for stage in stages
        if stage.reported_storage_units is not None
    )
    maximum = max(products) if products else None
    return PipelineAudit(
        stage_count=stage_count,
        equation_order_counts=equation_order,
        device_order_counts=device_order,
        stages=tuple(stages),
        target_storage_units=target,
        maximum_reported_storage_units=maximum,
        maximum_reported_overshoot_units=maximum - target
        if maximum is not None
        else None,
        reported_rounding_rule_available=False,
    )
