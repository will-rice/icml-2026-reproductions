from fractions import Fraction

import pytest

from agoq_repro.pipeline_allocator import audit_pipeline
from agoq_repro.provenance import load_verified_transcription


def test_four_stage_equation_and_reported_allocation(project_root):
    result = audit_pipeline(load_verified_transcription(project_root), 4)
    assert result.equation_order_counts == (5, 7, 9, 11)
    assert result.device_order_counts == (11, 9, 7, 5)
    assert tuple(stage.raw_bits for stage in result.stages) == (
        Fraction(4),
        Fraction(44, 9),
        Fraction(44, 7),
        Fraction(44, 5),
    )
    assert tuple(stage.reported_bits for stage in result.stages) == (4, 5, 6, 8)
    assert tuple(stage.reported_storage_units for stage in result.stages) == (
        44,
        45,
        42,
        40,
    )
    assert result.target_storage_units == 44
    assert result.maximum_reported_storage_units == 45
    assert result.maximum_reported_overshoot_units == 1
    assert result.reported_rounding_rule_available is False


def test_non_four_stage_case_has_no_invented_integer_policy(project_root):
    result = audit_pipeline(load_verified_transcription(project_root), 3)
    assert result.equation_order_counts == (4, 6, 8)
    assert result.device_order_counts == (8, 6, 4)
    assert all(stage.reported_bits is None for stage in result.stages)
    assert result.maximum_reported_storage_units is None


@pytest.mark.parametrize("stage_count", [0, -1, True, 2.5])
def test_invalid_stage_count_is_rejected(project_root, stage_count):
    with pytest.raises((TypeError, ValueError), match="stage_count"):
        audit_pipeline(load_verified_transcription(project_root), stage_count)
