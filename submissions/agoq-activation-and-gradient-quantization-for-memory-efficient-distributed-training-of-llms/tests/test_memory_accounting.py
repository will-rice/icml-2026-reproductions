from fractions import Fraction

import pytest

from agoq_repro.memory_accounting import (
    audit_table_1,
    fraction_text,
    project_model,
)
from agoq_repro.provenance import load_verified_transcription


def test_table_1_totals_are_recomputed(project_root):
    source = load_verified_transcription(project_root)
    audits = {row.method: row for row in audit_table_1(source)}
    assert audits["bf16"].total_u == Fraction(28)
    assert audits["coat"].total_u == Fraction(33, 2)
    assert audits["agoq"].total_u == Fraction(31, 4)
    assert audits["agoq"].components_u["linear_1"] == Fraction(1, 4)


def test_model_projection_uses_u_definition(project_root):
    audits = audit_table_1(load_verified_transcription(project_root))
    result = project_model(audits, batch=2, sequence=4096, hidden=8192, layers=32)
    assert result.bytes_per_u == 134_217_728
    assert result.totals_bytes["bf16"] == 120_259_084_288
    assert result.totals_bytes["coat"] == 70_866_960_384
    assert result.totals_bytes["agoq"] == 33_285_996_544


@pytest.mark.parametrize("field", ["batch", "sequence", "hidden", "layers"])
def test_projection_rejects_nonpositive_dimensions(project_root, field):
    kwargs = {"batch": 1, "sequence": 1, "hidden": 1, "layers": 1}
    kwargs[field] = 0
    with pytest.raises(ValueError, match=field):
        project_model(
            audit_table_1(load_verified_transcription(project_root)), **kwargs
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(Fraction(28), "28"), (Fraction(33, 2), "33/2")],
)
def test_fraction_text_is_canonical(value, expected):
    assert fraction_text(value) == expected
