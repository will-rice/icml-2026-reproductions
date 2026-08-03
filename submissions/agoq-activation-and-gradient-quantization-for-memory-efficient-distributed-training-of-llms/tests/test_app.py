import pytest

from app import (
    PROJECT_ROOT,
    create_demo,
    evidence_summary,
    load_committed_evidence,
    model_projection,
)


def test_space_metadata_and_launch_contract():
    readme = (PROJECT_ROOT / "README.md").read_text()
    app = (PROJECT_ROOT / "app.py").read_text()
    assert "paper-ymHDVBwmta" in readme
    assert "icml2026-repro" in readme
    assert 'server_name="0.0.0.0"' in app
    assert "server_port=7860" in app


def test_space_uses_committed_evidence_without_random_simulation():
    source = (PROJECT_ROOT / "app.py").read_text()
    assert "evidence.json" in source
    assert "quantization_sim" not in source
    assert "numpy" not in source
    claims, memory, limitation = evidence_summary()
    assert [row[0] for row in claims] == [
        "claim-1",
        "claim-2",
        "claim-3",
        "claim-4",
        "claim-5",
        "claim-6",
    ]
    assert all(len(row[2]) == 64 for row in claims)
    assert any(row[:2] == ["agoq", "31/4"] for row in memory)
    assert "one-unit" in limitation


def test_model_projection_accepts_only_positive_integer_inputs():
    result = model_projection(2, 4096, 8192, 32)
    assert result["bytes_per_u"] == 134_217_728
    assert result["totals_bytes"]["agoq"] == "33285996544"
    for values in (
        (0, 4096, 8192, 32),
        (2, 4096.5, 8192, 32),
        (True, 4096, 8192, 32),
    ):
        with pytest.raises(ValueError, match="positive integers"):
            model_projection(*values)


def test_docs_do_not_present_training_tables_as_reproduced():
    for name in ("README.md", "POSTER.md"):
        text = (PROJECT_ROOT / name).read_text()
        assert "unavailable" in text.lower()
        assert "64 GPUs" in text
        assert "16 NVIDIA Blackwell GPUs" in text
        assert "reproduced throughput" not in text.lower()


def test_committed_evidence_is_canonical_and_demo_constructs():
    evidence = load_committed_evidence()
    assert evidence["identity"]["paper_id"] == "ymHDVBwmta"
    assert create_demo() is not None


def test_tampered_evidence_is_rejected(tmp_path):
    tampered = tmp_path / "evidence.json"
    tampered.write_bytes((PROJECT_ROOT / "evidence.json").read_bytes() + b"\n")
    try:
        load_committed_evidence(tampered)
    except ValueError as exc:
        assert "canonical" in str(exc)
    else:
        raise AssertionError("tampered evidence was accepted")
