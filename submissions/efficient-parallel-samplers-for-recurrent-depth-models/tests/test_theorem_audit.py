from pathlib import Path
import pytest
import tempfile
from recurrent_sampler_repro.evidence import audit_theorem


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_theorem_audit_document_order():
    project_root = get_project_root()
    res = audit_theorem(project_root)

    assert res["evidence_status"] == "unavailable"
    assert res["proof_reproduced"] is False
    assert res["challenge_citation"] == "Theorem 4.2"

    thms = res["theorems"]
    assert thms["definition_4_1"]["statement_found"] is True
    assert thms["theorem_4_2"]["statement_found"] is True
    assert thms["theorem_4_2"]["scope"] == "prefilling"

    assert thms["remark_4_3"]["statement_found"] is True

    assert thms["theorem_4_4"]["statement_found"] is True
    assert thms["theorem_4_4"]["scope"] == "decoding"
    assert "d_{\\text{DF}}(T) = d_{\\text{AR}}(T)" in thms["theorem_4_4"]["statement"]
    assert any("r > 1" in a for a in thms["theorem_4_4"]["assumptions"])
    assert any("W \\le L" in a or "W \\leq L" in a for a in thms["theorem_4_4"]["assumptions"])
    assert thms["theorem_4_4"]["has_proof_environment"] is False

    assert thms["remark_4_5"]["statement_found"] is True

    audit = res["citation_audit"]
    assert audit["citation_mismatch_detected"] is True

    assert res["document_order"] == [
        "Definition 4.1",
        "Theorem 4.2",
        "Remark 4.3",
        "Theorem 4.4",
        "Remark 4.5",
    ]


def test_theorem_audit_mutation_missing_decoding_theorem_fails():
    project_root = get_project_root()
    tex_bytes = (project_root / "vendor" / "arxiv" / "arxiv_submission.tex").read_text(encoding="utf-8")

    mutated = tex_bytes.replace("Depth vs. Width Scaling in Decoding", "Disabled Decoding Theorem")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "vendor" / "arxiv").mkdir(parents=True)
        (tmp_path / "vendor" / "arxiv" / "arxiv_submission.tex").write_text(mutated, encoding="utf-8")

        with pytest.raises(ValueError, match="Failed closed: Theorem 4.4 decoding statement not found"):
            audit_theorem(tmp_path)


def test_theorem_audit_mutation_reordering_fails():
    project_root = get_project_root()
    tex_bytes = (project_root / "vendor" / "arxiv" / "arxiv_submission.tex").read_text(encoding="utf-8")

    # Swap Theorem 4.2 and Theorem 4.4 order in TeX
    t42_pos = tex_bytes.find("\\begin{theorem}[Depth vs. Width Scaling in Prefilling")
    t44_pos = tex_bytes.find("\\begin{theorem}[Depth vs. Width Scaling in Decoding")

    assert t42_pos != -1 and t44_pos != -1
    t42_end = tex_bytes.find("\\end{theorem}", t42_pos) + len("\\end{theorem}")
    t44_end = tex_bytes.find("\\end{theorem}", t44_pos) + len("\\end{theorem}")

    t42_block = tex_bytes[t42_pos:t42_end]
    t44_block = tex_bytes[t44_pos:t44_end]

    mutated = tex_bytes[:t42_pos] + t44_block + tex_bytes[t42_end:t44_pos] + t42_block + tex_bytes[t44_end:]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "vendor" / "arxiv").mkdir(parents=True)
        (tmp_path / "vendor" / "arxiv" / "arxiv_submission.tex").write_text(mutated, encoding="utf-8")

        with pytest.raises(ValueError, match="Section 4 structure mismatch or reordering detected"):
            audit_theorem(tmp_path)
