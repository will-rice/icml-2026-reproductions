from pathlib import Path
import pytest
import tempfile
from recurrent_sampler_repro.evidence import verify_provenance

ATTRIBUTION_SHA256 = "79775b50c72988b90eae75ef87e9d4df9dbd0bfceefaed60b398656a88d8a735"


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_provenance_verification():
    project_root = get_project_root()
    res = verify_provenance(project_root)

    assert res["claim_1"]["verified"] is True
    assert res["claim_2"]["verified"] is True
    assert res["inputs"]["arxiv_submission.tex"]["verified"] is True
    assert res["inputs"]["raven_modeling_minimal.py"]["verified"] is True
    assert res["inputs"]["LICENSE"]["verified"] is True
    assert res["inputs"]["ATTRIBUTION.md"]["verified"] is True
    assert res["inputs"]["ATTRIBUTION.md"]["sha256"] == ATTRIBUTION_SHA256
    assert res["inputs"]["ATTRIBUTION.md"]["sha256"] == "79775b50c72988b90eae75ef87e9d4df9dbd0bfceefaed60b398656a88d8a735"
    assert res["inputs"]["LICENSE"]["license"] == "Apache-2.0"
    assert res["inputs"]["ATTRIBUTION.md"]["license"] == "CC-BY-4.0"
    assert "arxiv:2510.14961v1" in res["provenance_token"]


def test_provenance_attribution_mutation_rejection():
    project_root = get_project_root()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "vendor" / "arxiv").mkdir(parents=True)
        (tmp_path / "vendor" / "recurrent-pretraining" / "recpre").mkdir(parents=True)

        tex_orig = (project_root / "vendor" / "arxiv" / "arxiv_submission.tex").read_bytes()
        sampler_orig = (project_root / "vendor" / "recurrent-pretraining" / "recpre" / "raven_modeling_minimal.py").read_bytes()
        license_orig = (project_root / "vendor" / "recurrent-pretraining" / "LICENSE").read_bytes()
        attr_orig = (project_root / "vendor" / "arxiv" / "ATTRIBUTION.md").read_bytes()

        (tmp_path / "vendor" / "arxiv" / "arxiv_submission.tex").write_bytes(tex_orig)
        (tmp_path / "vendor" / "recurrent-pretraining" / "recpre" / "raven_modeling_minimal.py").write_bytes(sampler_orig)
        (tmp_path / "vendor" / "recurrent-pretraining" / "LICENSE").write_bytes(license_orig)
        (tmp_path / "vendor" / "arxiv" / "ATTRIBUTION.md").write_bytes(attr_orig + b"\n# Mutated")

        with pytest.raises(ValueError, match="ATTRIBUTION.md digest mismatch"):
            verify_provenance(tmp_path)
