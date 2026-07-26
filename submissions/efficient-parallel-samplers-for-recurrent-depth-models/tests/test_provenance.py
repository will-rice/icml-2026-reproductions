from pathlib import Path
import pytest
import tempfile
from recurrent_sampler_repro.evidence import (
    verify_provenance,
    compute_sha256,
    compute_git_blob,
    TEX_SHA256,
    SAMPLER_SHA256,
    SAMPLER_GIT_BLOB,
    LICENSE_SHA256,
)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_provenance_verification():
    project_root = get_project_root()
    res = verify_provenance(project_root)
    assert res["inputs"]["arxiv_submission.tex"]["sha256"] == TEX_SHA256
    assert res["inputs"]["raven_modeling_minimal.py"]["sha256"] == SAMPLER_SHA256
    assert res["inputs"]["raven_modeling_minimal.py"]["git_blob"] == SAMPLER_GIT_BLOB
    assert res["inputs"]["LICENSE"]["sha256"] == LICENSE_SHA256


def test_provenance_mutation_failure():
    project_root = get_project_root()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create vendor structure
        (tmp_path / "vendor" / "arxiv").mkdir(parents=True)
        (tmp_path / "vendor" / "recurrent-pretraining" / "recpre").mkdir(parents=True)

        # Copy original contents
        tex_orig = (project_root / "vendor" / "arxiv" / "arxiv_submission.tex").read_bytes()
        sampler_orig = (project_root / "vendor" / "recurrent-pretraining" / "recpre" / "raven_modeling_minimal.py").read_bytes()
        license_orig = (project_root / "vendor" / "recurrent-pretraining" / "LICENSE").read_bytes()

        # Write mutated TeX
        (tmp_path / "vendor" / "arxiv" / "arxiv_submission.tex").write_bytes(tex_orig + b"\n")
        (tmp_path / "vendor" / "recurrent-pretraining" / "recpre" / "raven_modeling_minimal.py").write_bytes(sampler_orig)
        (tmp_path / "vendor" / "recurrent-pretraining" / "LICENSE").write_bytes(license_orig)

        with pytest.raises(ValueError, match="TeX digest mismatch"):
            verify_provenance(tmp_path)
