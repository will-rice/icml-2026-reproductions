from pathlib import Path
import pytest
import tempfile
from recurrent_sampler_repro.evidence import audit_source_ast, SAMPLER_GIT_BLOB


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_source_ast_audit():
    project_root = get_project_root()
    res = audit_source_ast(project_root)

    assert res["git_blob"] == SAMPLER_GIT_BLOB
    assert res["dispatcher"]["function"] == "generate"
    assert res["dispatcher"]["dispatches_to_diffusion_style"] is True

    sampler_info = res["sampler"]
    assert sampler_info["function"] == "generate_diffusion_style"
    assert sampler_info["defaults"]["headway"] == 1
    assert sampler_info["defaults"]["inner_recurrence"] == 4
    assert sampler_info["defaults"]["freeze_strategy"] == "latent-diff"
    assert sampler_info["defaults"]["max_wavefront"] == 128

    ctrl = sampler_info["control_flow"]
    assert ctrl["inner_recurrence_loop_found"] is True
    assert ctrl["latent_diff_check_found"] is True
    assert ctrl["max_wavefront_check_found"] is True
    assert ctrl["headway_extension_found"] is True


def test_source_ast_audit_mutation_failure():
    project_root = get_project_root()
    sampler_bytes = (project_root / "vendor" / "recurrent-pretraining" / "recpre" / "raven_modeling_minimal.py").read_text(encoding="utf-8")

    # Remove generate_diffusion_style method
    mutated_bytes = sampler_bytes.replace("def generate_diffusion_style", "def _disabled_diffusion_style")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "vendor" / "recurrent-pretraining" / "recpre").mkdir(parents=True)
        (tmp_path / "vendor" / "recurrent-pretraining" / "recpre" / "raven_modeling_minimal.py").write_text(mutated_bytes, encoding="utf-8")

        with pytest.raises(ValueError, match="Could not locate `generate_diffusion_style` in AST"):
            audit_source_ast(tmp_path)
