from pathlib import Path
import pytest
import tempfile
from recurrent_sampler_repro.evidence import audit_source_ast, SAMPLER_GIT_BLOB


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_source_ast_audit_derived_operation_order():
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
    assert ctrl["operation_order_valid"] is True
    assert ctrl["operations"] == [
        "recurrent_iterate",
        "prediction_logits",
        "sampling",
        "state_append",
        "prefix_max_wavefront_truncation",
        "latent_diff_freezing",
    ]
    assert ctrl["latent_diff_normalized_predicate_found"] is True


def test_source_ast_audit_mutation_missing_iterate_fails():
    project_root = get_project_root()
    sampler_bytes = (
        project_root / "vendor" / "recurrent-pretraining" / "recpre" / "raven_modeling_minimal.py"
    ).read_text(encoding="utf-8")

    mutated_bytes = sampler_bytes.replace("self.iterate_one_step", "self._disabled_iterate")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "vendor" / "recurrent-pretraining" / "recpre").mkdir(parents=True)
        (tmp_path / "vendor" / "recurrent-pretraining" / "recpre" / "raven_modeling_minimal.py").write_text(
            mutated_bytes, encoding="utf-8"
        )

        with pytest.raises(ValueError, match="Missing or reordered sampler operation"):
            audit_source_ast(tmp_path)


def test_source_ast_audit_mutation_unrelated_norm_fails():
    project_root = get_project_root()
    sampler_bytes = (
        project_root / "vendor" / "recurrent-pretraining" / "recpre" / "raven_modeling_minimal.py"
    ).read_text(encoding="utf-8")

    # Replace normalized latent diff with unnormalized norm
    old_line = 'criterion = (match_states - matching_prev_states).norm(dim=-1) / match_states.norm(dim=-1)'
    new_line = 'criterion = (match_states - matching_prev_states).norm(dim=-1)'
    mutated_bytes = sampler_bytes.replace(old_line, new_line)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "vendor" / "recurrent-pretraining" / "recpre").mkdir(parents=True)
        (tmp_path / "vendor" / "recurrent-pretraining" / "recpre" / "raven_modeling_minimal.py").write_text(
            mutated_bytes, encoding="utf-8"
        )

        with pytest.raises(ValueError, match="Normalized latent-difference freezing predicate not found"):
            audit_source_ast(tmp_path)
