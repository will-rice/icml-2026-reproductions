from pathlib import Path

import pytest

from skill_neologisms_repro.evidence import (
    EXPECTED_UPSTREAM_COMMIT,
    TARGET_CLAIMS,
    build_evidence_bundle,
    load_evidence_bundle,
    render_summary_markdown,
)


UPSTREAM_ROOT = Path("/tmp/icml-skill-neologisms-upstream-path").read_text(
    encoding="utf-8"
).strip()


def test_build_evidence_bundle_records_pinned_upstream_and_metadata():
    bundle = build_evidence_bundle(
        source_root=Path(UPSTREAM_ROOT),
        command_log=["pytest tests"],
    )

    assert bundle["paper_id"] == "5VgZUEpK6W"
    assert bundle["attempt_id"] == "6ee240d6-4363-419b-a6e8-d05aae509de4"
    assert bundle["owner"] == "codex-paper-owner-03"
    assert (
        bundle["snapshot_id"]
        == "31492eaeed30c533df53d18a5b536207a39cda717423ad7ff0b1ad2b017a82bc"
    )
    assert bundle["upstream"]["commit"] == EXPECTED_UPSTREAM_COMMIT
    assert bundle["upstream"]["repository"] == (
        "https://github.com/vanderschaarlab/skill-neologisms"
    )
    assert bundle["upstream"]["license"] == "Apache-2.0"
    assert bundle["reproduction_scope"] == "static_artifact_reproduction"


def test_build_evidence_bundle_verifies_skill_token_training_mechanism():
    bundle = build_evidence_bundle(source_root=Path(UPSTREAM_ROOT))
    artifacts = bundle["artifact_observations"]

    assert artifacts["skill_token_model_files"] == [
        "src/models/skill_token_model.py",
        "src/trainer_utils.py",
        "sequence_map_experiment/train_neologisms.py",
    ]
    assert artifacts["skill_token_operations"] == [
        "[SHIFT_RIGHT]",
        "[INVERT_POLARITY]",
    ]
    assert artifacts["skill_token_config"]["length"] == 20
    assert artifacts["skill_token_config"]["init_method"] == "from_pretrain_skills"
    assert artifacts["skill_token_config"]["save_dir"] == "skill_tokens"
    assert artifacts["trainable_embedding_parameter"] is True
    assert artifacts["non_skill_gradient_masking"] is True
    assert artifacts["base_model_parameter_freeze"] is True


def test_build_evidence_bundle_records_skill_centered_data_and_baselines():
    bundle = build_evidence_bundle(source_root=Path(UPSTREAM_ROOT))
    artifacts = bundle["artifact_observations"]

    assert artifacts["skill_centered_dataset_files"] == [
        "data/skill_mix/train_data_modus.csv",
        "data/skill_mix/train_data_stat.csv",
    ]
    assert artifacts["skill_mix_dataset_rows"] == {
        "train_data_modus.csv": 300,
        "train_data_stat.csv": 300,
    }
    assert artifacts["digit_sequence_pretrain_ops"] == [
        "[ASC]",
        "[DESC]",
        "[ADD]",
        "[SUB]",
        "[POLARITY]",
        "[REVERSE]",
        "[ID]",
    ]
    assert artifacts["digit_sequence_new_ops"] == [
        "[SHIFT_RIGHT]",
        "[INVERT_POLARITY]",
    ]
    assert artifacts["baseline_coverage"] == {
        "lora": True,
        "prompt_tuning": True,
        "full_finetuning_or_baseline_training": True,
    }


def test_build_evidence_bundle_records_composition_evaluation_hooks():
    bundle = build_evidence_bundle(source_root=Path(UPSTREAM_ROOT))
    artifacts = bundle["artifact_observations"]

    assert artifacts["digit_sequence_composition_scripts"] == [
        "sequence_map_experiment/evaluate_zs_compo_icl.py",
        "sequence_map_experiment/run_digitseq_all.sh",
    ]
    assert artifacts["composition_eval_skill_names"] == [
        "[SHIFT_RIGHT]",
        "[INVERT_POLARITY]",
    ]
    assert artifacts["composition_eval_num_ops"] == {"min": 2, "max": 2}
    assert artifacts["generate_test_datasets_tests_up_to_ops"] == 3
    assert artifacts["run_all_crosses_skill_and_test_ops"] is True
    assert artifacts["skill_mix_composition_scripts"] == [
        "scripts/skill-mix/eval_skill_mix.py",
        "scripts/skill-mix/eval_baselines.py",
    ]


def test_evidence_bundle_contains_four_selected_claims_with_limitations():
    bundle = load_evidence_bundle()

    assert TARGET_CLAIMS == bundle["target_claims"]
    assert len(bundle["target_claims"]) == 4
    statuses = {claim["status"] for claim in bundle["target_claims"]}
    assert statuses <= {"verified", "partial"}
    assert any(claim["status"] == "partial" for claim in bundle["target_claims"])
    assert all(claim["challenge_claim_sha256"] for claim in bundle["target_claims"])
    assert "paper-reported accuracy" in bundle["limitations"][0]


def test_build_evidence_bundle_rejects_wrong_upstream_commit(tmp_path):
    with pytest.raises(ValueError, match="upstream commit"):
        build_evidence_bundle(source_root=tmp_path)


def test_render_summary_markdown_surfaces_claim_statuses_and_limits():
    markdown = render_summary_markdown(load_evidence_bundle())

    assert "# Skill Neologisms Reproduction Evidence" in markdown
    assert "claim_1_skill_tokens: verified" in markdown
    assert "claim_4_digit_sequence_composition: partial" in markdown
    assert "static artifact reproduction" in markdown
