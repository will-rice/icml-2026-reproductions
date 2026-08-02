import json

from d2_repro.evidence import (
    CLAIMS,
    UPSTREAM_PINS,
    audit_result_artifacts,
    audit_source_components,
    build_evidence_bundle,
)


def test_source_audit_detects_anyorder_and_stepmerge_paths():
    audit = audit_source_components(
        {
            "diffu-grpo-ao/diffu_grpo_trainer_ao.py": "x_2L position_ids_2L pair_mask F.cross_entropy per_token_logps",
            "diffu-grpo/diffu_grpo_trainer.py": "trajectory trajectory_mask self.args.N _get_per_token_logps_oldandref torch.stack(per_token_logps).sum(dim=0)",
            "diffu-grpo/bash_scripts/gsm8k_d2stepmerge.sh": "--trainer_name d2-StepMerge --N 8 --dataset gsm8k",
            "diffu-grpo-ao/bash_scripts/anyorder_gsm8k_d2anyorder.sh": "--trainer_name d2-AnyOrder --dataset gsm8k",
            "eval/parse_and_get_acc.py": "parse_gsm_answers parse_math_answers parse_countdown_answers parse_sudoku_answers",
        }
    )

    assert audit["anyorder_source"] == "present"
    assert audit["stepmerge_source"] == "present"
    assert audit["rl_script_coverage"] == "present"
    assert audit["evaluation_parser"] == "present"


def test_result_artifact_audit_reports_datasets_but_no_raw_metrics():
    audit = audit_result_artifacts(
        {
            "dataset/countdown_cd3_test.jsonl",
            "dataset/4x4_test_sudoku.csv",
            "eval/bash_scripts/eval_gsm8k_d2stepmerge.sh",
        }
    )

    assert sorted(audit["released_dataset_files"]) == [
        "dataset/4x4_test_sudoku.csv",
        "dataset/countdown_cd3_test.jsonl",
    ]
    assert audit["machine_readable_results"] == []
    assert audit["benchmark_eval_scripts_present"] is True


def test_bundle_records_claim_statuses_and_pins():
    bundle = build_evidence_bundle(
        source_files={
            "diffu-grpo-ao/diffu_grpo_trainer_ao.py": "x_2L position_ids_2L pair_mask F.cross_entropy per_token_logps",
            "diffu-grpo/diffu_grpo_trainer.py": "trajectory trajectory_mask self.args.N _get_per_token_logps_oldandref torch.stack(per_token_logps).sum(dim=0)",
            "diffu-grpo/bash_scripts/gsm8k_d2stepmerge.sh": "--trainer_name d2-StepMerge --N 8 --dataset gsm8k",
            "diffu-grpo-ao/bash_scripts/anyorder_gsm8k_d2anyorder.sh": "--trainer_name d2-AnyOrder --dataset gsm8k",
            "eval/parse_and_get_acc.py": "parse_gsm_answers parse_math_answers parse_countdown_answers parse_sudoku_answers",
        },
        repo_files={"dataset/countdown_cd3_test.jsonl", "eval/bash_scripts/eval_gsm8k_d2stepmerge.sh"},
    )

    assert bundle["paper_id"] == "ldCiNVFt8O"
    assert bundle["snapshot_id"] == "730efd6146ac7814c07bd5e2d3908fb59c0435dd83f013b261493ce06c6b3d08"
    assert bundle["upstream_pins"] == UPSTREAM_PINS
    assert [result["claim_sha256"] for result in bundle["claim_results"]] == [
        claim["challenge_claim_sha256"] for claim in CLAIMS
    ]
    assert [result["status"] for result in bundle["claim_results"]] == [
        "verified",
        "verified",
        "toy",
        "toy",
        "inconclusive",
        "inconclusive",
    ]
    json.dumps(bundle)
