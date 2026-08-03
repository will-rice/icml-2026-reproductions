from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PATH = PROJECT_ROOT / "evidence" / "bundle.json"

EXPECTED_UPSTREAM_COMMIT = "d24f3851d1e10cf52a621ca439332c92062360e5"
UPSTREAM_REPOSITORY = "https://github.com/vanderschaarlab/skill-neologisms"

TARGET_CLAIMS = [
    {
        "claim_id": "claim_1_skill_tokens",
        "challenge_claim_sha256": (
            "3854fd0dd63c46d288b124aad9646ddf389e7c9e1a545f888c046a5a1a284c35"
        ),
        "claim": (
            "Skill neologisms extend a model vocabulary with trainable skill-token "
            "embeddings while keeping the pretrained model parameters fixed."
        ),
        "status": "verified",
        "provenance": (
            "The released SkillTokenModel registers skill-specific soft token rows, "
            "marks the embedding matrix trainable, uses an optimizer over that "
            "embedding matrix only, and the SkillTokenTrainer masks gradients for "
            "all non-skill token rows during training."
        ),
    },
    {
        "claim_id": "claim_2_skill_centered_datasets",
        "challenge_claim_sha256": (
            "9e14ca8c7f97d9e2946a4d3069a036cdf4926c1f844d9e49fe6e8e663b6ca191"
        ),
        "claim": (
            "The method is evaluated with skill-centered datasets in which a new "
            "skill is composed with skills already represented in the pretrained model."
        ),
        "status": "verified",
        "provenance": (
            "Digit-sequence data generation separates seven pretraining operations "
            "from SHIFT_RIGHT and INVERT_POLARITY, and training data always includes "
            "the selected new skill with sampled other operations. The SkillMix "
            "release also includes Modus Ponens and Statistical Syllogism training "
            "CSV files and configs."
        ),
    },
    {
        "claim_id": "claim_3_baseline_coverage",
        "challenge_claim_sha256": (
            "0233b25a01dfcb99d57b734092f96ad059f555578772c7815b5276c3c350b0ab"
        ),
        "claim": (
            "The released experiments compare skill neologisms against LoRA, prompt "
            "tuning, and related baseline training paths."
        ),
        "status": "verified",
        "provenance": (
            "The digit-sequence runner executes skill neologisms, prompt tuning, "
            "and LoRA for each skill/test operation pair, and the SkillMix configs "
            "and scripts contain matching LoRA and prompt-tuning baseline commands."
        ),
    },
    {
        "claim_id": "claim_4_digit_sequence_composition",
        "challenge_claim_sha256": (
            "bb8e78c2fa3716b338adfb7e30bca948f8d987e2522c5faec82419c33f3075cc"
        ),
        "claim": (
            "On digit-sequence tasks, independently learned skill neologisms transfer "
            "to held-out two-skill and three-skill compositions."
        ),
        "status": "partial",
        "provenance": (
            "The released code defines the zero-shot composition experiment for "
            "SHIFT_RIGHT and INVERT_POLARITY and the data generator can construct "
            "tests through three operations. This reproduction verifies those "
            "artifacts but does not rerun Qwen training/evaluation or reproduce "
            "paper-reported accuracies."
        ),
    },
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _git_commit(source_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("upstream commit could not be read") from exc
    return result.stdout.strip()


def _expect_pinned_commit(source_root: Path) -> str:
    commit = _git_commit(source_root)
    if commit != EXPECTED_UPSTREAM_COMMIT:
        raise ValueError(
            f"upstream commit mismatch: expected {EXPECTED_UPSTREAM_COMMIT}, got {commit}"
        )
    return commit


def _license_id(source_root: Path) -> str:
    license_text = _read_text(source_root / "LICENSE")
    if "Apache License" in license_text and "Version 2.0" in license_text:
        return "Apache-2.0"
    return "unknown"


def _parse_assignment(source: str, name: str) -> Any:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(target, "id", None) == name for target in node.targets):
            continue
        if isinstance(node.value, ast.Dict):
            return [
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
        return ast.literal_eval(node.value)
    raise ValueError(f"assignment {name!r} not found")


def _parse_yaml_scalar(source: str, section: str, key: str) -> str | int:
    pattern = rf"^{re.escape(section)}:\n(?P<body>(?:  .+\n)+)"
    match = re.search(pattern, source, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"YAML section {section!r} not found")
    for line in match.group("body").splitlines():
        key_match = re.match(rf"\s+{re.escape(key)}:\s*(.+?)(?:\s+#.*)?$", line)
        if not key_match:
            continue
        value = key_match.group(1).strip().strip("\"'")
        if value.isdigit():
            return int(value)
        return value
    raise ValueError(f"YAML key {section}.{key} not found")


def _csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _has_all(source: str, snippets: list[str]) -> bool:
    return all(snippet in source for snippet in snippets)


def _upstream_file(path: Path, source_root: Path) -> str:
    return str(path.relative_to(source_root)).replace("\\", "/")


def build_evidence_bundle(
    *, source_root: Path | str, command_log: list[str] | None = None
) -> dict[str, Any]:
    source_root = Path(source_root)
    commit = _expect_pinned_commit(source_root)

    skill_model_path = source_root / "src" / "models" / "skill_token_model.py"
    trainer_path = source_root / "src" / "trainer_utils.py"
    data_path = source_root / "sequence_map_experiment" / "data.py"
    skill_config_path = (
        source_root / "sequence_map_experiment" / "configs" / "skill_tokens.yaml"
    )
    train_neologisms_path = (
        source_root / "sequence_map_experiment" / "train_neologisms.py"
    )
    train_baselines_path = (
        source_root / "sequence_map_experiment" / "train_baselines.py"
    )
    train_prompt_tuning_path = (
        source_root / "sequence_map_experiment" / "train_prompt_tuning.py"
    )
    eval_composition_path = (
        source_root / "sequence_map_experiment" / "evaluate_zs_compo_icl.py"
    )
    run_all_path = source_root / "sequence_map_experiment" / "run_digitseq_all.sh"
    skill_mix_eval_path = source_root / "scripts" / "skill-mix" / "eval_skill_mix.py"
    skill_mix_eval_baselines_path = (
        source_root / "scripts" / "skill-mix" / "eval_baselines.py"
    )

    skill_model = _read_text(skill_model_path)
    trainer = _read_text(trainer_path)
    data_source = _read_text(data_path)
    skill_config = _read_text(skill_config_path)
    train_baselines = _read_text(train_baselines_path)
    train_prompt_tuning = _read_text(train_prompt_tuning_path)
    eval_composition = _read_text(eval_composition_path)
    run_all = _read_text(run_all_path)

    all_ops = _parse_assignment(data_source, "ALL_OPS")
    pretrain_ops = _parse_assignment(data_source, "PRETRAIN_OPS")
    new_ops = [op for op in all_ops if op not in pretrain_ops]
    skill_mix_csvs = [
        source_root / "data" / "skill_mix" / "train_data_modus.csv",
        source_root / "data" / "skill_mix" / "train_data_stat.csv",
    ]

    artifacts = {
        "skill_token_model_files": [
            _upstream_file(skill_model_path, source_root),
            _upstream_file(trainer_path, source_root),
            _upstream_file(train_neologisms_path, source_root),
        ],
        "skill_token_operations": new_ops,
        "skill_token_config": {
            "length": _parse_yaml_scalar(skill_config, "skill", "length"),
            "init_method": _parse_yaml_scalar(skill_config, "skill", "init_method"),
            "save_dir": _parse_yaml_scalar(skill_config, "output", "save_dir"),
        },
        "trainable_embedding_parameter": (
            "self.embedding_weights.requires_grad = True" in skill_model
        ),
        "non_skill_gradient_masking": _has_all(
            skill_model,
            ["def zero_out_non_skill_grads", "grad[~mask] = 0"],
        )
        and "self.adapter.zero_out_non_skill_grads(input_ids)" in trainer,
        "base_model_parameter_freeze": _has_all(
            skill_model,
            ["return torch.optim.Adam([self.embedding_weights]", "get_optimizer"],
        )
        and "self.optimizer = self.adapter.get_optimizer" in trainer,
        "skill_centered_dataset_files": [
            _upstream_file(path, source_root) for path in skill_mix_csvs
        ],
        "skill_mix_dataset_rows": {
            path.name: _csv_row_count(path) for path in skill_mix_csvs
        },
        "skill_mix_dataset_configs": [
            "configs/skill_mix/dataset/main/modus.yaml",
            "configs/skill_mix/dataset/main/stat_syllogism.yaml",
        ],
        "digit_sequence_pretrain_ops": pretrain_ops,
        "digit_sequence_new_ops": new_ops,
        "skill_data_generator_always_includes_new_skill": _has_all(
            data_source,
            ["def generate_sample_data_skill", "sample_ops.append(random.choice(main_ops))"],
        ),
        "baseline_coverage": {
            "lora": (
                "peft.type" in train_baselines
                and (source_root / "sequence_map_experiment" / "configs" / "baseline_lora.yaml").exists()
                and (source_root / "configs" / "skill_mix" / "train_baseline_lora.yaml").exists()
            ),
            "prompt_tuning": (
                "PREFIX" in train_prompt_tuning
                and (source_root / "sequence_map_experiment" / "configs" / "baseline_prompt_tuning.yaml").exists()
                and (source_root / "configs" / "skill_mix" / "train_baseline_pt.yaml").exists()
            ),
            "full_finetuning_or_baseline_training": (
                "Train prompt tuning baseline" in train_prompt_tuning
                and "Training script for baseline PEFT models" in train_baselines
            ),
        },
        "digit_sequence_composition_scripts": [
            _upstream_file(eval_composition_path, source_root),
            _upstream_file(run_all_path, source_root),
        ],
        "composition_eval_skill_names": _parse_assignment(eval_composition, "SKILL_NAMES"),
        "composition_eval_num_ops": {
            "min": _parse_assignment(eval_composition, "MIN_OPS"),
            "max": _parse_assignment(eval_composition, "MAX_OPS"),
        },
        "generate_test_datasets_tests_up_to_ops": (
            3 if "for num_ops in range(2, 4)" in data_source else None
        ),
        "run_all_crosses_skill_and_test_ops": _has_all(
            run_all,
            [
                'skill_ops=("[SHIFT_RIGHT]" "[INVERT_POLARITY]")',
                'test_ops=("[ADD]" "[SUB]" "[ASC]" "[DESC]" "[ID]" "[POLARITY]")',
                "train_neologisms.py",
                "train_prompt_tuning.py",
                "train_baselines.py",
            ],
        ),
        "skill_mix_composition_scripts": [
            _upstream_file(skill_mix_eval_path, source_root),
            _upstream_file(skill_mix_eval_baselines_path, source_root),
        ],
    }

    return {
        "paper_id": "5VgZUEpK6W",
        "title": "Skill Neologisms: Towards Skill-based Continual Learning",
        "attempt_id": "6ee240d6-4363-419b-a6e8-d05aae509de4",
        "owner": "codex-paper-owner-03",
        "snapshot_id": "31492eaeed30c533df53d18a5b536207a39cda717423ad7ff0b1ad2b017a82bc",
        "reproduction_scope": "static_artifact_reproduction",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": commit,
            "license": _license_id(source_root),
        },
        "target_claims": TARGET_CLAIMS,
        "artifact_observations": artifacts,
        "limitations": [
            "No paper-reported accuracy, pass rate, or table value is counted as reproduced.",
            "No Qwen or LLaMA training/evaluation run is executed in this CPU-only artifact check.",
            "No WandB history, Azure OpenAI generation, or GPT-5 grading output is imported.",
        ],
        "commands": command_log or [],
    }


def write_evidence_bundle(bundle: dict[str, Any], path: Path | None = None) -> Path:
    output_path = path or EVIDENCE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_evidence_bundle(path: Path | None = None) -> dict[str, Any]:
    input_path = path or EVIDENCE_PATH
    return json.loads(input_path.read_text(encoding="utf-8"))


def render_summary_markdown(bundle: dict[str, Any]) -> str:
    claim_lines = [
        f"- {claim['claim_id']}: {claim['status']} - {claim['provenance']}"
        for claim in bundle["target_claims"]
    ]
    artifact_lines = [
        f"- upstream commit: {bundle['upstream']['commit']}",
        f"- reproduction scope: {bundle['reproduction_scope'].replace('_', ' ')}",
        (
            "- digit sequence operations: "
            + ", ".join(bundle["artifact_observations"]["digit_sequence_new_ops"])
        ),
        (
            "- SkillMix CSV rows: "
            + json.dumps(bundle["artifact_observations"]["skill_mix_dataset_rows"], sort_keys=True)
        ),
    ]
    limitation_lines = [f"- {item}" for item in bundle["limitations"]]

    return "\n".join(
        [
            "# Skill Neologisms Reproduction Evidence",
            "",
            f"Paper ID: `{bundle['paper_id']}`",
            f"Attempt ID: `{bundle['attempt_id']}`",
            "",
            "## Claim Status",
            *claim_lines,
            "",
            "## Artifact Observations",
            *artifact_lines,
            "",
            "## Limitations",
            *limitation_lines,
            "",
        ]
    )
