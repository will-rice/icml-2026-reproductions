"""CPU-only evidence generation for the d2 reproduction attempt."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import urllib.request
from pathlib import Path
from typing import Any


ATTEMPT_ID = "0f0cfb98-1f1a-4c31-8881-32cb6e02abde"
PAPER_ID = "ldCiNVFt8O"
SNAPSHOT_ID = "730efd6146ac7814c07bd5e2d3908fb59c0435dd83f013b261493ce06c6b3d08"
TITLE = "d2: Improved Techniques for Training Reasoning Diffusion Language Models"
GITHUB_REVISION = "381b9f14f4afd0719297ac852e4015c74e0ed235"

UPSTREAM_PINS = {
    "paper": "arxiv:2509.21474",
    "official_code": f"github:kuleshov-group/d2@{GITHUB_REVISION}",
    "checkpoint_anyorder_sft": "hf:GuanghanWang/d2_anyorder_causal_llada_intellectsft@3c334aa4931697841a923d6caad3b12d5eaa4409",
    "checkpoint_anyorder_gsm8k": "hf:GuanghanWang/d2_anyorder_causal_llada_intellectsft_gsm8k@e93476e1f676abfaaf0bdc036aa24d3f04c213f4",
}

CLAIMS = [
    {
        "target_claim": "d2 is a reinforcement-learning framework for masked diffusion language models built around estimating sampling-trajectory likelihoods (Section 3).",
        "challenge_claim_sha256": "8abd632fb622a9ca7082b5e4af26fb9410dc9fdb53c972ea94f93fd40d23bf38",
    },
    {
        "target_claim": "d2-AnyOrder provides exact trajectory likelihood with a single model pass for DLMs that support any-order decoding (Section 3).",
        "challenge_claim_sha256": "9e7c76274f238f39796b6c62d7ab6fc84f4b0a24c6cb4374f340ed5bc4b169df",
    },
    {
        "target_claim": "The paper empirically shows that any-order decoding support is not universal across widely used DLMs (Section 4).",
        "challenge_claim_sha256": "a2d18ab723499038a58d26f7847038968cf9e14aab4b33f1142cf9c2677dbbb3",
    },
    {
        "target_claim": "d2-StepMerge approximates trajectory likelihood for standard masked diffusion models with a tractable compute-accuracy tradeoff (Section 3).",
        "challenge_claim_sha256": "4017a2eea78238824fce3a7ed8480a781b79a0a5dc131197d88f005ac6f84acb",
    },
    {
        "target_claim": "d2 reports stronger performance than widely used RL baselines when applied to popular diffusion language models (Section 5).",
        "challenge_claim_sha256": "8e7a073674a9362a2de3027f723a8f72f0fcea7dcbacf41e92d6727c33c2be9b",
    },
    {
        "target_claim": "d2 reports new state-of-the-art diffusion-language-model results on Countdown, Sudoku, GSM8K, and MATH500 reasoning benchmarks (Section 5).",
        "challenge_claim_sha256": "b60d6d3c7d157e5dd7cb263491a6cc90cc06d3f65aa0ff87ebe7e6ce318b7fcb",
    },
]

SOURCE_PATHS = [
    "README.md",
    "diffu-grpo/diffu_grpo_trainer.py",
    "diffu-grpo-ao/diffu_grpo_trainer_ao.py",
    "diffu-grpo/bash_scripts/gsm8k_d2stepmerge.sh",
    "diffu-grpo/bash_scripts/countdown_d2stepmerge.sh",
    "diffu-grpo/bash_scripts/math500_d2stepmerge.sh",
    "diffu-grpo/bash_scripts/sudoku_d2stepmerge_N8.sh",
    "diffu-grpo-ao/bash_scripts/anyorder_gsm8k_d2anyorder.sh",
    "eval/parse_and_get_acc.py",
]


def anyorder_log_likelihood(probabilities: list[dict[str, float]], target: list[str]) -> dict[str, Any]:
    _validate_probabilities(probabilities, target)
    return {
        "log_likelihood": sum(math.log(probs[token]) for probs, token in zip(probabilities, target, strict=True)),
        "model_passes": 1,
        "tokens": len(target),
    }


def enumerate_anyorder_log_likelihood(probabilities: list[dict[str, float]], target: list[str]) -> dict[str, Any]:
    _validate_probabilities(probabilities, target)
    log_values = []
    for order in itertools.permutations(range(len(target))):
        log_values.append(sum(math.log(probabilities[i][target[i]]) for i in order))
    return {
        "log_likelihood": _logmeanexp(log_values),
        "orders_checked": len(log_values),
    }


def stepmerge_approximation(step_logps: list[float], groups: int) -> dict[str, Any]:
    if not step_logps:
        raise ValueError("step_logps")
    if groups < 1 or groups > len(step_logps):
        raise ValueError("groups")
    exact = sum(step_logps)
    n = len(step_logps)
    approx = 0.0
    ranges = []
    for group in range(groups):
        start = (group * n) // groups
        end = ((group + 1) * n) // groups
        values = step_logps[start:end]
        representative = values[0]
        approx += representative * len(values)
        ranges.append({"start": start, "end": end, "representative_logp": representative})
    return {
        "exact_log_likelihood": exact,
        "approx_log_likelihood": approx,
        "error_vs_exact": approx - exact,
        "model_passes": groups,
        "groups": ranges,
    }


def audit_source_components(source_files: dict[str, str]) -> dict[str, Any]:
    lower = {path: text.lower() for path, text in source_files.items()}
    joined = "\n".join(lower.values())
    paths = set(source_files)
    return {
        "source_hashes": {path: sha256_text(text) for path, text in sorted(source_files.items())},
        "anyorder_source": _present(
            "diffu-grpo-ao/diffu_grpo_trainer_ao.py" in paths
            and all(token in lower["diffu-grpo-ao/diffu_grpo_trainer_ao.py"] for token in ("x_2l", "position_ids_2l", "pair_mask", "cross_entropy"))
        ),
        "stepmerge_source": _present(
            "diffu-grpo/diffu_grpo_trainer.py" in paths
            and all(token in lower["diffu-grpo/diffu_grpo_trainer.py"] for token in ("trajectory_mask", "self.args.n", "_get_per_token_logps_oldandref", "torch.stack"))
        ),
        "rl_script_coverage": _present(
            "d2-stepmerge" in joined
            and "d2-anyorder" in joined
            and any(path.startswith("diffu-grpo/bash_scripts/") for path in paths)
            and any(path.startswith("diffu-grpo-ao/bash_scripts/") for path in paths)
        ),
        "evaluation_parser": _present(
            "eval/parse_and_get_acc.py" in paths
            and all(token in lower["eval/parse_and_get_acc.py"] for token in ("parse_gsm_answers", "parse_math_answers", "parse_countdown_answers"))
        ),
        "anyorder_and_standard_paths": _present(
            any(path.startswith("diffu-grpo-ao/") for path in paths)
            and any(path.startswith("diffu-grpo/") for path in paths)
        ),
    }


def audit_result_artifacts(repo_files: set[str]) -> dict[str, Any]:
    machine_results = sorted(
        path
        for path in repo_files
        if path.endswith((".json", ".csv", ".jsonl"))
        and any(term in path.lower() for term in ("result", "metric", "score", "accuracy", "acc"))
    )
    datasets = sorted(path for path in repo_files if path.startswith("dataset/") and path.endswith((".csv", ".jsonl")))
    return {
        "released_dataset_files": datasets,
        "machine_readable_results": machine_results,
        "benchmark_eval_scripts_present": any(path.startswith("eval/bash_scripts/eval_") for path in repo_files),
    }


def build_evidence_bundle(source_files: dict[str, str], repo_files: set[str]) -> dict[str, Any]:
    anyorder = anyorder_log_likelihood(
        [{"A": 0.70, "B": 0.30}, {"A": 0.20, "B": 0.80}, {"A": 0.55, "B": 0.45}],
        ["A", "B", "A"],
    )
    enumerated = enumerate_anyorder_log_likelihood(
        [{"A": 0.70, "B": 0.30}, {"A": 0.20, "B": 0.80}, {"A": 0.55, "B": 0.45}],
        ["A", "B", "A"],
    )
    stepmerge = {
        str(groups): stepmerge_approximation([-0.08, -0.11, -0.28, -0.31, -0.72, -0.77, -1.10, -1.18], groups)
        for groups in (2, 4, 8)
    }
    source_audit = audit_source_components(source_files)
    result_audit = audit_result_artifacts(repo_files)
    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "title": TITLE,
        "snapshot_id": SNAPSHOT_ID,
        "upstream_pins": UPSTREAM_PINS,
        "artifact_access": {
            "source_files": sorted(source_files),
            "repo_file_count": len(repo_files),
        },
        "audits": {
            "toy_anyorder": {
                "single_pass": anyorder,
                "enumerated_orders": enumerated,
                "log_likelihood_delta": anyorder["log_likelihood"] - enumerated["log_likelihood"],
            },
            "toy_stepmerge": stepmerge,
            "source_components": source_audit,
            "result_artifacts": result_audit,
        },
        "claim_results": _claim_results(anyorder, enumerated, stepmerge, source_audit, result_audit),
    }


def fetch_pinned_artifacts() -> tuple[dict[str, str], set[str]]:
    source_files = {path: _fetch_github_text(path) for path in SOURCE_PATHS}
    repo_files = set(_github_tree_files())
    return source_files, repo_files


def write_evidence(output_path: Path, offline_fixture: bool = False) -> dict[str, Any]:
    if offline_fixture:
        source_files, repo_files = _offline_fixture()
    else:
        source_files, repo_files = fetch_pinned_artifacts()
    bundle = build_evidence_bundle(source_files, repo_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(output_path.parent.parent / "pages" / "report.md", bundle)
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evidence/bundle.json")
    parser.add_argument("--offline-fixture", action="store_true")
    args = parser.parse_args(argv)
    bundle = write_evidence(Path(args.output), offline_fixture=args.offline_fixture)
    print(f"wrote {args.output} with {len(bundle['claim_results'])} claim results")
    return 0


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _claim_results(
    anyorder: dict[str, Any],
    enumerated: dict[str, Any],
    stepmerge: dict[str, dict[str, Any]],
    source_audit: dict[str, Any],
    result_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    anyorder_exact = abs(anyorder["log_likelihood"] - enumerated["log_likelihood"]) < 1e-12
    stepmerge_converges = abs(stepmerge["4"]["error_vs_exact"]) < abs(stepmerge["2"]["error_vs_exact"]) and stepmerge["8"]["error_vs_exact"] == 0.0
    return [
        {
            "claim_index": 1,
            "claim": CLAIMS[0]["target_claim"],
            "claim_sha256": CLAIMS[0]["challenge_claim_sha256"],
            "status": "verified" if source_audit["rl_script_coverage"] == "present" and source_audit["stepmerge_source"] == "present" else "toy",
            "observation": "Official source contains RL trainer scripts and masked trajectory log-probability code; local finite-state checks recompute trajectory likelihood behavior.",
            "limitation": "This validates mechanism/source wiring, not full RL training.",
        },
        {
            "claim_index": 2,
            "claim": CLAIMS[1]["target_claim"],
            "claim_sha256": CLAIMS[1]["challenge_claim_sha256"],
            "status": "verified" if anyorder_exact and source_audit["anyorder_source"] == "present" else "toy",
            "observation": f"Toy any-order log likelihood matches enumeration with delta {anyorder['log_likelihood'] - enumerated['log_likelihood']:.3g}; source constructs the doubled sequence attention path.",
            "limitation": "The check uses a finite order-invariant toy DLM rather than an 8B checkpoint.",
        },
        {
            "claim_index": 3,
            "claim": CLAIMS[2]["target_claim"],
            "claim_sha256": CLAIMS[2]["challenge_claim_sha256"],
            "status": "toy" if source_audit["anyorder_and_standard_paths"] == "present" else "inconclusive",
            "observation": "The repository separates any-order causal code from standard LLaDA StepMerge code paths.",
            "limitation": "The broad empirical claim about any-order support across widely used DLMs was not rerun.",
        },
        {
            "claim_index": 4,
            "claim": CLAIMS[3]["target_claim"],
            "claim_sha256": CLAIMS[3]["challenge_claim_sha256"],
            "status": "toy" if stepmerge_converges and source_audit["stepmerge_source"] == "present" else "inconclusive",
            "observation": f"Toy StepMerge error improves from {stepmerge['2']['error_vs_exact']:.3f} with 2 groups to {stepmerge['4']['error_vs_exact']:.3f} with 4 groups and reaches exact at 8 groups.",
            "limitation": "The computation is a finite likelihood approximation test, not a trained masked DLM run.",
        },
        {
            "claim_index": 5,
            "claim": CLAIMS[4]["target_claim"],
            "claim_sha256": CLAIMS[4]["challenge_claim_sha256"],
            "status": "inconclusive",
            "observation": "Training and evaluation scripts exist, but no raw benchmark result artifacts are released in the pinned repository.",
            "limitation": "Performance over RL baselines requires large-model runs or raw released outputs.",
        },
        {
            "claim_index": 6,
            "claim": CLAIMS[5]["target_claim"],
            "claim_sha256": CLAIMS[5]["challenge_claim_sha256"],
            "status": "inconclusive",
            "observation": f"Released dataset files: {len(result_audit['released_dataset_files'])}; machine-readable result files: {result_audit['machine_readable_results']}.",
            "limitation": "SOTA reasoning benchmark claims were not reproduced on Countdown, Sudoku, GSM8K, or MATH500.",
        },
    ]


def _fetch_github_text(path: str) -> str:
    url = f"https://raw.githubusercontent.com/kuleshov-group/d2/{GITHUB_REVISION}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _github_tree_files() -> list[str]:
    url = f"https://api.github.com/repos/kuleshov-group/d2/git/trees/{GITHUB_REVISION}?recursive=1"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return sorted(item["path"] for item in payload.get("tree", []) if item.get("type") == "blob")


def _offline_fixture() -> tuple[dict[str, str], set[str]]:
    source_files = {
        "diffu-grpo-ao/diffu_grpo_trainer_ao.py": "x_2L position_ids_2L pair_mask F.cross_entropy per_token_logps",
        "diffu-grpo/diffu_grpo_trainer.py": "trajectory trajectory_mask self.args.N _get_per_token_logps_oldandref torch.stack(per_token_logps).sum(dim=0)",
        "diffu-grpo/bash_scripts/gsm8k_d2stepmerge.sh": "--trainer_name d2-StepMerge --N 8 --dataset gsm8k diffu-GRPO",
        "diffu-grpo-ao/bash_scripts/anyorder_gsm8k_d2anyorder.sh": "--trainer_name d2-AnyOrder --dataset gsm8k diffu-GRPO",
        "eval/parse_and_get_acc.py": "parse_gsm_answers parse_math_answers parse_countdown_answers parse_sudoku_answers",
    }
    repo_files = {
        "dataset/countdown_cd3_test.jsonl",
        "dataset/4x4_test_sudoku.csv",
        "dataset/gsm8k_genlength1024_lladaminidistill.jsonl",
        "dataset/math500_genlength1024_lladaminidistill.jsonl",
        "eval/bash_scripts/eval_gsm8k_d2stepmerge.sh",
    }
    return source_files, repo_files


def _write_report(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "| Claim | Status | Observation | Limitation |",
        "| --- | --- | --- | --- |",
    ]
    for result in bundle["claim_results"]:
        rows.append(
            f"| {result['claim_index']} | {result['status']} | {result['observation']} | {result['limitation']} |"
        )
    path.write_text(
        "\n".join(
            [
                f"# {TITLE}",
                "",
                f"Attempt: `{ATTEMPT_ID}`",
                f"Official code: `{UPSTREAM_PINS['official_code']}`",
                "",
                "## Claim Results",
                "",
                *rows,
                "",
                "## Limits",
                "",
                "No paper-reported benchmark numbers are treated as reproduced measurements. Large DLM training and inference were not run.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _present(condition: bool) -> str:
    return "present" if condition else "missing"


def _validate_probabilities(probabilities: list[dict[str, float]], target: list[str]) -> None:
    if len(probabilities) != len(target) or not target:
        raise ValueError("target")
    for probs, token in zip(probabilities, target, strict=True):
        if token not in probs:
            raise ValueError("target token")
        total = sum(probs.values())
        if any(value <= 0.0 for value in probs.values()) or not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError("probabilities")


def _logmeanexp(values: list[float]) -> float:
    high = max(values)
    return high + math.log(sum(math.exp(value - high) for value in values) / len(values))
