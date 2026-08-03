from __future__ import annotations

import ast
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PATH = PROJECT_ROOT / "evidence" / "bundle.json"

EXPECTED_UPSTREAM_COMMIT = "5b2c618ef146ea38890ea35dca8b07ec2d0284dd"
UPSTREAM_REPOSITORY = "https://github.com/inclusionAI/UI-Venus.git"
UPSTREAM_BRANCH = "VenusBench-Mobile"

TARGET_CLAIMS = [
    {
        "claim_id": "claim_1_benchmark_inventory",
        "challenge_claim_sha256": (
            "6ba62221083ae6c8a31d4a567742489997352c1863e00a40e302e7f361f42940"
        ),
        "claim": (
            "VenusBench-Mobile defines 149 primary tasks across 10 user-intent "
            "categories and 27 apps, plus 80 environment-variation tasks for "
            "stability testing (Figure 1, Figure 4)."
        ),
        "status": "verified",
        "provenance": (
            "README declares 27 apps and 149(+80) tasks; released JSON contains "
            "189 task records, which decomposes into the 149 primary pool plus "
            "40 instruction-variation records, while scripts define 20 stability "
            "base tasks run across dark and pad modes."
        ),
    },
    {
        "claim_id": "claim_2_pudam_taxonomy",
        "challenge_claim_sha256": (
            "22853dc5227d23efd85b2f042731212b5c9a0f7793d0adc81e29ef69885d61e8"
        ),
        "claim": (
            "The benchmark introduces the PUDAM taxonomy, scoring mobile GUI "
            "agents along Perception, Understanding, Decision, Action, and "
            "Memory dimensions (Figure 5, Table 6)."
        ),
        "status": "verified",
        "provenance": (
            "Every released task metadata record carries p/u/d/a/m ability keys, "
            "and utils/pudam_stats.py maps them to the five named dimensions."
        ),
    },
    {
        "claim_id": "claim_3_hybrid_verification",
        "challenge_claim_sha256": (
            "125402783909f19da5a2fd00c1809df109b2ac9aeb5b60be97f49b3254bcd580"
        ),
        "claim": (
            "Its evaluation infrastructure combines Android-emulator interaction "
            "with hybrid verification using MLLM-as-a-judge and programmatic "
            "checks (Figure 6)."
        ),
        "status": "verified",
        "provenance": (
            "README describes programmatic OS-state and MLLM-based verification; "
            "metadata uses both p and m evaluation methods; suite_utils calls "
            "task.is_successful while android_world/policy/verification.py "
            "implements VerifyPolicy over an LLMServer."
        ),
    },
    {
        "claim_id": "claim_4_stability_protocol",
        "challenge_claim_sha256": (
            "b453c14a66869fad6f888cf36e2a40319661ea11d75a0786ede62c3fb42300e1"
        ),
        "claim": (
            "Stability evaluation requires success across original, Chinese, "
            "dark-mode, pad, and min/max setting variants, and reports low "
            "stability pass rates for agents (Table 4)."
        ),
        "status": "partial",
        "provenance": (
            "Released README and scripts support five-mode stability evaluation "
            "as Original, Question Variation, Chinese, Mobile Dark mode, and "
            "Pad mode. The released scripts do not expose a distinct min/max "
            "setting-variant mode, and this reproduction does not rerun agents "
            "to reproduce Table 4 pass rates."
        ),
    },
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _parse_first_assigned_list(source: str, name: str) -> list[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if not any(getattr(target, "id", None) == name for target in node.targets):
                continue
            if isinstance(node.value, ast.List):
                return [
                    item.value
                    for item in node.value.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                ]
    raise ValueError(f"list {name!r} not found")


def _parse_function_returned_list(source: str, function_name: str) -> list[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != function_name:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and isinstance(child.value, ast.List):
                return [
                    item.value
                    for item in child.value.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                ]
    raise ValueError(f"return list for {function_name!r} not found")


def _parse_param_names(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if not any(getattr(target, "id", None) == "param_names" for target in node.targets):
                continue
            return ast.literal_eval(node.value)
    raise ValueError("param_names mapping not found")


def _readme_declared_counts(readme: str) -> dict[str, int]:
    app_match = re.search(r"incorporates\s+(\d+)\s+open-source Android applications", readme)
    primary_match = re.search(r"primary pool of\s+(\d+)\s+manually curated tasks", readme)
    stability_match = re.search(r"additional\s+(\d+)\s+systematic variants", readme)
    if not (app_match and primary_match and stability_match):
        raise ValueError("README count declarations not found")
    return {
        "apps": int(app_match.group(1)),
        "primary_tasks": int(primary_match.group(1)),
        "stability_variants": int(stability_match.group(1)),
    }


def _stability_modes_from_readme(readme: str) -> list[str]:
    match = re.search(
        r"five different modes:\s*([^.\n]+)",
        readme,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("README stability modes not found")
    return [re.sub(r"^and\s+", "", part.strip()) for part in match.group(1).split(",")]


def build_evidence_bundle(
    *, source_root: Path | str, command_log: list[str] | None = None
) -> dict[str, Any]:
    source_root = Path(source_root)
    commit = _expect_pinned_commit(source_root)

    readme = (source_root / "README.md").read_text(encoding="utf-8")
    metadata = _read_json(source_root / "android_world" / "task_metadata.json")
    androidworld_metadata = _read_json(
        source_root / "android_world" / "task_metadata_aw.json"
    )
    task_instance_goal = _read_json(source_root / "task_instance_goal.json")
    pudam_source = (source_root / "utils" / "pudam_stats.py").read_text(
        encoding="utf-8"
    )
    run_source = (source_root / "run_venusbenchnavi.py").read_text(encoding="utf-8")
    total_source = (source_root / "utils" / "total.py").read_text(encoding="utf-8")
    verification_source = (
        source_root / "android_world" / "policy" / "verification.py"
    ).read_text(encoding="utf-8")
    suite_source = (source_root / "android_world" / "suite_utils.py").read_text(
        encoding="utf-8"
    )

    declared = _readme_declared_counts(readme)
    param_names = _parse_param_names(pudam_source)
    stability_base = _parse_first_assigned_list(run_source, "stability_subset")
    stability_variations = _parse_first_assigned_list(
        run_source, "stability_subset_instruction_variations"
    )
    total_base = _parse_function_returned_list(total_source, "get_stability_subset")
    total_variations = _parse_function_returned_list(
        total_source, "get_stability_variations"
    )
    method_counts = Counter(item.get("evaluation_method", "") for item in metadata)
    metadata_apps = sorted({app.lower() for item in metadata for app in item.get("app", [])})
    pudam_keys = list(param_names.keys())

    hybrid_files = []
    if "programmatic OS state inspection" in readme and "MLLM-based judgment" in readme:
        hybrid_files.append("README.md")
    if "class VerifyPolicy" in verification_source and "LLMServer" in verification_source:
        hybrid_files.append("android_world/policy/verification.py")
    if "task.is_successful(env)" in suite_source:
        hybrid_files.append("android_world/suite_utils.py")

    artifacts = {
        "task_instance_goal_count": len(task_instance_goal),
        "metadata_task_count": len(metadata),
        "androidworld_baseline_task_count": len(androidworld_metadata),
        "readme_claimed_primary_tasks": declared["primary_tasks"],
        "readme_claimed_stability_variants": declared["stability_variants"],
        "readme_claimed_apps": declared["apps"],
        "metadata_app_token_count": len(metadata_apps),
        "metadata_app_tokens": metadata_apps,
        "pudam_keys": pudam_keys,
        "pudam_dimensions": [param_names[key] for key in pudam_keys],
        "evaluation_method_counts": dict(sorted(method_counts.items())),
        "hybrid_verification_files": hybrid_files,
        "stability_base_subset_count": len(stability_base),
        "stability_instruction_variation_count": len(stability_variations),
        "stability_total_execution_modes": 5,
        "stability_modes_from_readme": _stability_modes_from_readme(readme),
        "stability_lists_consistent_between_runner_and_total": (
            stability_base == total_base and stability_variations == total_variations
        ),
        "min_max_setting_variant_evidence": (
            "found"
            if re.search(r"\bmin/max\b|\bminimum\b|\bmaximum\b", run_source + total_source)
            else "not_found_in_released_scripts"
        ),
    }

    return {
        "paper_id": "coHiGZOFtS",
        "title": (
            "VenusBench-Mobile: A Challenging and User-Centric Benchmark for Mobile "
            "GUI Agents with Capability Diagnostics"
        ),
        "attempt_id": "879494d6-2a2c-429d-9d77-bba9d2eb2d70",
        "snapshot_id": "7a772df815d3e0f41377b4a3c16b7b561282b0972d4536c0ee5cd5ea08d74dea",
        "owner": "codex-paper-owner-04",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "branch": UPSTREAM_BRANCH,
            "commit": commit,
            "arxiv": "2604.06182",
            "license": "Apache-2.0",
        },
        "target_claims": TARGET_CLAIMS,
        "artifact_observations": artifacts,
        "commands": command_log or [],
        "limitations": [
            "This reproduction inspects the released benchmark artifacts and does not run Android emulator episodes.",
            "Full agent success-rate and Table 4 pass-rate values are not recomputed here.",
            "The released repository points to additional large files through Google Drive; this bundle records the available pinned code and metadata tree.",
        ],
    }


def load_evidence_bundle(path: Path | str = EVIDENCE_PATH) -> dict[str, Any]:
    return _read_json(Path(path))


def render_summary_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "# VenusBench-Mobile Reproduction Evidence",
        "",
        f"Paper: `{bundle['paper_id']}`",
        f"Attempt: `{bundle['attempt_id']}`",
        f"Snapshot: `{bundle['snapshot_id']}`",
        f"Upstream commit: `{bundle['upstream']['commit']}`",
        "",
        "## Artifact Observations",
        "",
    ]
    observations = bundle["artifact_observations"]
    for key in [
        "task_instance_goal_count",
        "metadata_task_count",
        "androidworld_baseline_task_count",
        "readme_claimed_primary_tasks",
        "readme_claimed_stability_variants",
        "readme_claimed_apps",
        "stability_base_subset_count",
        "stability_instruction_variation_count",
        "stability_total_execution_modes",
        "min_max_setting_variant_evidence",
    ]:
        lines.append(f"- `{key}`: `{observations[key]}`")

    lines.extend(["", "## Target Claims", ""])
    for claim in bundle["target_claims"]:
        lines.extend(
            [
                f"### {claim['claim_id']}: {claim['status']}",
                "",
                claim["claim"],
                "",
                f"Challenge claim SHA-256: `{claim['challenge_claim_sha256']}`",
                "",
                f"Provenance: {claim['provenance']}",
                "",
            ]
        )

    lines.extend(["## Limitations", ""])
    for limitation in bundle["limitations"]:
        lines.append(f"- {limitation}")
    return "\n".join(lines)
