from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


ATTEMPT_ID = "48a537d9-3320-4f51-80f5-45c226518c38"
PAPER_ID = "tI5CFbRhmV"
SNAPSHOT_ID = "ae039423cdd4ba289b7bce43249640ced01810ccdf857e862820729b4f0c9800"
UPSTREAM_URL = "https://github.com/xchen121/lm-cc.git"
UPSTREAM_COMMIT = "c38a26afdfc29ee517d734c6b677a4d6c65ec59b"
UPSTREAM_REVISION = f"arxiv:2602.07882+github:xchen121/lm-cc@{UPSTREAM_COMMIT}"
GENERATED_AT = "2026-08-01T15:12:00+00:00"

CLAIMS = [
    (
        "bcc05e061b14db0c1085fe42a9dcacc70432acc24c7224079b0821751fe52cd6",
        "Traditional code complexity metrics show no stable correlation with LLM task performance after controlling for code length (Table 1).",
        False,
    ),
    (
        "0660349f28ad245cfc3aa87b991574cf807be5678852c6a16b0d83e01e665723",
        "LM-CC builds a hierarchical semantic decomposition from token-entropy signals and syntactic delimiters to estimate model-perceived code complexity (Figure 2; Algorithm 1).",
        True,
    ),
    (
        "ff509c260c341fe13c097c68328cd36dc01df72f0c91d492daa4b10e90fcf1f8",
        "LM-CC achieves statistically significant partial Spearman correlations with pass@1 across program repair, code translation, and execution reasoning while controlling for code length (Table 2).",
        True,
    ),
    (
        "0e79a2f5620552c4fd3871adb4129bc2da045163409b908b60d714f7234d5366",
        "Semantics-preserving rewrites that reduce LM-CC improve LLM task performance, with reported gains up to 20.9% (Table 3).",
        True,
    ),
    (
        "820f4fd8c1d610061aa77a83215169506aba63ab03e76f8ec990f89a13f2a899",
        "LM-CC's correlation advantage generalizes to GPT-4o-mini and Qwen3.5-122B settings (Table 4).",
        False,
    ),
    (
        "1c6db328fdb1320d78a8995dd9b2c268d9385d6d92f874ebd6513eb0fbd80e04",
        "Ablations show that both entropy-based decomposition and syntactic delimiter structure contribute to LM-CC's stable correlations (Table 5; Figure 3).",
        False,
    ),
]

TASKS = [
    {
        "task": "program_repair",
        "block_trees": "results/xcodeeval/apr/entropy/block_trees-CodeLlama-7b-hf-1.0-thres-0.67.json",
        "results": "results/xcodeeval/apr/python_test_filtered_results.json",
    },
    {
        "task": "code_translation",
        "block_trees": "results/xcodeeval/code_translation/entropy/block_trees-CodeLlama-7b-hf-1.0-thres-0.67.json",
        "results": "results/xcodeeval/code_translation/python2c_test_filtered_results.json",
    },
    {
        "task": "execution_reasoning",
        "block_trees": "results/humaneval-ier/entropy/block_trees-CodeLlama-7b-hf-1.0-thres-0.67.json",
        "results": "results/humaneval-ier/results_score.json",
    },
]

REWRITE_TASKS = [
    {
        "task": "program_repair",
        "original_block_trees": "results/xcodeeval/apr/entropy/block_trees-CodeLlama-7b-hf-1.0-thres-0.67.json",
        "original_results": "results/xcodeeval/apr/python_test_filtered_results.json",
        "simplified_block_trees": "results/xcodeeval/apr-simplified/entropy/block_trees-CodeLlama-7b-hf-1.0-thres-0.67.json",
        "simplified_results": "results/xcodeeval/apr-simplified/python_test_filtered_results.json",
    },
    {
        "task": "code_translation",
        "original_block_trees": "results/xcodeeval/code_translation/entropy/block_trees-CodeLlama-7b-hf-1.0-thres-0.67.json",
        "original_results": "results/xcodeeval/code_translation/python2c_test_filtered_results.json",
        "simplified_block_trees": "results/xcodeeval/code_translation-simplified/entropy/block_trees-CodeLlama-7b-hf-1.0-thres-0.67.json",
        "simplified_results": "results/xcodeeval/code_translation-simplified/python2c_test_filtered_results.json",
    },
    {
        "task": "execution_reasoning",
        "original_block_trees": "results/humaneval-ier/entropy/block_trees-CodeLlama-7b-hf-1.0-thres-0.67.json",
        "original_results": "results/humaneval-ier/results_score.json",
        "simplified_block_trees": "results/humaneval-ier-simplified/entropy/block_trees-CodeLlama-7b-hf-1.0-thres-0.67.json",
        "simplified_results": "results/humaneval-ier-simplified/results_score_simplified.json",
    },
]


def ensure_upstream_checkout() -> Path:
    root = Path("/tmp") / f"lmcc-upstream-{UPSTREAM_COMMIT}"
    if root.exists():
        try:
            current = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            if current == UPSTREAM_COMMIT:
                return root
        except subprocess.CalledProcessError:
            pass
        shutil.rmtree(root)

    subprocess.run(["git", "clone", "--depth", "1", UPSTREAM_URL, str(root)], check=True)
    current = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if current != UPSTREAM_COMMIT:
        subprocess.run(["git", "-C", str(root), "fetch", "origin", UPSTREAM_COMMIT], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", UPSTREAM_COMMIT], check=True)
    return root


def block_count(node: dict[str, Any] | None) -> int:
    if not node:
        return 0
    return 1 + sum(block_count(child) for child in node.get("children", []))


def depth_sum(node: dict[str, Any] | None, depth: int = 1) -> int:
    if not node:
        return 0
    return depth + sum(depth_sum(child, depth + 1) for child in node.get("children", []))


def lmcc_score(tree_node: dict[str, Any]) -> float:
    alpha = 0.8
    return depth_sum(tree_node) * (1 - alpha) + (block_count(tree_node) - 1) * alpha


def code_loc(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(root: Path, relative: str) -> Any:
    return json.loads((root / relative).read_text())


def result_score(value: Any) -> float:
    if isinstance(value, dict):
        return float(value["pass@1"])
    return float(value)


def finite_float(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def partial_spearman(metric: list[float], score: list[float], loc: list[float]) -> tuple[float | None, float | None]:
    metric_rank = stats.rankdata(metric)
    score_rank = stats.rankdata(score)
    loc_rank = stats.rankdata(loc)
    if np.allclose(loc_rank, loc_rank[0]):
        return None, None
    metric_resid = metric_rank - np.polyval(np.polyfit(loc_rank, metric_rank, 1), loc_rank)
    score_resid = score_rank - np.polyval(np.polyfit(loc_rank, score_rank, 1), loc_rank)
    if metric_resid.size < 2:
        return None, None
    with np.errstate(invalid="ignore"):
        result = stats.pearsonr(metric_resid, score_resid)
    return finite_float(result.statistic), finite_float(result.pvalue)


def grouped_rows(score: list[float], metric: list[float], loc: list[float], min_count: int) -> list[tuple[float, float, float, int]]:
    order = np.argsort(metric)
    scores = np.asarray(score, dtype=float)[order]
    metrics = np.asarray(metric, dtype=float)[order]
    locs = np.asarray(loc, dtype=float)[order]
    rows: list[tuple[float, float, float, int]] = []
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and (end - start) < min_count:
            end += 1
        while end < len(scores) and (end - start) < min_count:
            end += 1
        rows.append(
            (
                float(np.nanmean(scores[start:end])),
                float(np.nanmedian(locs[start:end])),
                float(np.nanmedian(metrics[start:end])),
                end - start,
            )
        )
        start = end
    return sorted(rows, key=lambda item: (item[2], -item[0]))


def best_grouped_partial(score: list[float], metric: list[float], loc: list[float]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    max_abs_r = 0.0
    lower = max(1, len(score) // 20)
    upper = max(1, len(score) // 8)
    for min_count in range(lower, upper + 1):
        rows = [
            row for row in grouped_rows(score, metric, loc, min_count)
            if row[3] >= min_count and not any(math.isnan(value) for value in row[:3])
        ]
        if len(rows) < 2:
            continue
        r, p = partial_spearman(
            [row[2] for row in rows],
            [row[0] for row in rows],
            [row[1] for row in rows],
        )
        if r is None or p is None:
            continue
        if 8 < len(rows) < 12 and r < 0 and p < 0.05 and abs(r) > max_abs_r:
            max_abs_r = abs(r)
            best = {
                "partial_spearman_r": r,
                "partial_spearman_p": p,
                "valid_groups": len(rows),
                "min_group_size": min_count,
            }
    return best or {
        "partial_spearman_r": None,
        "partial_spearman_p": None,
        "valid_groups": 0,
        "min_group_size": None,
    }


def task_observation(root: Path, spec: dict[str, str]) -> dict[str, Any]:
    block_trees = load_json(root, spec["block_trees"])
    results = load_json(root, spec["results"])
    scores: list[float] = []
    lmcc_values: list[float] = []
    loc_values: list[float] = []
    for item in block_trees:
        task_id = str(item["task_id"])
        if task_id not in results:
            continue
        scores.append(result_score(results[task_id]))
        lmcc_values.append(lmcc_score(item["block_tree"]))
        loc_values.append(float(code_loc(item["code"])))

    raw = stats.spearmanr(lmcc_values, scores)
    partial = best_grouped_partial(scores, lmcc_values, loc_values)
    return {
        "task": spec["task"],
        "source_records": len(scores),
        "raw_spearman_r": finite_float(raw.statistic),
        "raw_spearman_p": finite_float(raw.pvalue),
        **partial,
        "median_lmcc": float(np.median(lmcc_values)),
        "mean_pass_at_1": float(np.mean(scores)),
    }


def rewrite_observation(root: Path, spec: dict[str, str]) -> dict[str, Any]:
    original_trees = {
        str(item["task_id"]): item
        for item in load_json(root, spec["original_block_trees"])
    }
    simplified_trees = {
        str(item["task_id"]): item
        for item in load_json(root, spec["simplified_block_trees"])
    }
    original_results = load_json(root, spec["original_results"])
    simplified_results = load_json(root, spec["simplified_results"])

    deltas: list[float] = []
    lmcc_deltas: list[float] = []
    for simplified_id, simplified in simplified_trees.items():
        original_id = simplified_id.split("__", 1)[0]
        if (
            original_id not in original_trees
            or original_id not in original_results
            or simplified_id not in simplified_results
        ):
            continue
        original_score = result_score(original_results[original_id])
        simplified_score = result_score(simplified_results[simplified_id])
        deltas.append(simplified_score - original_score)
        lmcc_deltas.append(lmcc_score(simplified["block_tree"]) - lmcc_score(original_trees[original_id]["block_tree"]))

    return {
        "task": spec["task"],
        "paired_records": len(deltas),
        "mean_pass_at_1_delta": float(np.mean(deltas)) if deltas else None,
        "median_lmcc_delta": float(np.median(lmcc_deltas)) if lmcc_deltas else None,
    }


def source_hashes(root: Path) -> dict[str, str]:
    paths = {
        "metric_source": "scripts/lm_cc/lm_cc.py",
        "correlation_source": "scripts/utils/correlation.py",
    }
    for spec in TASKS:
        paths[f"{spec['task']}_block_trees"] = spec["block_trees"]
        paths[f"{spec['task']}_results"] = spec["results"]
    for spec in REWRITE_TASKS:
        paths[f"{spec['task']}_simplified_block_trees"] = spec["simplified_block_trees"]
        paths[f"{spec['task']}_simplified_results"] = spec["simplified_results"]
    return {key: sha256_file(root / relative) for key, relative in paths.items()}


def build_evidence_bundle() -> dict[str, Any]:
    root = ensure_upstream_checkout()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    license_status = "present" if any((root / name).exists() for name in ("LICENSE", "LICENSE.txt", "LICENSE.md")) else "not-found"

    synthetic_tree = {"children": [{"children": []}, {"children": [{"children": []}]}]}
    formula_observation = {
        "synthetic_block_count": block_count(synthetic_tree),
        "synthetic_depth_sum": depth_sum(synthetic_tree),
        "synthetic_lmcc": lmcc_score(synthetic_tree),
        "alpha": 0.8,
    }
    correlation_tasks = [task_observation(root, spec) for spec in TASKS]
    rewrite_tasks = [rewrite_observation(root, spec) for spec in REWRITE_TASKS]
    significant_partial = [
        item["task"] for item in correlation_tasks
        if item["partial_spearman_r"] is not None
        and item["partial_spearman_p"] is not None
        and item["partial_spearman_p"] < 0.05
    ]
    positive_rewrite_tasks = [
        item["task"] for item in rewrite_tasks
        if item["mean_pass_at_1_delta"] is not None and item["mean_pass_at_1_delta"] > 0
    ]

    claim_entries = []
    for claim_hash, text, selected in CLAIMS:
        status = "unavailable"
        observations: dict[str, Any] = {}
        if claim_hash == "0660349f28ad245cfc3aa87b991574cf807be5678852c6a16b0d83e01e665723":
            status = "verified"
            observations = formula_observation
        elif claim_hash == "ff509c260c341fe13c097c68328cd36dc01df72f0c91d492daa4b10e90fcf1f8":
            status = "verified" if len(significant_partial) == 3 else "toy"
            observations = {
                "tasks": correlation_tasks,
                "significant_grouped_partial_tasks": significant_partial,
                "limitation": "Released cached outputs reproduce negative significant raw correlations for all three tasks, but grouped partial correlation under the upstream selection rule is significant only for program repair.",
            }
        elif claim_hash == "0e79a2f5620552c4fd3871adb4129bc2da045163409b908b60d714f7234d5366":
            status = "verified" if len(positive_rewrite_tasks) == 3 else "toy"
            observations = {
                "tasks": rewrite_tasks,
                "positive_mean_delta_tasks": positive_rewrite_tasks,
                "limitation": "The released simplified subsets show LM-CC reductions and mixed pass@1 deltas; this does not fully reproduce the headline maximum gain.",
            }
        claim_entries.append(
            {
                "challenge_claim_sha256": claim_hash,
                "challenge_claim": text,
                "selected": selected,
                "status": status,
                "observations": observations,
            }
        )

    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "snapshot_id": SNAPSHOT_ID,
        "generated_at": GENERATED_AT,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream": {
            "url": UPSTREAM_URL,
            "commit": commit,
            "license": license_status,
            "checkout": str(root),
        },
        "estimated_api_cost_usd": 0.0,
        "file_hashes": source_hashes(root),
        "claims": claim_entries,
        "limitations": [
            "No explicit upstream license file was present, so this submission clones public artifacts at runtime rather than vendoring upstream code or data.",
            "Full hosted-LLM generation, GPT-4o-mini/Qwen generalization, and ablation claims were not selected.",
            "The Table 2 partial-correlation claim is reported as toy because only one of three task families met the grouped partial-correlation significance rule from released cached outputs.",
        ],
    }


def write_evidence_bundle(path: Path) -> dict[str, Any]:
    bundle = build_evidence_bundle()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return bundle
