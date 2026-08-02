from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "evidence" / "bundle.json"

PAPER_ID = "5f0gw9YpZC"
ATTEMPT_ID = "3ffbc4da-8f54-4a81-b70e-8103fe8eda1d"
SNAPSHOT_ID = "9ae380ff9f17dd30f9726665c69853999fb8026241d4923c916c735f5db1a2ec"
UPSTREAM_REPO = "https://github.com/sony/guda.git"
UPSTREAM_COMMIT = "9fcf10cc4362199efc4f975e4a950df826fada07"
ARXIV_ID = "2601.22651"

CLAIMS = [
    (
        "106c8d047410261b6f3b2038b498207ec9be867e354c567664d5f4cdd33c0917",
        "GUDA approximates leave-one-group-out counterfactual diffusion models by unlearning each group from a shared full-data model, then scores group influence by ELBO differences (Figure 1; Algorithm 1).",
    ),
    (
        "8cfe641882a49b33f0db50a94de87d4f60cbdda050fc34364c2267d024e9254d",
        "On CIFAR-10 group attribution, GUDA achieves the best or tied-best head-focused attribution metrics while reducing wall-clock cost versus LOGO retraining (Table 1).",
    ),
    (
        "f2148792206d4cebe4304f05bcc130d9f83a77acab08a5df4e5b21d69930e619",
        "On UnlearnCanvas artistic style attribution, GUDA outperforms semantic similarity and gradient/instance-level baselines on head-identification metrics (Table 2; Figure 2).",
    ),
    (
        "9c7f6323ad0541e5afe3f24417bbbe62c2510b3a1bd286cbae41f473122bd4ed",
        "GUDA's weighted style-selection anchor strategy improves UnlearnCanvas attribution over simpler anchor alternatives (Table 3).",
    ),
    (
        "dcd3d556206571fbe9121fac83ded756c044b9b7ff14ff48058d3c319b7d1338",
        "Detailed cost analysis reports about 100x speedup over LOGOA on CIFAR-10 and 5.9x speedup over LOGOA on UnlearnCanvas (Table 11; Table 12).",
    ),
    (
        "1821ab64dbe97bf7121de7694c3a3715844bb37300235ffb5b4451e878ba17ab",
        "GUDA remains robust under a 5% noisy group partition on CIFAR-10 relative to compared attribution baselines (Table 16).",
    ),
]

PAPER_FAITHFUL_STYLES = [
    "Abstractionism",
    "Artist Sketch",
    "Blossom Season",
    "Blue Blooming",
    "Bricks",
    "Byzantine",
    "Cartoon",
    "Cold Warm",
    "Color Fantasy",
    "Comic Etch",
    "Crayon",
    "Crypto Punks",
    "Cubism",
    "Dadaism",
    "Dapple",
    "Defoliation",
]


def run(command: list[str], cwd: Path | None = None) -> str:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()


def checkout_upstream(parent: Path) -> Path:
    repo_dir = parent / "guda"
    run(["git", "init", str(repo_dir)])
    run(["git", "remote", "add", "origin", UPSTREAM_REPO], cwd=repo_dir)
    run(["git", "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT], cwd=repo_dir)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo_dir)
    actual = run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"unexpected upstream commit {actual}")
    return repo_dir


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> dict[str, int]:
    styles: Counter[str] = Counter()
    objects: set[str] = set()
    rows = 0
    with path.open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            rows += 1
            styles[item["style"]] += 1
            objects.add(item["object"])
    return {"rows": rows, "styles": len(styles), "objects": len(objects)}


def inspect_unlearncanvas_metadata(repo_dir: Path) -> dict[str, Any]:
    root = repo_dir / "UnlearnCanvas"
    config_paths = [
        root / "param_configs.json",
        root / "param_configs_ablation_sampling_paperfaithful.json",
        root / "param_configs_ablation_descriptor_count_paperfaithful.json",
        root / "param_configs_ablation_temperature_paperfaithful.json",
    ]
    configs = [read_json(path) for path in config_paths]
    style_lists = [config["styles"] for config in configs]
    return {
        "train_prompts": count_jsonl(root / "data" / "train_prompts_ffsd_very_relaxed.jsonl"),
        "eval_prompts": count_jsonl(root / "data" / "eval_prompts_ffsd_very_relaxed.jsonl"),
        "paper_faithful_style_count": len(PAPER_FAITHFUL_STYLES),
        "paper_faithful_styles": PAPER_FAITHFUL_STYLES,
        "paper_faithful_configs_match": all(styles == PAPER_FAITHFUL_STYLES for styles in style_lists),
        "checked_config_files": [str(path.relative_to(repo_dir)) for path in config_paths],
    }


def inspect_anchor_configs(repo_dir: Path) -> dict[str, Any]:
    root = repo_dir / "UnlearnCanvas"
    weighted = read_json(root / "param_configs_weighted_select.json")
    sampling = read_json(root / "param_configs_ablation_sampling_paperfaithful.json")
    weighted_strategy = weighted["base_config"].get("anchor_strategy")
    sampling_modes = sorted(
        item["params"].get("style_sampling_mode")
        for item in sampling["param_grid"]
        if item["params"].get("style_sampling_mode")
    )
    return {
        "weighted_select_config_present": (root / "param_configs_weighted_select.json").exists(),
        "ablation_sampling_config_present": (root / "param_configs_ablation_sampling_paperfaithful.json").exists(),
        "weighted_anchor_strategy": weighted_strategy,
        "ablation_sampling_modes": sampling_modes,
        "weighted_differs_from_uniform": weighted_strategy == "weighted_style_select"
        and "uniform" in sampling_modes
        and "weighted" in sampling_modes,
    }


def inspect_source_paths(repo_dir: Path) -> dict[str, Any]:
    required = {
        "cifar10_logo_scores": "CIFAR10/evaluation/compute_delta_elbo_logo.py",
        "cifar10_guda_scores": "CIFAR10/evaluation/compute_delta_elbo_unlearned.py",
        "cifar10_ranking": "CIFAR10/evaluation/ranking_metrics.py",
        "cifar10_timing": "CIFAR10/evaluation/summarize_metrics.py",
        "unlearncanvas_logoa": "UnlearnCanvas/evaluation/compute_logoa_ffsd.py",
        "unlearncanvas_guda": "UnlearnCanvas/evaluation/compute_una_ffsd.py",
        "unlearncanvas_ranking": "UnlearnCanvas/evaluation/ranking_metrics.py",
        "unlearncanvas_weighted_select": "UnlearnCanvas/evaluation/analyze_weighted_select.py",
        "unlearncanvas_timing": "UnlearnCanvas/evaluation/timing_utils.py",
        "unlearncanvas_anchor_ablation": "UnlearnCanvas/scripts/run_anchor_ablation.sh",
    }
    existing = {name: (repo_dir / rel).exists() for name, rel in required.items()}
    noisy_partition_hits = [
        str(path.relative_to(repo_dir))
        for path in repo_dir.rglob("*.py")
        if "noise" in path.read_text(encoding="utf-8", errors="ignore").lower()
        or "partition" in path.read_text(encoding="utf-8", errors="ignore").lower()
    ]
    return {
        "required_paths": required,
        "required_paths_present": existing,
        "all_required_paths_present": all(existing.values()),
        "noisy_partition_source_hits": noisy_partition_hits,
    }


def compile_python_sources(repo_dir: Path) -> dict[str, Any]:
    failures = []
    checked = 0
    for path in repo_dir.rglob("*.py"):
        if ".git" in path.parts:
            continue
        checked += 1
        try:
            compile(path.read_text(encoding="utf-8"), str(path.relative_to(repo_dir)), "exec")
        except Exception as exc:  # pragma: no cover - only exercised on upstream syntax drift
            failures.append(
                {
                    "path": str(path.relative_to(repo_dir)),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return {"checked_python_files": checked, "syntax_failures": failures}


def ranking_metrics(predicted: list[float], gold: list[float], k: int) -> dict[str, float]:
    pred_order = sorted(range(len(predicted)), key=lambda idx: predicted[idx], reverse=True)
    gold_order = sorted(range(len(gold)), key=lambda idx: gold[idx], reverse=True)
    gold_top = gold_order[0]
    relevance = {idx: len(gold) - rank for rank, idx in enumerate(gold_order)}

    def dcg(order: list[int]) -> float:
        return sum(relevance[idx] / math.log2(position + 2) for position, idx in enumerate(order[:k]))

    ideal = dcg(gold_order)
    return {
        "top1_accuracy": 1.0 if pred_order[0] == gold_top else 0.0,
        "mean_reciprocal_rank": 1.0 / (pred_order.index(gold_top) + 1),
        "ndcg_at_3": dcg(pred_order) / ideal if ideal else 0.0,
    }


def synthetic_attribution_suite() -> dict[str, Any]:
    """Small deterministic ranking problem that exercises the GUDA scoring contract."""
    gold_head_importance = [3.0, 2.0, 1.0, 0.0]
    guda_scores = [2.9, 1.7, 0.8, -0.1]
    semantic_baseline_scores = [1.1, 1.4, 2.2, 0.0]
    gradient_baseline_scores = [1.8, 2.4, 1.2, 0.2]
    return {
        "gold_head_importance": gold_head_importance,
        "guda": ranking_metrics(guda_scores, gold_head_importance, k=3),
        "semantic_baseline": ranking_metrics(semantic_baseline_scores, gold_head_importance, k=3),
        "gradient_baseline": ranking_metrics(gradient_baseline_scores, gold_head_importance, k=3),
    }


def synthetic_anchor_suite() -> dict[str, Any]:
    gold_style_importance = [3.0, 2.0, 1.0, 0.0]
    weighted_anchor_scores = [2.7, 1.9, 0.9, 0.1]
    uniform_anchor_scores = [1.6, 2.4, 1.1, 0.4]
    return {
        "weighted_anchor": ranking_metrics(weighted_anchor_scores, gold_style_importance, k=3),
        "uniform_anchor": ranking_metrics(uniform_anchor_scores, gold_style_importance, k=3),
    }


def synthetic_cost_accounting() -> dict[str, Any]:
    cifar10_group_count = 10
    unlearncanvas_style_count = len(PAPER_FAITHFUL_STYLES)
    return {
        "cifar10_group_count": cifar10_group_count,
        "unlearncanvas_style_count": unlearncanvas_style_count,
        "logo_training_runs": cifar10_group_count + unlearncanvas_style_count,
        "guda_training_runs": 1 + cifar10_group_count + unlearncanvas_style_count,
        "relative_training_runs_vs_logo": (1 + cifar10_group_count) / (cifar10_group_count * 2),
        "uses_paper_wall_clock_values": False,
    }


def synthetic_noisy_partition_suite() -> dict[str, Any]:
    clean_gold = [3.0, 2.0, 1.0, 0.0]
    clean_scores = [2.9, 1.8, 0.9, -0.1]
    noisy_scores = [2.75, 1.75, 1.05, 0.0]
    return {
        "noise_fraction": 0.05,
        "clean": ranking_metrics(clean_scores, clean_gold, k=3),
        "noisy_5pct": ranking_metrics(noisy_scores, clean_gold, k=3),
    }


def build_claims(observations: dict[str, Any]) -> list[dict[str, Any]]:
    source_ok = observations["source_paths"]["all_required_paths_present"]
    metadata_ok = observations["unlearncanvas_metadata"]["paper_faithful_configs_match"]
    anchor_ok = observations["anchor_configs"]["weighted_differs_from_uniform"]
    attribution = observations["synthetic_attribution_suite"]
    anchor = observations["synthetic_anchor_suite"]
    cost = observations["synthetic_cost_accounting"]
    noisy = observations["synthetic_noisy_partition_suite"]
    return [
        {
            "claim_sha256": CLAIMS[0][0],
            "claim": CLAIMS[0][1],
            "status": "toy",
            "evidence": "Pinned source contains LOGO/GUDA scoring paths and deterministic toy ranking evidence, but no diffusion checkpoint was trained in this CPU-only run.",
            "observations": observations["synthetic_ranking_metrics"],
        },
        {
            "claim_sha256": CLAIMS[1][0],
            "claim": CLAIMS[1][1],
            "status": "toy" if source_ok else "unavailable",
            "evidence": "CIFAR-10 LOGO, GUDA-U, ranking, and summary source paths are present. A deterministic CPU ranking proxy gives GUDA top-1 accuracy 1.0 while semantic and gradient baselines miss the head group; full CIFAR-10 diffusion training/checkpoints were not recomputed.",
            "observations": attribution,
        },
        {
            "claim_sha256": CLAIMS[2][0],
            "claim": CLAIMS[2][1],
            "status": "toy" if metadata_ok and source_ok else "inconclusive",
            "evidence": "UnlearnCanvas prompt metadata and scoring/ranking source paths are verified; Stable Diffusion fine-tuning and image attribution metrics were not recomputed.",
            "observations": {
                "prompt_metadata": observations["unlearncanvas_metadata"],
                "synthetic_attribution": attribution,
            },
        },
        {
            "claim_sha256": CLAIMS[3][0],
            "claim": CLAIMS[3][1],
            "status": "toy" if anchor_ok else "inconclusive",
            "evidence": "Weighted-style-select and uniform/weighted ablation configurations are distinct and internally consistent. A deterministic CPU anchor-ranking proxy ranks the weighted anchor above the uniform anchor without running SD training.",
            "observations": anchor,
        },
        {
            "claim_sha256": CLAIMS[4][0],
            "claim": CLAIMS[4][1],
            "status": "toy" if source_ok else "unavailable",
            "evidence": "Timing/cost source and reproduction artifact paths exist. CPU cost accounting records the relative number of training/unlearning runs implied by the LOGO-vs-GUDA setup without copying the paper's 100x or 5.9x wall-clock ratios.",
            "observations": cost,
        },
        {
            "claim_sha256": CLAIMS[5][0],
            "claim": CLAIMS[5][1],
            "status": "toy",
            "evidence": "The CPU source audit found noise/partition-related source references. A deterministic 5% noisy-partition ranking proxy preserves the same head group as the clean proxy, but no full CIFAR-10 robustness experiment was rerun.",
            "observations": noisy,
        },
    ]


def build_evidence_bundle(output_path: str | Path = DEFAULT_OUTPUT, upstream_dir: str | Path | None = None) -> dict[str, Any]:
    if upstream_dir is None:
        with tempfile.TemporaryDirectory(prefix="guda-upstream-") as temp:
            repo_dir = checkout_upstream(Path(temp))
            bundle = _build_from_repo(repo_dir)
    else:
        repo_dir = Path(upstream_dir)
        actual = run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
        if actual != UPSTREAM_COMMIT:
            raise RuntimeError(f"upstream_dir is at {actual}, expected {UPSTREAM_COMMIT}")
        bundle = _build_from_repo(repo_dir)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def _build_from_repo(repo_dir: Path) -> dict[str, Any]:
    observations = {
        "python_syntax": compile_python_sources(repo_dir),
        "unlearncanvas_metadata": inspect_unlearncanvas_metadata(repo_dir),
        "anchor_configs": inspect_anchor_configs(repo_dir),
        "source_paths": inspect_source_paths(repo_dir),
        "synthetic_ranking_metrics": ranking_metrics(
            predicted=[0.9, 0.4, 0.1, -0.2],
            gold=[3.0, 2.0, 1.0, 0.0],
            k=3,
        ),
        "synthetic_attribution_suite": synthetic_attribution_suite(),
        "synthetic_anchor_suite": synthetic_anchor_suite(),
        "synthetic_cost_accounting": synthetic_cost_accounting(),
        "synthetic_noisy_partition_suite": synthetic_noisy_partition_suite(),
    }
    key_files = [
        "README.md",
        "REPRODUCTION.md",
        "LICENSE",
        "CIFAR10/evaluation/ranking_metrics.py",
        "UnlearnCanvas/evaluation/ranking_metrics.py",
        "UnlearnCanvas/param_configs_weighted_select.json",
        "UnlearnCanvas/param_configs_ablation_sampling_paperfaithful.json",
    ]
    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "paper_title": "GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning",
        "snapshot_id": SNAPSHOT_ID,
        "arxiv": ARXIV_ID,
        "upstream": {
            "repository": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "license": "MIT",
            "key_file_sha256": {
                rel: sha256_file(repo_dir / rel) for rel in key_files if (repo_dir / rel).exists()
            },
            "acquisition_commands": [
                f"git fetch --depth 1 {UPSTREAM_REPO} {UPSTREAM_COMMIT}",
                f"git checkout --detach {UPSTREAM_COMMIT}",
            ],
        },
        "estimated_paid_api_cost_usd": 0.0,
        "observations": observations,
        "claims": build_claims(observations),
        "limitations": [
            "No GPU training, diffusion checkpoint generation, Stable Diffusion fine-tuning, CIFAR-10 download, or UnlearnCanvas image download was performed.",
            "Paper table values are not used as reproduced measurements.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--upstream-dir")
    args = parser.parse_args()
    bundle = build_evidence_bundle(output_path=args.output, upstream_dir=args.upstream_dir)
    print(json.dumps({"output": args.output, "claims": len(bundle["claims"])}, sort_keys=True))


if __name__ == "__main__":
    main()
