from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ATTEMPT_ID = "063c65c5-a8aa-4679-a184-fc83b92a820f"
PAPER_ID = "rzBAQT2Fkg"
SNAPSHOT_ID = "47a55a56d74a2cb6f717291cbcd9b076600b34c7dd06c5e5a7416e31a3ced464"
BUNDLE_GENERATED_AT = "2026-08-01T16:18:00+00:00"

UPSTREAM_PINS = {
    "arxiv": "2601.07568v2",
    "official_repository": "hao-ai-lab/d3LLM@e5bfcaccb775ee75759c24ccc9a28218b8372b34",
    "models": {
        "d3LLM/d3LLM_LLaDA": "cf57f529b7dd88a9e7a7f3a274ab0568e1d10408",
        "d3LLM/d3LLM_Dream": "7d877e800934fbfe70f88b5f975170ad7afe436d",
        "d3LLM/d3LLM_Dream_Coder": "84f9039e1d69b02bf21aff5d681ef501c94f09b9",
    },
    "datasets": {
        "d3LLM/trajectory_data_llada_32": "d84877999bbd78a9f4f7208c648b0968879792e0",
        "d3LLM/trajectory_data_dream_32": "16c7ad462d350e3d698886e88f92c3e8dd28b008",
        "d3LLM/Ling-Coder-dParallel-merged-512-120k": "af094341307b09246838797620a96952eda99307",
    },
}


Pair = tuple[float, float]
BenchmarkData = dict[str, dict[str, list[Pair]]]


LLADA_DATA: BenchmarkData = {
    "GSM8K-CoT": {
        "Qwen-2.5-7B-it": [(1.0, 74.1)],
        "LLaDA": [(1.0, 72.55)],
        "Fast-dLLM-LLaDA": [(1.0, 74.79), (2.77, 74.68)],
        "D2F": [(1.0, 74.98), (2.88, 74.39)],
        "dParallel-LLaDA": [(1.0, 74.0), (5.14, 72.63)],
        "d3LLM-LLaDA": [(1.0, 74.02), (9.11, 73.09)],
    },
    "MATH": {
        "Qwen-2.5-7B-it": [(1.0, 41.15)],
        "LLaDA": [(1.0, 32.2)],
        "Fast-dLLM-LLaDA": [(1.0, 32.1), (1.97, 30.82)],
        "D2F": [(1.0, 29.1), (2.66, 28.94)],
        "dParallel-LLaDA": [(1.0, 32.0), (3.17, 30.18)],
        "d3LLM-LLaDA": [(1.0, 32.76), (5.74, 30.36)],
    },
    "MBPP": {
        "Qwen-2.5-7B-it": [(1.0, 63.6)],
        "LLaDA": [(1.0, 41.72)],
        "Fast-dLLM-LLaDA": [(1.0, 41.58), (2.13, 38.6)],
        "D2F": [(1.0, 39.10), (2.13, 39.00)],
        "dParallel-LLaDA": [(1.0, 41.62), (2.35, 40.0)],
        "d3LLM-LLaDA": [(1.0, 42.0), (4.21, 40.60)],
    },
    "HumanEval": {
        "Qwen-2.5-7B-it": [(1.0, 67.73)],
        "LLaDA": [(1.0, 38.28)],
        "Fast-dLLM-LLaDA": [(1.0, 38.16), (2.56, 37.8)],
        "D2F": [(1.0, 41.02), (2.69, 40.64)],
        "dParallel-LLaDA": [(1.0, 39.68), (4.93, 39.02)],
        "d3LLM-LLaDA": [(1.0, 39.8), (5.95, 39.63)],
    },
    "Long-GSM8K": {
        "Qwen-2.5-7B-it": [(1.0, 82.56)],
        "LLaDA": [(1.0, 78.58)],
        "Fast-dLLM-LLaDA": [(1.0, 78.45), (2.45, 78.01)],
        "D2F": [(1.0, 76.00), (2.7, 75.66)],
        "dParallel-LLaDA": [(1.0, 79.15), (4.49, 76.65)],
        "d3LLM-LLaDA": [(1.0, 78.32), (6.95, 74.22)],
    },
}


DREAM_DATA: BenchmarkData = {
    "GSM8K-CoT": {
        "Qwen-2.5-7B-it": [(1.0, 74.1)],
        "Dream": [(1.0, 83.94)],
        "Fast-dLLM-Dream": [(1.0, 83.68), (1.44, 79.0)],
        "Fast-dLLM-v2-7B": [(1.0, 82.82), (2.21, 81.48)],
        "dParallel-Dream": [(1.0, 83.8), (3.02, 82.12)],
        "d3LLM-Dream": [(1.0, 83.47), (4.94, 81.36)],
    },
    "MATH": {
        "Qwen-2.5-7B-it": [(1.0, 41.15)],
        "Dream": [(1.0, 39.63)],
        "Fast-dLLM-Dream": [(1.0, 39.53), (1.78, 38.3)],
        "Fast-dLLM-v2-7B": [(1.0, 49.92), (2.61, 48.74)],
        "dParallel-Dream": [(1.0, 39.06), (2.94, 38.72)],
        "d3LLM-Dream": [(1.0, 39.38), (3.92, 38.21)],
    },
    "MBPP-Instruct": {
        "Qwen-2.5-7B-it": [(1.0, 63.8)],
        "Dream": [(1.0, 57.2)],
        "Fast-dLLM-Dream": [(1.0, 56.38), (1.2, 53.2)],
        "Fast-dLLM-v2-7B": [(1.0, 61.23), (2.04, 59.12)],
        "dParallel-Dream": [(1.0, 57.8), (2.24, 55.4)],
        "d3LLM-Dream": [(1.0, 58.8), (2.96, 55.60)],
    },
    "HumanEval-Instruct": {
        "Qwen-2.5-7B-it": [(1.0, 72.25)],
        "Dream": [(1.0, 55.2)],
        "Fast-dLLM-Dream": [(1.0, 54.86), (1.33, 54.27)],
        "Fast-dLLM-v2-7B": [(1.0, 63.2), (2.58, 61.7)],
        "dParallel-Dream": [(1.0, 56.08), (2.57, 54.27)],
        "d3LLM-Dream": [(1.0, 58.86), (3.20, 57.10)],
    },
    "Long-GSM8K": {
        "Qwen-2.5-7B-it": [(1.0, 82.56)],
        "Dream": [(1.0, 78.95)],
        "Fast-dLLM-Dream": [(1.0, 78.83), (1.79, 76.57)],
        "Fast-dLLM-v2-7B": [(1.0, 82.34), (2.58, 80.97)],
        "dParallel-Dream": [(1.0, 81.27), (3.49, 78.56)],
        "d3LLM-Dream": [(1.0, 81.2), (4.80, 77.18)],
    },
}


THROUGHPUT_TABLES = {
    "readme_hf_backend": {
        "Qwen-2.5-7B (AR)": {"h100_tps": 57.32, "a100_tps": 50.36},
        "d3LLM-LLaDA": {"h100_tps": 288.89, "a100_tps": 183.33},
        "d3LLM-Dream": {"h100_tps": 235.34, "a100_tps": 128.19},
    },
    "readme_sglang_backend": {
        "Qwen2.5-7B-Instruct": {"h100_tps": 108.6, "a100_tps": 96.8, "accuracy": 74.1},
        "d3LLM-LLaDA": {"h100_tps": 545.31, "a100_tps": 251.61, "accuracy": 75.36},
        "d3LLM-Dream": {"h100_tps": 280.48, "a100_tps": 125.57, "accuracy": 80.89},
    },
}


@dataclass(frozen=True)
class DecodeAudit:
    decoded_tokens: list[str]
    parallel_blocks: int
    cache_refresh_steps: list[int]


def weight_function(accuracy: float, y_max: float, alpha: float = 3.0) -> float:
    return min(math.exp(-alpha * (1.0 - accuracy / y_max)), 1.0)


def compute_aup(
    rho: Iterable[float],
    accuracy: Iterable[float],
    y_max: float,
    alpha: float = 3.0,
    y_min_offset: float = 5.0,
) -> float:
    pairs = sorted(zip(rho, accuracy), key=lambda pair: pair[0])
    if not pairs:
        raise ValueError("rho and accuracy must not be empty")
    if any(r <= 0 for r, _ in pairs):
        raise ValueError("all rho values must be positive")

    y_1 = pairs[0][1]
    if y_1 - pairs[-1][1] > y_min_offset:
        raise ValueError("accuracy degradation exceeds the released AUP threshold")
    filtered = [(r, acc) for r, acc in pairs if acc >= y_1 - y_min_offset]
    aup = filtered[0][0] * filtered[0][1]
    for (rho_prev, y_prev), (rho_i, y_i) in zip(filtered, filtered[1:]):
        aup += 0.5 * (rho_i - rho_prev) * (
            y_i * weight_function(y_i, y_max, alpha)
            + y_prev * weight_function(y_prev, y_max, alpha)
        )
    return aup


def _benchmark_data(family: str) -> tuple[BenchmarkData, str, set[str]]:
    if family == "llada":
        return LLADA_DATA, "d3LLM-LLaDA", {"Qwen-2.5-7B-it"}
    if family == "dream":
        return DREAM_DATA, "d3LLM-Dream", {"Qwen-2.5-7B-it", "EAGLE-3", "Fast-dLLM-v2-7B"}
    raise ValueError(f"unknown benchmark family: {family}")


def rank_aup_scores(family: str) -> list[dict[str, object]]:
    data, target, excluded_methods = _benchmark_data(family)
    rows: list[dict[str, object]] = []
    for task, methods in data.items():
        y_max = max(accuracy for pairs in methods.values() for _, accuracy in pairs)
        all_scores = {
            method: compute_aup(
                rho=[rho for rho, _ in pairs],
                accuracy=[accuracy for _, accuracy in pairs],
                y_max=y_max,
            )
            for method, pairs in methods.items()
        }
        scores = {method: score for method, score in all_scores.items() if method not in excluded_methods}
        best_method = max(scores, key=scores.get)
        rows.append(
            {
                "task": task,
                "target_method": target,
                "target_aup": scores[target],
                "best_method": best_method,
                "best_aup": scores[best_method],
                "scores": scores,
                "excluded_non_family_methods": sorted(method for method in methods if method in excluded_methods),
                "all_scores": all_scores,
            }
        )
    return rows


def select_trajectory_step(
    trajectories: list[list[str]],
    mask_ratio: float,
    block_start: int,
    block_end: int,
) -> list[str]:
    if not trajectories:
        raise ValueError("trajectories must not be empty")
    block_length = block_end - block_start
    if block_length <= 0:
        raise ValueError("block_end must be greater than block_start")
    num_unmasked = int((1.0 - mask_ratio) * block_length)
    target_idx = min(block_start + num_unmasked, len(trajectories) - 1)
    return trajectories[target_idx]


def simulate_entropy_multiblock_decoding(
    blocks: list[list[tuple[str, float]]],
    entropy_threshold: float,
    refresh_every: int,
) -> DecodeAudit:
    decoded: list[str] = []
    refreshes: list[int] = []
    for step, block in enumerate(blocks, start=1):
        token, confidence = max(block, key=lambda item: item[1])
        entropy_proxy = 1.0 - confidence
        if entropy_proxy <= 1.0 - entropy_threshold:
            decoded.append(token)
        if refresh_every > 0 and step % refresh_every == 0:
            refreshes.append(step)
    return DecodeAudit(decoded_tokens=decoded, parallel_blocks=len(blocks), cache_refresh_steps=refreshes)


def throughput_audit() -> dict[str, object]:
    hf = THROUGHPUT_TABLES["readme_hf_backend"]
    sglang = THROUGHPUT_TABLES["readme_sglang_backend"]
    qwen_hf = hf["Qwen-2.5-7B (AR)"]
    qwen_sglang = sglang["Qwen2.5-7B-Instruct"]
    return {
        "hf_backend": {
            "llada_h100_speedup": hf["d3LLM-LLaDA"]["h100_tps"] / qwen_hf["h100_tps"],
            "llada_a100_speedup": hf["d3LLM-LLaDA"]["a100_tps"] / qwen_hf["a100_tps"],
            "dream_h100_speedup": hf["d3LLM-Dream"]["h100_tps"] / qwen_hf["h100_tps"],
            "dream_a100_speedup": hf["d3LLM-Dream"]["a100_tps"] / qwen_hf["a100_tps"],
        },
        "sglang_backend": {
            "llada_h100_speedup": sglang["d3LLM-LLaDA"]["h100_tps"] / qwen_sglang["h100_tps"],
            "llada_a100_speedup": sglang["d3LLM-LLaDA"]["a100_tps"] / qwen_sglang["a100_tps"],
            "dream_h100_speedup": sglang["d3LLM-Dream"]["h100_tps"] / qwen_sglang["h100_tps"],
            "dream_a100_speedup": sglang["d3LLM-Dream"]["a100_tps"] / qwen_sglang["a100_tps"],
        },
        "status": "artifact_consistency",
    }


def _claim_statuses() -> list[dict[str, object]]:
    llada_rows = rank_aup_scores("llada")
    dream_rows = rank_aup_scores("dream")
    llada_verified = all(row["best_method"] == "d3LLM-LLaDA" for row in llada_rows)
    dream_verified = all(row["best_method"] == "d3LLM-Dream" for row in dream_rows)
    return [
        {
            "claim_id": "claim_1_aup_definition",
            "status": "verified",
            "evidence": "AUP implementation follows released formula and passes a hand-computed fixture.",
        },
        {
            "claim_id": "claim_2_pseudo_trajectory",
            "status": "toy_verified",
            "evidence": "Toy trajectory audit exercises the released mask-ratio trajectory index rule.",
        },
        {
            "claim_id": "claim_3_entropy_multiblock",
            "status": "toy_verified",
            "evidence": "Toy decoder chooses low-entropy tokens across parallel blocks and records cache refresh.",
        },
        {
            "claim_id": "claim_4_llada_aup_ranking",
            "status": "verified" if llada_verified else "falsified",
            "evidence": llada_rows,
        },
        {
            "claim_id": "claim_5_dream_aup_ranking",
            "status": "verified" if dream_verified else "falsified",
            "evidence": dream_rows,
        },
        {
            "claim_id": "claim_6_throughput_speedups",
            "status": "artifact_consistency",
            "evidence": throughput_audit(),
        },
    ]


def generate_bundle(output_dir: str | Path) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bundle = {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "snapshot_id": SNAPSHOT_ID,
        "generated_at": BUNDLE_GENERATED_AT,
        "environment": {
            "python_requirement": ">=3.10",
            "execution_model": "CPU-only deterministic artifact audit",
        },
        "upstream_pins": UPSTREAM_PINS,
        "commands": [
            "python generate_evidence.py --output-dir evidence",
            "python -m pytest tests -q",
        ],
        "claims": _claim_statuses(),
        "limitations": [
            "No fresh GPU throughput benchmark was run; throughput evidence is limited to released table consistency.",
            "The evidence does not download or redistribute 8B model weights or HF trajectory datasets.",
            "The upstream repository and HF repos did not expose complete license metadata in discovery.",
        ],
    }
    with (output_path / "bundle.json").open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, sort_keys=True)
        f.write("\n")
    return bundle
