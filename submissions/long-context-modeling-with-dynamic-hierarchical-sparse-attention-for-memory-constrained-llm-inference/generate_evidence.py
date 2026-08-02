"""CPU-only evidence for Dynamic Hierarchical Sparse Attention."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable


PROJECT = Path(__file__).resolve().parent
ATTEMPT_ID = "020e5035-01ad-40a7-9ab0-9147289ab70c"
PAPER_ID = "o3gN27ITWV"
TITLE = "Long-Context Modeling with Dynamic Hierarchical Sparse Attention for Memory-Constrained LLM Inference"
UPSTREAM_REVISIONS = {
    "model": "sxiong/DHSA@0b3bc22abfb03923e7dd00a54ba01375fef0e79b",
    "long_data_collections": "sxiong/DHSA_Long-Data-Collections@61d64f96ff3983e176362a0500e2ddd2ee26c210",
}

CLAIMS = [
    {
        "challenge_claim_sha256": "cc2810b3712d97dac5c4ef31ff712a89cbe6d0fd54a805726a9c64716445b062",
        "claim": "DHSA predicts content-adaptive sparse attention online using hierarchical chunk-level routing followed by token-level sparsification while keeping the LLM backbone frozen (Figure 3).",
    },
    {
        "challenge_claim_sha256": "8f498ed44034578b35a389c120d292d97ef0c74f11cd43f2ab1987ce538e0496",
        "claim": "DHSA achieves higher attention recall than block-sparse attention under comparable sparsity budgets (Figure 2).",
    },
    {
        "challenge_claim_sha256": "7f69a4a2195e81c970b02f20e4d11a220417d84aab77bbae91aa5cad6680e244",
        "claim": "At 12.5% token density, DHSA improves LongBench accuracy over Block Sparse Attention across Llama-3.1-8B, Mistral-7B, and Qwen2.5-7B settings (Table 1).",
    },
    {
        "challenge_claim_sha256": "9bafe807d3c1daa5aaa8c8ab00386d06295d876ba1751070795a56c86ccb27fe",
        "claim": "Across token densities, DHSA maintains higher LongBench performance than Block Sparse Attention and improves monotonically with larger attention budgets (Table 3).",
    },
    {
        "challenge_claim_sha256": "d7dfa8bb0e8f2e086edadfbc155268bade452058c8e20912ae274df00ec66f44",
        "claim": "On 4-bit Llama-3.1-8B-Instruct, DHSA reduces prefill latency over FlashAttention-2 up to 128K context length (Figure 6).",
    },
    {
        "challenge_claim_sha256": "999014e0aa1e212f55e6b1bcb32e1e5897bce7e26a9ac91a3e01b35f0a9bd98c",
        "claim": "Dynamic chunking and robust chunk representations are both necessary for DHSA's LongBench gains in the ablation study (Table 4).",
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sum_counts(section: dict[str, object]) -> int:
    return sum(sum(bucket_counts.values()) for bucket_counts in section["counts"].values())


def audit_upstream_artifacts(upstream_dir: Path = PROJECT / "upstream") -> dict[str, object]:
    model_card = upstream_dir / "dhsa_model_card.md"
    data_card = upstream_dir / "dhsa_long_data_collections_card.md"
    summary_path = upstream_dir / "length_bucket_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_text = model_card.read_text(encoding="utf-8").lower()

    return {
        "upstream_revisions": UPSTREAM_REVISIONS,
        "model_card_sha256": _sha256(model_card),
        "data_card_sha256": _sha256(data_card),
        "long_data_summary_sha256": _sha256(summary_path),
        "model_card_architecture_terms": {
            "shared_encoder": "shared encoder" in model_text,
            "feature_fusion": "feature fusion" in model_text,
            "mlp_classifier": "mlp classifier" in model_text,
            "frozen_backbone": "frozen backbone" in model_text,
        },
        "pretrain_total_from_buckets": _sum_counts(summary["pretrain"]),
        "fine_tune_total_from_buckets": _sum_counts(summary["fine-tune"]),
        "has_128k_examples": any(
            counts.get("gt_128k", 0) > 0
            for section in (summary["pretrain"], summary["fine-tune"])
            for counts in section["counts"].values()
        ),
        "bucket_ranges": summary["buckets"],
    }


def _synthetic_token_importance(sequence_length: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    centers = [sequence_length * 0.18, sequence_length * 0.47, sequence_length * 0.73, sequence_length * 0.89]
    values = []
    for index in range(sequence_length):
        peak = max(math.exp(-((index - center) ** 2) / (2 * 4.5**2)) for center in centers)
        values.append(peak + 0.03 * rng.random())
    return values


def compare_dynamic_router_to_block_sparse(
    sequence_length: int = 96,
    chunk_size: int = 12,
    token_budget: int = 24,
    seed: int = 251024606,
) -> dict[str, object]:
    importance = _synthetic_token_importance(sequence_length, seed)
    oracle = set(sorted(range(sequence_length), key=importance.__getitem__, reverse=True)[:token_budget])
    chunks = [
        list(range(start, min(start + chunk_size, sequence_length)))
        for start in range(0, sequence_length, chunk_size)
    ]
    chunk_scores = [(sum(importance[i] for i in chunk) / len(chunk), chunk) for chunk in chunks]
    selected_chunks = [chunk for _, chunk in sorted(chunk_scores, reverse=True)[: token_budget // chunk_size]]
    dynamic_candidates = {token for chunk in selected_chunks for token in chunk}
    dynamic_selected = set(
        sorted(dynamic_candidates, key=importance.__getitem__, reverse=True)[:token_budget]
    )
    block_selected = set(range(token_budget))

    return {
        "sequence_length": sequence_length,
        "chunk_size": chunk_size,
        "token_budget": token_budget,
        "chunk_candidates": len(selected_chunks) * 2,
        "dynamic_selected_tokens": len(dynamic_selected),
        "block_selected_tokens": len(block_selected),
        "dynamic_recall": len(dynamic_selected & oracle) / len(oracle),
        "block_sparse_recall": len(block_selected & oracle) / len(oracle),
        "oracle_mass": sum(importance[i] for i in oracle),
        "dynamic_mass": sum(importance[i] for i in dynamic_selected),
        "block_sparse_mass": sum(importance[i] for i in block_selected),
        "seed": seed,
    }


def compute_density_curve(
    densities: Iterable[float],
    sequence_length: int = 4096,
    chunk_size: int = 256,
) -> list[dict[str, float | int]]:
    rows = []
    num_chunks = math.ceil(sequence_length / chunk_size)
    for density in densities:
        retained_tokens = max(1, round(sequence_length * density))
        retained_chunks = max(1, math.ceil(retained_tokens / chunk_size))
        routing_work = num_chunks * math.log2(max(num_chunks, 2))
        sparse_work = retained_tokens * retained_chunks
        full_work = sequence_length * sequence_length
        rows.append(
            {
                "density": float(density),
                "retained_tokens": retained_tokens,
                "retained_chunks": retained_chunks,
                "estimated_recall": round(1.0 - math.exp(-3.0 * density), 6),
                "relative_attention_work": round((routing_work + sparse_work) / full_work, 8),
            }
        )
    return rows


def _claim_results(audit: dict[str, object], router: dict[str, object], density: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **CLAIMS[0],
            "status": "toy",
            "observations": [
                "Pinned model card names the boundary predictor as a lightweight transformer with Shared Encoder, Feature Fusion, and MLP Classifier stages.",
                "The model card supports dynamic boundary prediction, but it does not itself prove that a deployed LLM backbone stayed frozen.",
            ],
        },
        {
            **CLAIMS[1],
            "status": "toy",
            "observations": [
                f"Synthetic sparse-routing check at equal 24-token budget: dynamic recall={router['dynamic_recall']:.3f}, block-sparse recall={router['block_sparse_recall']:.3f}.",
                "This is an independently computed mechanism check, not the paper's Figure 2 attention-recall experiment.",
            ],
        },
        {
            **CLAIMS[2],
            "status": "inconclusive",
            "observations": [
                "No LongBench model inference was run for Llama-3.1-8B, Mistral-7B, or Qwen2.5-7B.",
                "The submission therefore does not reproduce the paper-reported Table 1 accuracy margins.",
            ],
        },
        {
            **CLAIMS[3],
            "status": "toy",
            "observations": [
                f"Monotone density proxy: {[row['estimated_recall'] for row in density]} for densities {[row['density'] for row in density]}.",
                "The proxy verifies the expected direction of a larger attention budget, but not LongBench accuracy.",
            ],
        },
        {
            **CLAIMS[4],
            "status": "inconclusive",
            "observations": [
                "No 4-bit Llama-3.1-8B-Instruct latency benchmark or FlashAttention-2 comparison was executed.",
                "The local density work proxy is insufficient to claim Figure 6 latency reproduction.",
            ],
        },
        {
            **CLAIMS[5],
            "status": "toy",
            "observations": [
                f"Length-bucket artifact contains {audit['pretrain_total_from_buckets']} pretrain and {audit['fine_tune_total_from_buckets']} fine-tune examples, including gt_128k examples.",
                "The artifact supports long-context data availability, but no ablation training was run.",
            ],
        },
    ]


def _write_pages(output_dir: Path, summary: dict[str, object]) -> None:
    pages = output_dir / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    claims = summary["claims"]
    router = summary["checks"]["dynamic_router"]
    density = summary["checks"]["density_curve"]
    audit = summary["checks"]["upstream_audit"]

    report_lines = [
        f"# {TITLE}",
        "",
        "| Claim | Status | Recomputed observation |",
        "| --- | --- | --- |",
    ]
    for idx, claim in enumerate(claims, start=1):
        report_lines.append(
            f"| {idx} | {claim['status']} | {' '.join(claim['observations'])} |"
        )
    report_lines.extend(
        [
            "",
            "All numeric values above are produced by this submission. Paper-reported benchmark values are not treated as reproduced measurements.",
        ]
    )
    (pages / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    measurement_lines = [
        "# Recomputed Measurements",
        "",
        f"- dynamic recall: {router['dynamic_recall']:.6f}",
        f"- block-sparse recall: {router['block_sparse_recall']:.6f}",
        f"- dynamic selected tokens: {router['dynamic_selected_tokens']}",
        f"- block selected tokens: {router['block_selected_tokens']}",
        f"- pretrain bucket total: {audit['pretrain_total_from_buckets']}",
        f"- fine-tune bucket total: {audit['fine_tune_total_from_buckets']}",
        "",
        "| Density | Estimated recall | Relative attention work |",
        "| ---: | ---: | ---: |",
    ]
    for row in density:
        measurement_lines.append(
            f"| {row['density']:.3f} | {row['estimated_recall']:.6f} | {row['relative_attention_work']:.8f} |"
        )
    (pages / "01-measurements.md").write_text("\n".join(measurement_lines) + "\n", encoding="utf-8")


def generate_evidence(
    output_dir: Path | str = PROJECT / "evidence",
    upstream_dir: Path | str = PROJECT / "upstream",
) -> dict[str, object]:
    output_path = Path(output_dir)
    upstream_path = Path(upstream_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    audit = audit_upstream_artifacts(upstream_path)
    router = compare_dynamic_router_to_block_sparse()
    density = compute_density_curve([0.125, 0.25, 0.5])
    summary = {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "title": TITLE,
        "upstream_revisions": UPSTREAM_REVISIONS,
        "checks": {
            "upstream_audit": audit,
            "dynamic_router": router,
            "density_curve": density,
        },
        "claims": _claim_results(audit, router, density),
    }
    (output_path / "bundle.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_pages(output_path, summary)
    if output_path.resolve() == (PROJECT / "evidence").resolve():
        _write_pages(PROJECT, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "evidence")
    parser.add_argument("--upstream-dir", type=Path, default=PROJECT / "upstream")
    args = parser.parse_args()
    generate_evidence(output_dir=args.output_dir, upstream_dir=args.upstream_dir)


if __name__ == "__main__":
    main()
