from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Iterable


ATTEMPT_ID = "8a83f44b-e3db-4c2b-acf7-d233a750fdcc"
PAPER_ID = "4P9cEcinYP"
SNAPSHOT_ID = "34237d5702ab85038fbe25e4409a2115b90ef0257ab437d9511f2d66ded5fdd5"
TITLE = "Efficient, Property-Aligned Fan-Out Retrieval via RL-Amortized Diffusion"
ARXIV_ID = "2603.06397"
EPRINT_SHA256 = "3602626196bb2747970029de2a6f9b8086e4450a8f67c105d2954911a1d8a568"
PDF_SHA256 = "19b97f8264d22e28dbdced297d16d1686c68de2bbc0c756a2f9330fc023490bb"

UPSTREAM_PINS = {
    "google_publication": "https://research.google/pubs/efficient-property-aligned-fan-out-retrieval-via-rl-compiled-diffusion-2/",
    "arxiv_eprint": f"arxiv:{ARXIV_ID}@sha256:{EPRINT_SHA256}",
    "arxiv_pdf": f"arxiv:{ARXIV_ID}@sha256:{PDF_SHA256}",
}

CLAIM_TEXTS = [
    "R4T trains a fan-out language model with set-level rewards, synthesizes objective-consistent query-set supervision, and trains a diffusion retriever for single-pass fan-out in embedding space (Figure 1).",
    "R4T consistently outperforms fan-out baselines across Open-Ended Abstract Retrieval datasets and metrics (Table 1).",
    "R4T improves weakly supervised compositional retrieval on Polyvore relative to no-fan-out, zero-shot fan-out, and best-of-N baselines (Table 2).",
    "Jointly optimizing groundedness, alignment, and diversity prevents reward-collapse behavior during fan-out LM training (Figure 4).",
    "The diffusion fan-out retriever maintains sub-second latency for small batches and reaches order-of-magnitude speedups over autoregressive LLM fan-out at larger batch sizes (Figure 5).",
]

CLAIMS = [
    {
        "index": index,
        "target_claim": text,
        "challenge_claim_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    for index, text in enumerate(CLAIM_TEXTS, start=1)
]


def audit_source_manifest(files: Iterable[str]) -> dict[str, Any]:
    file_set = set(files)
    return {
        "file_count": len(file_set),
        "has_main_tex": "main.tex" in file_set,
        "has_table1_tex": "task_1_result.tex" in file_set,
        "has_table2_tex": "task_2_result.tex" in file_set,
        "has_latency_figure": "query_fanout_efficiency.pdf" in file_set,
        "has_reward_figures": any(name.startswith("bq_reward_") for name in file_set),
        "python_files": sorted(name for name in file_set if name.endswith(".py")),
        "dataset_files": sorted(name for name in file_set if name.endswith((".jsonl", ".parquet", ".csv"))),
        "model_files": sorted(name for name in file_set if name.endswith((".pt", ".pth", ".safetensors", ".ckpt"))),
    }


def audit_tables(task1_tex: str, task2_tex: str) -> dict[str, Any]:
    return {
        "task1_r4t_mentions": task1_tex.count("R4T-"),
        "task2_r4t_mentions": task2_tex.count("R4T-"),
        "task1_has_best_of_n": "Best-of-N" in task1_tex,
        "task2_has_polyvore_metrics": all(metric in task2_tex for metric in ("Recall@5K", "Hit@5K", "VS")),
        "table_values_are_recomputed": False,
    }


def toy_set_reward(query: list[float], outputs: list[list[float]], database: list[list[float]]) -> dict[str, float]:
    if not outputs:
        return {"groundedness": 0.0, "alignment": 0.0, "diversity": 0.0, "total": 0.0}
    groundedness = sum(max(_cosine(output, item) for item in database) for output in outputs) / len(outputs)
    alignment = sum(_cosine(query, output) for output in outputs) / len(outputs)
    pairwise = [
        1.0 - _cosine(outputs[i], outputs[j])
        for i in range(len(outputs))
        for j in range(i + 1, len(outputs))
    ]
    diversity = sum(pairwise) / len(pairwise) if pairwise else 0.0
    total = (0.45 * groundedness) + (0.25 * alignment) + (0.30 * diversity)
    return {
        "groundedness": round(groundedness, 6),
        "alignment": round(alignment, 6),
        "diversity": round(diversity, 6),
        "total": round(total, 6),
    }


def compile_synthetic_supervision(samples: dict[str, list[list[float]]]) -> list[dict[str, Any]]:
    return [
        {
            "query": query,
            "target_count": len(targets),
            "target_centroid": [
                round(sum(target[index] for target in targets) / len(targets), 6)
                for index in range(len(targets[0]))
            ]
            if targets
            else [],
        }
        for query, targets in sorted(samples.items())
    ]


def build_evidence_bundle(source_files: dict[str, str] | None = None) -> dict[str, Any]:
    source_files = source_files if source_files is not None else fetch_arxiv_source()
    manifest = audit_source_manifest(source_files)
    tables = audit_tables(
        source_files.get("task_1_result.tex", ""),
        source_files.get("task_2_result.tex", ""),
    )
    method_text = source_files.get("main.tex", "")
    toy = {
        "collapsed": toy_set_reward(
            [1.0, 0.0],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        ),
        "diverse": toy_set_reward(
            [1.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        ),
        "supervision": compile_synthetic_supervision(
            {"bohemian festival style": [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]}
        ),
    }

    statuses = [
        {
            "status": "toy",
            "evidence": (
                "The pinned arXiv source describes the three R4T stages and this package exercises "
                "a tiny set-level reward plus synthetic target compilation. No released FOLM or "
                "diffusion training code is available."
            ),
        },
        {
            "status": "inconclusive",
            "evidence": (
                "task_1_result.tex contains paper-reported OAR table rows, but no raw Polyvore/Music "
                "inputs, model outputs, or evaluation scripts are released for recomputation."
            ),
        },
        {
            "status": "inconclusive",
            "evidence": (
                "task_2_result.tex contains paper-reported WSCR values, but no Polyvore preprocessing, "
                "broad-query generation outputs, or retrieval outputs are released."
            ),
        },
        {
            "status": "toy",
            "evidence": (
                "The source includes reward-collapse figure assets and text. The toy reward check gives "
                f"collapsed total {toy['collapsed']['total']} and diverse total {toy['diverse']['total']}, "
                "showing diversity/alignment terms can penalize collapse on a synthetic fixture."
            ),
        },
        {
            "status": "inconclusive",
            "evidence": (
                "query_fanout_efficiency.pdf and manuscript text report latency, but no executable "
                "diffusion retriever, autoregressive baseline, hardware harness, or raw latency log is released."
            ),
        },
    ]

    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "snapshot_id": SNAPSHOT_ID,
        "title": TITLE,
        "generated_at": os.environ.get("REPRO_GENERATED_AT", "2026-08-01T17:48:00+00:00"),
        "upstream_pins": UPSTREAM_PINS,
        "commands": {
            "generate": "python generate_evidence.py",
            "test": "python -m pytest tests -q",
        },
        "audits": {
            "source_manifest": manifest,
            "tables": tables,
            "method_text_markers": {
                "three_stage_method": all(marker in method_text for marker in ("fan-out policy", "synthetic", "diffusion")),
                "reward_components": all(marker in method_text for marker in ("groundedness", "diversity", "alignment")),
                "latency_text": "4.21 seconds" in method_text and "50 seconds" in method_text,
            },
            "toy_reward": toy,
        },
        "claim_results": [
            {
                "claim_index": claim["index"],
                "claim_sha256": claim["challenge_claim_sha256"],
                "target_claim": claim["target_claim"],
                **statuses[index],
            }
            for index, claim in enumerate(CLAIMS)
        ],
    }


def fetch_arxiv_source() -> dict[str, str]:
    request = urllib.request.Request(
        f"https://arxiv.org/e-print/{ARXIV_ID}",
        headers={"User-Agent": "icml-repro-loop"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EPRINT_SHA256:
        raise ValueError(f"arXiv e-print digest mismatch: {digest}")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        source: dict[str, str] = {}
        for member in tar.getmembers():
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            try:
                source[member.name] = payload.decode("utf-8")
            except UnicodeDecodeError:
                source[member.name] = "binary"
        return source


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def write_evidence(bundle: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    project = Path(__file__).resolve().parents[2]
    default_output = project / "evidence" / "bundle.json"
    if output.resolve() != default_output.resolve():
        return
    (project / "pages").mkdir(exist_ok=True)
    (project / "pages" / "report.md").write_text(render_report(bundle), encoding="utf-8")


def render_report(bundle: dict[str, Any]) -> str:
    lines = [
        f"# {bundle['title']}",
        "",
        f"- Attempt: `{bundle['attempt_id']}`",
        f"- Paper: `{bundle['paper_id']}`",
        f"- Snapshot: `{bundle['snapshot_id']}`",
        f"- Source pin: `{bundle['upstream_pins']['arxiv_eprint']}`",
        "",
        "## Claim Evidence",
    ]
    for result in bundle["claim_results"]:
        lines.extend(["", f"### Claim {result['claim_index']}: {result['status']}", "", result["target_claim"], "", result["evidence"]])
    lines.extend(["", "## Audits", "", "```json", json.dumps(bundle["audits"], indent=2, sort_keys=True), "```", "", "No paper-reported table or latency value is presented as a recomputed measurement.", ""])
    return "\n".join(lines)
