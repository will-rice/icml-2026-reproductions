from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any


ATTEMPT_ID = "8d3d77be-6e6a-48a0-b50e-3a078786181d"
PAPER_ID = "oBgLvd5YC6"
SNAPSHOT_ID = "b4f93b39c8b36e72a5c6e1cb4712583e6f78510e414749bc064f46a09ce81885"
GENERATED_AT = "2026-08-01T14:05:00+00:00"

CODE_REPO = "https://github.com/aiming-lab/SimpleMem.git"
CODE_REVISION = "db80b6a7c591e0ea730a058e9f5fc4eb06572299"
ARXIV_SOURCE_SHA256 = "d75d00ede2529a7656b6c6030161d489630dcf03a9f538dbd20322aa6a69f08a"
ARXIV_PDF_SHA256 = "8752aa223e004ca286995bc1e8cbde8e89e67ad3aeb9ba0266f3ccab3cc11078"

CLAIM_BINDINGS = [
    {
        "target_claim": "SimpleMem uses a three-stage memory pipeline: semantic structured compression, online semantic synthesis, and intent-aware retrieval planning (Figure 2).",
        "challenge_claim": "SimpleMem uses a three-stage memory pipeline: semantic structured compression, online semantic synthesis, and intent-aware retrieval planning (Figure 2).",
        "challenge_claim_sha256": "c447ee8488120566377eecb650e62b3b423204b6facb5d5847e9a6d667f6f438",
    },
    {
        "target_claim": "On LoCoMo with high-capability backends, SimpleMem achieves a superior F1/token-cost trade-off over baseline memory systems (Figure 1; Table 1).",
        "challenge_claim": "On LoCoMo with high-capability backends, SimpleMem achieves a superior F1/token-cost trade-off over baseline memory systems (Figure 1; Table 1).",
        "challenge_claim_sha256": "e6e9fc9c337f6c86aabec2bbd55b0cb537bee0cb2e47fe49c7481532978d742d",
    },
    {
        "target_claim": "On LongMemEval-S, SimpleMem obtains the best overall performance while maintaining balanced sub-task accuracy (Table 2).",
        "challenge_claim": "On LongMemEval-S, SimpleMem obtains the best overall performance while maintaining balanced sub-task accuracy (Table 2).",
        "challenge_claim_sha256": "e23d442bfe5504a53e95a3d90213c1af85a36777d7f6b136fb81e8ab749508fe",
    },
    {
        "target_claim": "SimpleMem remains effective with small models, with 1.5B/3B backends often outperforming larger models using baseline memory systems (Table 3).",
        "challenge_claim": "SimpleMem remains effective with small models, with 1.5B/3B backends often outperforming larger models using baseline memory systems (Table 3).",
        "challenge_claim_sha256": "7860b82b3959720fe3736730a0965082b8e688a9a1919587994e23128f39eccb",
    },
    {
        "target_claim": "On LoCoMo-10 with GPT-4.1-mini, SimpleMem reduces construction and retrieval time versus graph- or summary-based baselines while achieving the highest average F1 (Table 4).",
        "challenge_claim": "On LoCoMo-10 with GPT-4.1-mini, SimpleMem reduces construction and retrieval time versus graph- or summary-based baselines while achieving the highest average F1 (Table 4).",
        "challenge_claim_sha256": "a025d2297e521330de7565c181e5bd5fc11600ed721695c68001f15db54fa44b",
    },
    {
        "target_claim": "Ablations show that semantic structured compression, online synthesis, and intent-aware retrieval planning each contribute materially to reasoning performance (Table 5).",
        "challenge_claim": "Ablations show that semantic structured compression, online synthesis, and intent-aware retrieval planning each contribute materially to reasoning performance (Table 5).",
        "challenge_claim_sha256": "240a8b0158dff372ed76fde5954fa36aec3f12501b1227938cc74280143f8ae4",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_arxiv_main(arxiv_source: Path) -> str:
    with tarfile.open(arxiv_source, "r:*") as archive:
        member = archive.extractfile("main_arxiv.tex")
        if member is None:
            raise ValueError("main_arxiv.tex missing from arXiv source")
        return member.read().decode("utf-8", "replace")


def detect_license(source_root: Path) -> str:
    text = read_text(source_root / "LICENSE")
    if "MIT License" not in text:
        raise ValueError("Expected MIT license text in upstream repository")
    return "MIT"


def collect_source_release(source_root: Path, arxiv_source: Path) -> dict[str, Any]:
    readme = read_text(source_root / "README.md")
    arxiv_text = read_arxiv_main(arxiv_source)
    combined = f"{readme}\n{arxiv_text}".lower()

    table_labels = {
        "high_capacity_results": "tab:high_cap_results" in arxiv_text,
        "longmemeval_full": "tab:longmemeval_full" in arxiv_text,
        "efficient_results": "tab:efficient_results" in arxiv_text,
        "memory_time_accuracy": "tab:mem_times_acc" in arxiv_text,
        "ablation": "tab:ablation" in arxiv_text,
    }
    component_paths = {
        "memory_builder": (source_root / "simplemem" / "core" / "memory_builder.py").is_file(),
        "hybrid_retriever": (source_root / "simplemem" / "core" / "hybrid_retriever.py").is_file(),
        "answer_generator": (source_root / "simplemem" / "core" / "answer_generator.py").is_file(),
    }
    return {
        "code_revision": run_git(["rev-parse", "HEAD"], source_root),
        "tracked_file_count": len(run_git(["ls-files"], source_root).splitlines()),
        "pipeline_terms": {
            "semantic_structured_compression": "semantic structured compression" in combined,
            "online_semantic_synthesis": "online semantic synthesis" in combined,
            "intent_aware_retrieval_planning": "intent-aware retrieval planning" in combined,
        },
        "component_paths_present": component_paths,
        "arxiv_table_labels_present": table_labels,
    }


def collect_benchmark_release(source_root: Path) -> dict[str, Any]:
    locomo_runner = source_root / "test_locomo10.py"
    evolvemem_runner = source_root / "EvolveMem" / "run_benchmark.py"
    benchmark_text = "\n".join(
        read_text(path) for path in [locomo_runner, evolvemem_runner] if path.is_file()
    )
    raw_artifacts = find_raw_result_artifacts(source_root)
    return {
        "locomo_runner_present": locomo_runner.is_file(),
        "evolvemem_runner_present": evolvemem_runner.is_file(),
        "requires_openai_api_key": "OPENAI_API_KEY" in benchmark_text,
        "llm_judge_present": "llm_judge" in benchmark_text or "LLM-as-judge" in benchmark_text,
        "raw_result_artifact_count": len(raw_artifacts),
        "raw_result_artifacts": raw_artifacts,
    }


def find_raw_result_artifacts(source_root: Path) -> list[str]:
    result_names = {
        "locomo10_test_results.json",
        "raw_results.jsonl",
        "summary.json",
        "evolution_summary.json",
        "run_meta.json",
    }
    result_dirs = {"evolution_results", "results", "outputs"}
    artifacts: list[str] = []
    for path in source_root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(source_root)
        if path.name in result_names or any(part in result_dirs for part in relative.parts):
            if path.suffix.lower() in {".json", ".jsonl", ".csv", ".parquet"}:
                artifacts.append(relative.as_posix())
    return sorted(artifacts)


def build_claims(observations: dict[str, Any]) -> list[dict[str, Any]]:
    source = observations["source_release"]
    benchmark = observations["benchmark_release"]
    pipeline_detected = all(source["pipeline_terms"].values()) and all(
        source["component_paths_present"].values()
    )
    empirical_summary = (
        "The pinned repository exposes API-backed benchmark runners, but no raw "
        "result artifacts are present; paper table values are not reproduced evidence."
    )
    return [
        {
            "claim_index": 1,
            "status": "toy" if pipeline_detected else "unavailable",
            "summary": "The pinned source and arXiv source expose the three SimpleMem stages and corresponding implementation modules, but no end-to-end benchmark run was executed.",
            "evidence_basis": ["pinned_github_source", "pinned_arxiv_source"],
            "observed": {
                "pipeline_terms": source["pipeline_terms"],
                "component_paths_present": source["component_paths_present"],
            },
        },
        {
            "claim_index": 2,
            "status": "unavailable",
            "summary": empirical_summary,
            "evidence_basis": ["api_backed_runner", "artifact_absence"],
            "observed": benchmark,
        },
        {
            "claim_index": 3,
            "status": "unavailable",
            "summary": empirical_summary,
            "evidence_basis": ["api_backed_runner", "artifact_absence"],
            "observed": benchmark,
        },
        {
            "claim_index": 4,
            "status": "unavailable",
            "summary": empirical_summary,
            "evidence_basis": ["api_backed_runner", "artifact_absence"],
            "observed": benchmark,
        },
        {
            "claim_index": 5,
            "status": "unavailable",
            "summary": empirical_summary,
            "evidence_basis": ["api_backed_runner", "artifact_absence"],
            "observed": benchmark,
        },
        {
            "claim_index": 6,
            "status": "unavailable",
            "summary": "The arXiv source contains the ablation table label, but the release does not include raw ablation outputs or a CPU-only rerun path without model APIs.",
            "evidence_basis": ["pinned_arxiv_source", "artifact_absence"],
            "observed": {
                "ablation_table_label_present": source["arxiv_table_labels_present"]["ablation"],
                "raw_result_artifact_count": benchmark["raw_result_artifact_count"],
            },
        },
    ]


def build_bundle(
    source_root: Path,
    arxiv_source: Path,
    arxiv_pdf: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    arxiv_source = arxiv_source.resolve()
    arxiv_pdf = arxiv_pdf.resolve()

    code_revision = run_git(["rev-parse", "HEAD"], source_root)
    if code_revision != CODE_REVISION:
        raise ValueError(f"source root is at {code_revision}, expected {CODE_REVISION}")
    arxiv_source_sha = sha256_file(arxiv_source)
    arxiv_pdf_sha = sha256_file(arxiv_pdf)
    if arxiv_source_sha != ARXIV_SOURCE_SHA256:
        raise ValueError(f"arXiv source SHA mismatch: {arxiv_source_sha}")
    if arxiv_pdf_sha != ARXIV_PDF_SHA256:
        raise ValueError(f"arXiv PDF SHA mismatch: {arxiv_pdf_sha}")

    observations = {
        "source_release": collect_source_release(source_root, arxiv_source),
        "benchmark_release": collect_benchmark_release(source_root),
    }
    bundle = {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "challenge_snapshot_id": SNAPSHOT_ID,
        "generated_at": GENERATED_AT,
        "claim_bindings": CLAIM_BINDINGS,
        "provenance": {
            "code": {
                "repo": CODE_REPO,
                "revision": code_revision,
            },
            "arxiv_source_sha256": arxiv_source_sha,
            "arxiv_pdf_sha256": arxiv_pdf_sha,
            "license": detect_license(source_root),
            "commands": [
                "git clone https://github.com/aiming-lab/SimpleMem.git /tmp/simplemem-upstream-codex03",
                f"git -C /tmp/simplemem-upstream-codex03 checkout {CODE_REVISION}",
                "curl -L https://arxiv.org/e-print/2601.02553 -o /tmp/simplemem-2601.02553-src.tar",
                "curl -L https://arxiv.org/pdf/2601.02553 -o /tmp/simplemem-2601.02553.pdf",
            ],
        },
        "observations": observations,
        "claims": build_claims(observations),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--arxiv-source", type=Path, required=True)
    parser.add_argument("--arxiv-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_bundle(args.source_root, args.arxiv_source, args.arxiv_pdf, args.output)


if __name__ == "__main__":
    main()
