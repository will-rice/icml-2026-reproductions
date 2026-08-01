from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


PAPER_ID = "DVHpvumD60"
ATTEMPT_ID = "3d8acf83-97a0-43d3-a2fb-ea3e2b3c2b12"
SNAPSHOT_ID = "98fe583a0a55974d2d28e1beba12e398eb7e8b7f05fadb6d33fdd243b1988644"
UPSTREAM_REPO = "https://github.com/LLMServe/WarmServe.git"
UPSTREAM_LABEL = "LLMServe/WarmServe"
UPSTREAM_COMMIT = "a60121519e077d2f128b597cbabc947e3e618aaf"
ARXIV_SOURCE_SHA256 = "c945e654f3309f6de207c29c6151305a8ae4e53163846946958befc30a9d05d5"
ARXIV_PDF_SHA256 = "9ac0e419ccfb3f8132f73ad662bbbaca842535b0e882346bac0b9882f2399ed2"
EVIDENCE_GENERATED_AT = "2026-08-01T12:58:00+00:00"

CONTROLLER = Path("vllm-0.6.3.post1/vllm/entrypoints/controller")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_text(root: Path, relative: Path) -> str:
    return (root / relative).read_text(encoding="utf-8")


def run_git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def ensure_source(source_root: str | None = None) -> Path:
    configured = source_root or os.environ.get("WARMSERVE_SOURCE_ROOT")
    if configured:
        root = Path(configured).resolve()
        commit = run_git(["rev-parse", "HEAD"], root)
        if commit != UPSTREAM_COMMIT:
            raise ValueError(f"source root is at {commit}, expected {UPSTREAM_COMMIT}")
        return root
    root = Path("/tmp/warmserve-upstream-cache") / UPSTREAM_COMMIT
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        run_git(["clone", "--filter=blob:none", UPSTREAM_REPO, str(root)])
    run_git(["checkout", "--detach", UPSTREAM_COMMIT], root)
    return root


def file_record(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    return {"path": relative.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def snippet(text: str, marker: str, radius: int = 180) -> str:
    index = text.find(marker)
    if index < 0:
        return ""
    return " ".join(text[max(0, index - radius) : index + len(marker) + radius].split())


def contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def arxiv_record(path: str | None, expected: str, label: str) -> dict[str, Any]:
    if not path:
        return {"label": label, "available": False, "expected_sha256": expected}
    p = Path(path)
    actual = sha256_file(p)
    return {
        "label": label,
        "available": p.exists(),
        "path": str(p),
        "sha256": actual,
        "expected_sha256": expected,
        "matches_expected": actual == expected,
        "bytes": p.stat().st_size,
    }


def collect_observations(root: Path, arxiv_source: str | None, arxiv_pdf: str | None) -> dict[str, Any]:
    scheduler = read_text(root, CONTROLLER / "scheduler.py")
    manager = read_text(root, CONTROLLER / "prewarm_manager.py")
    vmm = read_text(root, CONTROLLER / "vmm.py")
    utils = read_text(root, CONTROLLER / "utils.py")
    model_config = read_text(root, CONTROLLER / "ModelConfig.py")
    trace_generator = read_text(root, Path("trace-generator/trace_generator.py"))
    trace_models = yaml.safe_load(read_text(root, Path("trace-generator/models.yaml")))

    indicators = {
        "scheduler": {
            "present": contains_all(
                scheduler,
                ["class Scheduler", "predicted_peak_load", "prewarm(", "_calc_model_scores"],
            ),
            "path": (CONTROLLER / "scheduler.py").as_posix(),
            "snippet": snippet(scheduler, "class Scheduler"),
        },
        "prewarm_manager": {
            "present": contains_all(
                manager,
                ["class PrewarmManager", "use_unified_memory", "disable_kv_prewarm", "Scheduler"],
            ),
            "path": (CONTROLLER / "prewarm_manager.py").as_posix(),
            "snippet": snippet(manager, "class PrewarmManager"),
        },
        "vmm": {
            "present": contains_all(
                vmm,
                ["cuMemAddressReserve", "cuMemMap", "cuMemUnmap", "map_blocks"],
            ),
            "path": (CONTROLLER / "vmm.py").as_posix(),
            "snippet": snippet(vmm, "class CUDAVMMPool"),
        },
        "worker_hooks": {
            "present": contains_all(
                utils,
                ["class MyWorkerWrapper", "init_unified_memory", "free_kv_space", "load_model"],
            ),
            "path": (CONTROLLER / "utils.py").as_posix(),
            "snippet": snippet(utils, "class MyWorkerWrapper"),
        },
        "trace_generator": {
            "present": contains_all(
                trace_generator,
                ["INTERVAL = 5 * 60", "get_character_from_csv", "get_prewarm_placement"],
            ),
            "path": "trace-generator/trace_generator.py",
            "snippet": snippet(trace_generator, "def get_prewarm_placement"),
        },
        "model_config": {
            "present": contains_all(model_config, ["ModelList", "ModelKVConfig", "MODEL_PATH"]),
            "path": (CONTROLLER / "ModelConfig.py").as_posix(),
            "snippet": snippet(model_config, "ModelKVConfig"),
        },
    }

    raw_results = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and "vllm-0.6.3.post1/tests" not in path.as_posix()
        and path.suffix.lower() in {".csv", ".jsonl", ".parquet", ".feather", ".pkl"}
        and any(part in path.as_posix().lower() for part in ["result", "log", "eval"])
    ]

    return {
        "source_tree": {
            "repo": UPSTREAM_REPO,
            "commit": run_git(["rev-parse", "HEAD"], root),
            "tracked_file_count": len(run_git(["ls-files"], root).splitlines()),
        },
        "files": [
            file_record(root, Path("README.md")),
            file_record(root, Path("LICENSE")),
            file_record(root, CONTROLLER / "scheduler.py"),
            file_record(root, CONTROLLER / "prewarm_manager.py"),
            file_record(root, CONTROLLER / "vmm.py"),
            file_record(root, CONTROLLER / "utils.py"),
            file_record(root, Path("trace-generator/trace_generator.py")),
            file_record(root, Path("trace-generator/models.yaml")),
        ],
        "arxiv": {
            "source": arxiv_record(arxiv_source, ARXIV_SOURCE_SHA256, "arxiv-source-v2"),
            "pdf": arxiv_record(arxiv_pdf, ARXIV_PDF_SHA256, "arxiv-pdf-v2"),
        },
        "source_indicators": indicators,
        "trace_generator": {
            "interval_seconds": 300 if "INTERVAL = 5 * 60" in trace_generator else None,
            "cluster": trace_models.get("cluster"),
            "model_yaml_models": len(trace_models.get("models", [])),
        },
        "runtime_constraints": {
            "cpu_only_reproduction": True,
            "gpu_cluster_required": True,
            "requires_cuda_extension": True,
            "requires_ray_vllm_models_and_traces": True,
        },
        "raw_result_artifacts": raw_results,
    }


def build_claim_results(observations: dict[str, Any]) -> dict[str, Any]:
    indicators = observations["source_indicators"]
    source_verified = all(item["present"] for item in indicators.values())
    raw_results = bool(observations["raw_result_artifacts"])
    return {
        "claim-1": {
            "challenge_claim_sha256": "0c0a74f9b2474f2a5c7861cd114a8540f6a35cad8376727574edc2deb6595b72",
            "status": "verified" if source_verified else "inconclusive",
            "summary": "Pinned source contains scheduler, prewarm manager, CUDA VMM, worker hooks, model config, and trace generator paths for the WarmServe design.",
        },
        "claim-2": {
            "challenge_claim_sha256": "74062b2b618fbcd71336f1910dc2ca3c093fb405878a352dc4025c9b85680135",
            "status": "toy",
            "summary": "Trace-character generation and 5-minute windows are present, but AzureConv 7.3% prediction error is not recomputed without the dataset.",
        },
        "claim-3": {
            "challenge_claim_sha256": "554d16b3aacae8ab9f26fda580eea541c9a22b74a2a4597602f63057eae3a5ac",
            "status": "unavailable" if not raw_results else "inconclusive",
            "summary": "TTFT prewarming measurements require CUDA/Ray/vLLM deployment and raw logs absent from the pinned repo.",
        },
        "claim-4": {
            "challenge_claim_sha256": "aa8aacfa03b2d61f40b5bc023835e12031c912f34805afa06ac9c0cb99970ad2",
            "status": "unavailable" if not raw_results else "inconclusive",
            "summary": "End-to-end 50.8x tail TTFT claim was not rerun on a GPU cluster.",
        },
        "claim-5": {
            "challenge_claim_sha256": "6182c76e4e228313cf3145aae8dc6e915d9645424f5b5f8d8032b07ec05264a0",
            "status": "unavailable" if not raw_results else "inconclusive",
            "summary": "Ablation claims require benchmark logs or reruns that are not available in this CPU audit.",
        },
        "claim-6": {
            "challenge_claim_sha256": "b6ad7a59b380d5d0e26e414fc8d831d0da015dc23ed944cd1e5d1bdd7f2e44c8",
            "status": "unavailable" if not raw_results else "inconclusive",
            "summary": "512-GPU P99 TTFT simulation was not rerun and no raw simulation artifact was found.",
        },
    }


def build_evidence(
    source_root: str | None = None,
    arxiv_source: str | None = None,
    arxiv_pdf: str | None = None,
) -> dict[str, Any]:
    root = ensure_source(source_root)
    observations = collect_observations(root, arxiv_source, arxiv_pdf)
    return {
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "snapshot_id": SNAPSHOT_ID,
        "generated_at": EVIDENCE_GENERATED_AT,
        "upstream": {
            "github": f"{UPSTREAM_LABEL}@{UPSTREAM_COMMIT}",
            "url": UPSTREAM_REPO,
            "code_license": "Apache-2.0",
            "paper": "arxiv:2512.09472v2",
            "arxiv_source_sha256": ARXIV_SOURCE_SHA256,
            "arxiv_pdf_sha256": ARXIV_PDF_SHA256,
        },
        "observations": observations,
        "claim_results": build_claim_results(observations),
        "unreplicated": [
            "AzureConv 7.3% peak-load prediction error was not recomputed because the released repository contains dataset paths, not the trace CSV.",
            "TTFT reductions require CUDA memory extension, Ray, vLLM, model checkpoints, and generated workloads.",
            "Ablation and end-to-end benchmark claims were not rerun without GPU cluster logs.",
            "512-GPU simulation P99 TTFT was not rerun and no raw simulation output was found.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--arxiv-source", default=None)
    parser.add_argument("--arxiv-pdf", default=None)
    parser.add_argument("--output", default="evidence/bundle.json")
    args = parser.parse_args(argv)
    bundle = build_evidence(args.source_root, args.arxiv_source, args.arxiv_pdf)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "paper_id": PAPER_ID, "claim_results": bundle["claim_results"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
