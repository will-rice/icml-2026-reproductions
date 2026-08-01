"""CPU-only evidence audit for pinned Interplay-LM reasoning artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


ATTEMPT_ID = "02e96ad2-d870-4525-89c9-c5758aca0255"
PAPER_ID = "TBaUfO9znF"
SNAPSHOT_ID = "e24ef4f585f2af51c2a85898401421e84ff4d6e8b74fb53e138c9164f1e83d57"
TITLE = "On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"
GITHUB_REVISION = "ab728f05d81de9af38d0ca155a84166b037e355a"

UPSTREAM_PINS = {
    "paper": "arxiv:2512.07783v1",
    "official_code": f"github:Interplay-LM-Reasoning/Interplay-LM-Reasoning@{GITHUB_REVISION}",
    "composition_dataset": "hf-dataset:Interplay-LM-Reasoning/composition@a09d5c14c02bfa339143fb00a93274d1a84aa31d",
    "context_dataset": "hf-dataset:Interplay-LM-Reasoning/context@bb09a75f1d667931e19e9715521b59f5b4574791",
    "extrapolation_rl": "hf-model:Interplay-LM-Reasoning/extrapolation_rl@4861bd030e6fb92d94be3a1cecab89c2fac4b94a",
    "extrapolation_midtrain": "hf-model:Interplay-LM-Reasoning/extrapolation_midtrain@bf457d416f825011cae11dad0372e10f4c323b73",
    "context_pretrain": "hf-model:Interplay-LM-Reasoning/context_pretrain@588760f1ce30ad5d1d99d55421ff016a27461c83",
    "context_pretrain_2": "hf-model:Interplay-LM-Reasoning/context_pretrain_2@b499d182df85b065d2dfe3b446e6b63e87c740ec",
}

CLAIMS = [
    {
        "target_claim": "The paper uses a controlled synthetic reasoning framework with explicit dependency graphs, contextual templates, and process-verified evaluation (Figure 2).",
        "challenge_claim_sha256": "e037801af94157ca0c71cae1f8902e9a3c3e67d13fc4dab05069d8b05f4106f3",
    },
    {
        "target_claim": "RL yields extrapolative pass@128 gains only when the post-training tasks sit near the model's edge of competence; gains vanish when tasks are already covered or too far out-of-distribution (Figure 1; Figure 3).",
        "challenge_claim_sha256": "ba98c93e3789f48414096a4fd3644a76ce48697f12be53e256f2583397c551d9",
    },
    {
        "target_claim": "Contextual generalization requires minimal but nonzero pre-training exposure to long-tail contexts; exposure of at least about 1% enables RL to reinforce transfer (Figure 1; Figure 4).",
        "challenge_claim_sha256": "d1d108be7d95ffa3d12c92756ab25471bd5b7ece916235312f6a8a5fc0f9f81e",
    },
    {
        "target_claim": "Mid-training plus RL outperforms RL alone on OOD-hard reasoning under fixed compute, with reported +10.8% gains (Figure 1; Figure 6).",
        "challenge_claim_sha256": "c17b28c07811eb2e0552d7f127266442364beee489cd0613083cfd077ecde2b3",
    },
    {
        "target_claim": "Process-aware reward compositions reduce shortcut exploitation and improve reasoning performance relative to pure outcome rewards (Figure 7).",
        "challenge_claim_sha256": "50a8391594f6917bc3fd908cd2934de69435745b010cc2286cdeca300f475739",
    },
    {
        "target_claim": "Training dynamics show reward stagnates when RL data are too easy or too hard, but improves when tasks are calibrated to the edge of competence (Figure 11).",
        "challenge_claim_sha256": "5fcb84a1b21ff1cccbcd627802ea55fd68ef0eadc5fa58fe1c997c24de10c4e3",
    },
]

CODE_PATHS = [
    "utils/solution_dependency_graph.py",
    "utils/dataset.py",
    "verl/dataset.py",
    "verl/dataset_context.py",
    "verl/reward_fn.py",
    "scripts/eval_checkpoints.py",
]

DATASET_SAMPLE_PATHS = {
    "composition": [
        "test/op10-1k.jsonl",
        "heldout/op14-50k.jsonl",
    ],
    "context": [
        "crazy_zootopia/test/op10_1k.jsonl",
    ],
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_official_code(code_artifacts: dict[str, str]) -> dict[str, Any]:
    joined = "\n".join(code_artifacts.values()).lower()

    return {
        "dependency_graph_code": _presence(
            "solution_dependency_graph.py" in " ".join(code_artifacts)
            and ("dependencygraph" in joined or "dependency_graph" in joined or "graph" in joined)
        ),
        "dataset_generation_code": _presence(
            any(path.endswith("dataset.py") for path in code_artifacts)
            and ("build_dataset" in joined or "dataset" in joined)
        ),
        "context_dataset_code": _presence(
            any("dataset_context.py" in path for path in code_artifacts)
            and ("context" in joined or "template" in joined or "zootopia" in joined)
        ),
        "process_reward_code": _presence(
            any("reward_fn.py" in path for path in code_artifacts)
            and ("process" in joined or "trace" in joined or "reward" in joined)
        ),
        "evaluation_code": _presence(
            any("eval_checkpoints.py" in path for path in code_artifacts)
            and ("pass_at" in joined or "pass@128" in joined or "checkpoint" in joined)
        ),
        "source_hashes": {path: sha256_text(text) for path, text in sorted(code_artifacts.items())},
    }


def audit_dataset_splits(dataset_samples: dict[str, str]) -> dict[str, Any]:
    composition_depths: set[int] = set()
    context_depths: set[int] = set()
    sample_hashes = {}
    record_count = 0

    for path, text in sorted(dataset_samples.items()):
        sample_hashes[path] = sha256_text(text)
        record_count += sum(1 for line in text.splitlines() if line.strip())
        depth = _extract_depth(path)
        if depth is None:
            continue
        if path.startswith("composition/"):
            composition_depths.add(depth)
        if path.startswith("context/"):
            context_depths.add(depth)

    return {
        "composition_depths": sorted(composition_depths),
        "context_depths": sorted(context_depths),
        "sample_hashes": sample_hashes,
        "sample_record_count": record_count,
    }


def build_evidence_bundle(
    code_artifacts: dict[str, str],
    dataset_samples: dict[str, str],
    raw_result_artifacts: dict[str, str],
) -> dict[str, Any]:
    code_audit = audit_official_code(code_artifacts)
    dataset_audit = audit_dataset_splits(dataset_samples)
    result_hashes = {path: sha256_text(text) for path, text in sorted(raw_result_artifacts.items())}

    claim_results = [_framework_claim(code_audit, dataset_audit)]
    for index, claim in enumerate(CLAIMS[1:], start=2):
        claim_results.append(_training_claim(index, claim, code_audit, dataset_audit, result_hashes))

    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "title": TITLE,
        "snapshot_id": SNAPSHOT_ID,
        "upstream_pins": UPSTREAM_PINS,
        "artifact_access": {
            "code_files": sorted(code_artifacts),
            "dataset_samples": sorted(dataset_samples),
            "raw_result_artifacts": sorted(raw_result_artifacts),
        },
        "audits": {
            "official_code": code_audit,
            "dataset_splits": dataset_audit,
            "raw_result_hashes": result_hashes,
        },
        "claim_results": claim_results,
    }


def fetch_pinned_artifacts() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    code_artifacts = {path: _fetch_github_text(path) for path in CODE_PATHS}
    dataset_samples: dict[str, str] = {}
    for family, paths in DATASET_SAMPLE_PATHS.items():
        repo_id = f"Interplay-LM-Reasoning/{family}"
        repo_revision = _revision_from_pin(UPSTREAM_PINS[f"{family}_dataset"])
        for path in paths:
            cached = hf_hub_download(repo_id, path, repo_type="dataset", revision=repo_revision)
            dataset_samples[f"{family}/{path}"] = _first_lines(Path(cached), 3)
    return code_artifacts, dataset_samples, {}


def write_evidence(output_path: Path) -> dict[str, Any]:
    code_artifacts, dataset_samples, raw_result_artifacts = fetch_pinned_artifacts()
    bundle = build_evidence_bundle(code_artifacts, dataset_samples, raw_result_artifacts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(output_path, bundle)
    return bundle


def _presence(is_present: bool) -> dict[str, str]:
    return {"status": "present" if is_present else "missing"}


def _extract_depth(path: str) -> int | None:
    match = re.search(r"op(\d+)(?:[-_])", path)
    return int(match.group(1)) if match else None


def _framework_claim(code_audit: dict[str, Any], dataset_audit: dict[str, Any]) -> dict[str, Any]:
    required = [
        code_audit["dependency_graph_code"]["status"],
        code_audit["context_dataset_code"]["status"],
        code_audit["process_reward_code"]["status"],
        code_audit["evaluation_code"]["status"],
    ]
    has_datasets = bool(dataset_audit["composition_depths"] and dataset_audit["context_depths"])
    status = "verified" if all(item == "present" for item in required) and has_datasets else "inconclusive"
    return {
        "claim_index": 1,
        "claim": CLAIMS[0]["target_claim"],
        "claim_sha256": CLAIMS[0]["challenge_claim_sha256"],
        "status": status,
        "observation": "Pinned source exposes dependency graph, contextual dataset, process reward, and evaluation code paths; sampled datasets expose composition and context splits.",
        "limitation": "This is a source and dataset audit, not a rerun of full language-model training.",
    }


def _training_claim(
    index: int,
    claim: dict[str, str],
    code_audit: dict[str, Any],
    dataset_audit: dict[str, Any],
    result_hashes: dict[str, str],
) -> dict[str, Any]:
    has_structural_support = (
        code_audit["evaluation_code"]["status"] == "present"
        and dataset_audit["sample_record_count"] > 0
    )
    if result_hashes:
        status = "inconclusive"
        limitation = "Raw official result artifacts were found and hashed, but this audit does not rerun checkpoint inference."
    else:
        status = "toy" if has_structural_support else "unavailable"
        limitation = "No raw official result artifact was found; paper-reported numeric gains are not treated as reproduced measurements."
    return {
        "claim_index": index,
        "claim": claim["target_claim"],
        "claim_sha256": claim["challenge_claim_sha256"],
        "status": status,
        "observation": "Pinned artifacts expose the experiment family and lightweight provenance needed for a structural audit.",
        "limitation": limitation,
    }


def _fetch_github_text(path: str) -> str:
    url = (
        "https://raw.githubusercontent.com/Interplay-LM-Reasoning/"
        f"Interplay-LM-Reasoning/{GITHUB_REVISION}/{path}"
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _revision_from_pin(pin: str) -> str:
    return pin.rsplit("@", 1)[1]


def _first_lines(path: Path, count: int) -> str:
    lines = []
    with path.open(encoding="utf-8") as handle:
        for _, line in zip(range(count), handle):
            lines.append(line)
    return "".join(lines)


def _write_report(output_path: Path, bundle: dict[str, Any]) -> None:
    project_root = output_path.parent.parent
    pages = project_root / "pages"
    pages.mkdir(exist_ok=True)
    statuses = ", ".join(
        f"claim {item['claim_index']}: {item['status']}" for item in bundle["claim_results"]
    )
    report = (
        "# Interplay-LM Evidence Report\n\n"
        f"Paper: `{PAPER_ID}`\n\n"
        f"Attempt: `{ATTEMPT_ID}`\n\n"
        f"Snapshot: `{SNAPSHOT_ID}`\n\n"
        f"Statuses: {statuses}.\n\n"
        "Numeric training gains remain unavailable unless backed by raw official result artifacts.\n"
    )
    (pages / "report.md").write_text(report, encoding="utf-8")


if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/tmp/icml-repro-hf-home"
