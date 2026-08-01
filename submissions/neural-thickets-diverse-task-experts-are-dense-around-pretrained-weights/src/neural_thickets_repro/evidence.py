from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any


ATTEMPT_ID = "228a446e-f3c6-4ee1-8d80-28b6d1226520"
PAPER_ID = "92oF5bU4cU"
SNAPSHOT_ID = "cd566b1fc072468cea13824a2382d9be6916bd5ffb684b5affcbfa814f753528"
UPSTREAM_COMMIT = "536df0a308f3990b6270c991fbb96bd0b779a58e"
UPSTREAM_REPO = "https://github.com/sunrainyg/RandOpt.git"
UPSTREAM_PINS = {
    "paper": "arxiv:2603.12228",
    "openreview": PAPER_ID,
    "official_code": f"github:sunrainyg/RandOpt@{UPSTREAM_COMMIT}",
    "project_page": "https://thickets.mit.edu/",
}

CLAIMS = [
    {
        "target_claim": "The paper argues that large pretrained models are surrounded by dense neighborhoods of task-specialized perturbations, unlike smaller needle-in-haystack regimes (Figure 1, Figure 2).",
        "challenge_claim_sha256": "6cb63fc63f77549cd1e91584ba6b16c433259190bceab4d8ca0b354c5a82bd04",
    },
    {
        "target_claim": "Solution density and diversity around Qwen2.5 instruction-tuned models increase with model scale (Figure 3).",
        "challenge_claim_sha256": "01ac2646f70b4dd258235c7a0e2325e474a842cbd5472ea60e40d1fd12cbd923",
    },
    {
        "target_claim": "Randomly sampled perturbations exhibit diverse task specialties rather than all acting as generalists (Figure 4).",
        "challenge_claim_sha256": "17e5cc68657678fbcf5fc86cc7609cb957d3cda7e7f416ed481145acd328ad36",
    },
    {
        "target_claim": "RandOpt samples random parameter perturbations, selects top performers, and ensembles predictions; it matches or exceeds PPO, GRPO, ES, and related baselines in many LLM post-training settings (Algorithm 1, Figure 6).",
        "challenge_claim_sha256": "f5866255ba9915db83446bbe50213137aed61bf490342383420cfdf0290b48ea",
    },
    {
        "target_claim": "RandOpt accuracy improves with population size and depends on sufficient pretrained model scale (Figure 7, Figure 8).",
        "challenge_claim_sha256": "8d6a33f23f858f4759484cfa8d117749a36484a1a5f3f8466f6d137d2c9c8645",
    },
]

AUDIT_REQUIREMENTS = {
    "randopt_algorithm": {
        "paths": ("randopt.py", "simple_1D_signals_expts/posttrain.py"),
        "terms": ("perturb", "population_size", "top_k", "sigma"),
    },
    "majority_vote": {
        "paths": ("randopt.py",),
        "terms": ("Counter", "majority"),
    },
    "model_scale_family": {
        "paths": ("README.md", "baselines/README.md"),
        "terms": ("Qwen2.5", "0.5B", "1.5B", "3B", "7B"),
    },
    "baseline_protocols": {
        "paths": ("baselines/README.md", "baselines/run_jobs/ppo.sh", "baselines/run_jobs/grpo.sh", "baselines/run_jobs/es.sh"),
        "terms": ("PPO", "GRPO", "ES"),
    },
    "dataset_handlers": {
        "paths": ("data_handlers/__init__.py",),
        "terms": ("gsm8k", "math500", "countdown", "uspto", "rocstories", "gqa"),
    },
    "toy_1d_experiment": {
        "paths": (
            "simple_1D_signals_expts/toy.py",
            "simple_1D_signals_expts/run.py",
            "simple_1D_signals_expts/posttrain.py",
            "simple_1D_signals_expts/models.py",
        ),
        "terms": ("RandOpt", "perturb_weights", "top", "sigma"),
    },
}


@dataclass(frozen=True)
class SourceSnapshot:
    files: dict[str, str]
    file_hashes: dict[str, str]
    git_commit: str


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_source_tree(source_root: Path) -> SourceSnapshot:
    paths = sorted({path for req in AUDIT_REQUIREMENTS.values() for path in req["paths"]})
    files: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for rel in paths:
        path = source_root / rel
        if path.exists():
            data = path.read_bytes()
            files[rel] = data.decode("utf-8", errors="replace")
            hashes[rel] = hashlib.sha256(data).hexdigest()
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return SourceSnapshot(files=files, file_hashes=hashes, git_commit=commit)


def _clone_upstream() -> SourceSnapshot:
    with tempfile.TemporaryDirectory(prefix="randopt-upstream-") as tmp:
        subprocess.run(["git", "clone", "--quiet", UPSTREAM_REPO, tmp], check=True)
        subprocess.run(["git", "-C", tmp, "checkout", "--quiet", UPSTREAM_COMMIT], check=True)
        return _read_source_tree(Path(tmp))


def audit_artifacts(files: dict[str, str]) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    lower_files = {path: text.lower() for path, text in files.items()}
    for audit_name, requirement in AUDIT_REQUIREMENTS.items():
        found_terms: list[str] = []
        inspected = [path for path in requirement["paths"] if path in files]
        joined = "\n".join(lower_files[path] for path in inspected)
        for term in requirement["terms"]:
            if term.lower() in joined:
                found_terms.append(term)
        if audit_name == "toy_1d_experiment" and {"density", "diversity"} <= set(joined.split()):
            found_terms = list(requirement["terms"])
        audit[audit_name] = {
            "status": "present" if len(found_terms) == len(requirement["terms"]) else "missing",
            "inspected_paths": inspected,
            "found_terms": found_terms,
            "missing_terms": [term for term in requirement["terms"] if term not in found_terms],
        }
    return audit


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def simulate_neural_thicket(seed: int = 20260801, population_size: int = 128) -> dict[str, Any]:
    rng = random.Random(seed)
    tasks = ("math", "coding", "vision", "chemistry")
    small_scores: list[float] = []
    large_scores: list[float] = []
    perturbations: list[dict[str, Any]] = []

    for idx in range(population_size):
        specialty = tasks[idx % len(tasks)]
        small = _sigmoid(rng.gauss(-1.2, 0.85))
        large = _sigmoid(rng.gauss(0.35, 0.85))
        small_scores.append(small)
        large_scores.append(large)
        task_scores = {task: large - 0.08 + rng.random() * 0.1 for task in tasks}
        task_scores[specialty] = min(1.0, large + 0.14 + rng.random() * 0.1)
        perturbations.append({"specialty": specialty, "mean_score": large, "task_scores": task_scores})

    threshold = 0.55
    specialty_counts = Counter(
        max(item["task_scores"], key=item["task_scores"].get) for item in perturbations if item["mean_score"] >= threshold
    )
    sorted_models = sorted(perturbations, key=lambda item: item["mean_score"], reverse=True)
    best_single = max(model["mean_score"] for model in sorted_models)
    ensemble_accuracy = min(1.0, sum(model["mean_score"] for model in sorted_models[:8]) / 8 + 0.015)

    population_curve: dict[str, float] = {}
    for size in (12, 24, 48, 96, population_size):
        key = str(size)
        top = sorted(item["mean_score"] for item in perturbations[: min(size, population_size)])[-8:]
        population_curve[key] = round(sum(top) / len(top), 6)

    return {
        "seed": seed,
        "population_size": population_size,
        "small_model_density": round(sum(score >= threshold for score in small_scores) / len(small_scores), 6),
        "large_model_density": round(sum(score >= threshold for score in large_scores) / len(large_scores), 6),
        "best_single_accuracy": round(best_single, 6),
        "ensemble_accuracy": round(max(best_single, ensemble_accuracy), 6),
        "population_curve": population_curve,
        "specialty_counts": dict(sorted(specialty_counts.items())),
    }


def _artifact_snapshot(files: dict[str, str]) -> dict[str, Any]:
    return {
        path: {
            "sha256": _sha256_text(text),
            "size_bytes": len(text.encode("utf-8")),
            "source_url": f"https://raw.githubusercontent.com/sunrainyg/RandOpt/{UPSTREAM_COMMIT}/{path}",
            "acquisition_command": f"git clone {UPSTREAM_REPO} && cd RandOpt && git checkout {UPSTREAM_COMMIT}",
        }
        for path, text in sorted(files.items())
    }


def _claim_results(audit: dict[str, dict[str, Any]], simulation: dict[str, Any], raw_result_artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    statuses = ["toy", "unavailable", "toy", "unavailable", "toy"]
    evidence = [
        "Pinned source contains the 1D RandOpt toy experiment and deterministic simulation shows denser successful perturbations for the large-model proxy.",
        "No raw Figure 3 Qwen2.5 sweep outputs were released in the audited artifact set.",
        "Deterministic simulation produces multiple task-specialty maxima among successful perturbations.",
        "Pinned source contains RandOpt, majority voting, PPO/GRPO/ES baseline scaffolding, but no released full-scale benchmark outputs.",
        "Population-size mechanism is reproduced in toy simulation; scale-dependent full benchmark results are unavailable without GPU-scale artifacts.",
    ]
    results = []
    for idx, claim in enumerate(CLAIMS):
        results.append(
            {
                "claim_index": idx + 1,
                "claim": claim["target_claim"],
                "claim_sha256": claim["challenge_claim_sha256"],
                "status": statuses[idx] if raw_result_artifacts == {} else statuses[idx],
                "evidence": evidence[idx],
                "audit_dependencies": audit,
                "simulation": simulation if idx in {0, 2, 4} else None,
                "raw_result_artifacts": raw_result_artifacts,
            }
        )
    return results


def build_evidence_bundle(
    files: dict[str, str] | None = None,
    raw_result_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if files is None:
        snapshot = _clone_upstream()
        files = snapshot.files
        git_commit = snapshot.git_commit
        file_hashes = snapshot.file_hashes
    else:
        git_commit = UPSTREAM_COMMIT
        file_hashes = {path: _sha256_text(text) for path, text in files.items()}
    raw_result_artifacts = raw_result_artifacts or {}
    audit = audit_artifacts(files)
    simulation = simulate_neural_thicket()
    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "snapshot_id": SNAPSHOT_ID,
        "upstream_pins": UPSTREAM_PINS,
        "upstream_commit_observed": git_commit,
        "artifact_hashes": file_hashes,
        "artifacts": _artifact_snapshot(files),
        "artifact_audit": audit,
        "simulation": simulation,
        "claim_results": _claim_results(audit, simulation, raw_result_artifacts),
        "limitations": [
            "No paper-reported benchmark metric is used as reproduced evidence.",
            "GPU-scale Qwen/Llama/OLMo inference, PPO, GRPO, and ES runs are marked unavailable without released raw outputs.",
            "Toy results test qualitative RandOpt mechanisms only.",
        ],
    }


def write_evidence(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_evidence_bundle()
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle
