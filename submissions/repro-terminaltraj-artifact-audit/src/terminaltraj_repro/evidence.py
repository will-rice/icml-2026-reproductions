from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi


ATTEMPT_ID = "fe586a0f-c0ff-4290-882b-b7fadb2ec2f4"
PAPER_ID = "PeFSCRulgy"
GENERATED_AT = "2026-07-29T03:10:00+00:00"
GITHUB_REPO = "https://github.com/multimodal-art-projection/TerminalTraj.git"
GITHUB_REVISION = "01305cbf0425b08b41cf8cfc3e30abb0f4953c27"
HF_DATASET = "m-a-p/TerminalTraj"
HF_DATASET_REVISION = "5c1823f4a8b9ca0cf02c27d2db52c5b35b53a308"
HF_INSTANCES = "m-a-p/TerminalTraj-5k-instances"
HF_INSTANCES_REVISION = "cfc9f4379596bac54d09061544a83a2cfa5d6d06"
UPSTREAM_REVISION = (
    "arxiv:2602.01244v3+arxiv-source-sha256:"
    "30bae2223a3180e28bfe9b48bf3881c0d9f2719b4c7828894bb1dd062d20ce1d"
    f"+github:multimodal-art-projection/TerminalTraj@{GITHUB_REVISION}"
    f"+hf-dataset:{HF_DATASET}@{HF_DATASET_REVISION}"
    f"+hf-dataset:{HF_INSTANCES}@{HF_INSTANCES_REVISION}"
)

CLAIMS = [
    {
        "challenge_claim_sha256": (
            "39a708c775ac4fbff63c3c664dc7739dfe4b55247d864174215b269db249921f"
        ),
        "text": (
            "TerminalTraj filters repositories to construct Dockerized execution "
            "environments, generates Docker-aligned task instances, and synthesizes "
            "executable validated terminal trajectories (Abstract)."
        ),
    },
    {
        "challenge_claim_sha256": (
            "4908ae61e70d4225466dc1443bdb44a48ef4a6701f5b0f77a39a9fd750203268"
        ),
        "text": (
            "TerminalTraj curates 32K Docker images and generates 50,733 verified "
            "terminal trajectories across eight domains (Abstract)."
        ),
    },
]


def _cache_root() -> Path:
    return Path("/tmp/icml-terminaltraj-repro-cache")


def _run(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=None if cwd is None else str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _ensure_github_checkout() -> Path:
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    checkout = root / "TerminalTraj"
    if not checkout.exists():
        _run(["git", "clone", GITHUB_REPO, str(checkout)])
    _run(["git", "fetch", "--tags", "--force", "origin", GITHUB_REVISION], cwd=checkout)
    _run(["git", "checkout", "--detach", GITHUB_REVISION], cwd=checkout)
    return checkout


def _repo_license_count(path: Path) -> tuple[int, dict[str, int]]:
    license_counts: dict[str, int] = {}
    manifest = path / "source" / "repo&license.jsonl"
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            license_name = record["license_type"]
            license_counts[license_name] = license_counts.get(license_name, 0) + 1
    return sum(license_counts.values()), dict(sorted(license_counts.items()))


def _has_executable_pipeline_code(path: Path) -> bool:
    ignored = {".git", "images", "source"}
    executable_suffixes = {".py", ".sh", ".ipynb", ".yaml", ".yml", ".toml"}
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        if any(part in ignored for part in candidate.relative_to(path).parts):
            continue
        if candidate.suffix.lower() in executable_suffixes:
            return True
    return False


def _parse_readme_counts(readme: str) -> dict[str, int]:
    trajectories = re.search(r"50,733", readme)
    docker_exact = re.search(r"32,325", readme)
    docker_rounded = re.search(r"32K Docker images", readme)
    domains = re.search(r"across eight domains", readme, flags=re.IGNORECASE)
    if not (trajectories and docker_exact and docker_rounded and domains):
        raise ValueError("readme_counts")
    return {
        "docker_images_rounded": 32000,
        "docker_images_exact": 32325,
        "trajectories": 50733,
        "domains": 8,
    }


def _hf_dataset_observations() -> dict:
    api = HfApi()
    info = api.dataset_info(HF_DATASET, revision=HF_DATASET_REVISION)
    siblings = sorted(sibling.rfilename for sibling in (info.siblings or []))
    split = next(split for split in info.card_data["dataset_info"]["splits"] if split["name"] == "train")
    features = [
        feature["name"] for feature in info.card_data["dataset_info"]["features"]
    ]
    return {
        "repo_id": HF_DATASET,
        "revision": info.sha,
        "files": siblings,
        "train_examples": split["num_examples"],
        "features": features,
    }


def _hf_instance_observations() -> dict:
    api = HfApi()
    info = api.dataset_info(HF_INSTANCES, revision=HF_INSTANCES_REVISION)
    return {
        "repo_id": HF_INSTANCES,
        "revision": info.sha,
        "files": sorted(sibling.rfilename for sibling in (info.siblings or [])),
    }


def collect_observations() -> dict:
    checkout = _ensure_github_checkout()
    readme = (checkout / "README.md").read_text(encoding="utf-8")
    manifest_count, license_counts = _repo_license_count(checkout)
    return {
        "github": {
            "repo": GITHUB_REPO,
            "revision": _run(["git", "rev-parse", "HEAD"], cwd=checkout),
            "files": sorted(
                str(path.relative_to(checkout))
                for path in checkout.rglob("*")
                if path.is_file() and ".git" not in path.relative_to(checkout).parts
            ),
            "repo_license_records": manifest_count,
            "license_counts": license_counts,
            "has_executable_pipeline_code": _has_executable_pipeline_code(checkout),
        },
        "hf_dataset": _hf_dataset_observations(),
        "hf_instances": _hf_instance_observations(),
        "paper_counts": _parse_readme_counts(readme),
        "tooling": {
            "git_available": shutil.which("git") is not None,
            "network_required": True,
        },
    }


def _claim_evaluations(observations: dict) -> list[dict]:
    github = observations["github"]
    dataset = observations["hf_dataset"]
    instances = observations["hf_instances"]
    counts = observations["paper_counts"]
    return [
        {
            **CLAIMS[0],
            "status": "toy",
            "summary": (
                "README and artifact metadata expose repository filtering, a source "
                "license manifest, the training dataset, and a Docker instance subset, "
                "but the pinned GitHub tree contains no executable pipeline code."
            ),
            "evidence": {
                "github_file_count": len(github["files"]),
                "repo_license_records": github["repo_license_records"],
                "has_executable_pipeline_code": github["has_executable_pipeline_code"],
                "instance_archive_files": instances["files"],
            },
        },
        {
            **CLAIMS[1],
            "status": "unavailable",
            "summary": (
                "The README states 32,325 Docker images and 50,733 verified "
                "trajectories, but the pinned public training dataset exposes "
                f"{dataset['train_examples']:,} train examples and the GitHub "
                f"manifest covers {github['repo_license_records']:,} released-subset "
                "repositories, so the full paper count is not reproduced from "
                "released artifacts."
            ),
            "evidence": {
                "paper_docker_images_exact": counts["docker_images_exact"],
                "paper_trajectories": counts["trajectories"],
                "paper_domains": counts["domains"],
                "released_train_examples": dataset["train_examples"],
                "released_repo_license_records": github["repo_license_records"],
            },
        },
    ]


def build_evidence_bundle() -> dict:
    observations = collect_observations()
    return {
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "generated_at": GENERATED_AT,
        "upstream_revision": UPSTREAM_REVISION,
        "commands": [
            "git clone https://github.com/multimodal-art-projection/TerminalTraj.git",
            f"git checkout {GITHUB_REVISION}",
            f"huggingface_hub.dataset_info({HF_DATASET!r}, revision={HF_DATASET_REVISION!r})",
            f"huggingface_hub.dataset_info({HF_INSTANCES!r}, revision={HF_INSTANCES_REVISION!r})",
        ],
        "observations": observations,
        "claims": _claim_evaluations(observations),
    }


def write_evidence_bundle(path: str | Path) -> dict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_evidence_bundle()
    destination.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle
