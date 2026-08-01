from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PAPER_ID = "Mkal0hTCnh"
ATTEMPT_ID = "9d2896b2-1cf9-41ad-ba71-0cf34fceb55e"
SNAPSHOT_ID = "bdb659717fc36b718f037c86e25b07c732c5efdb270b366a2fb8ac135476a0e6"
GENERATED_AT = "2026-08-01T13:25:00+00:00"

CODE_REPO = "https://github.com/ernie-research/MEnvAgent.git"
CODE_REVISION = "d9e63881f7c4a4670bb536c89add24573459bbee"
ARXIV_SOURCE_SHA256 = "1cd2573993bd41a200a225da3f17af2b927dc5c307844166e5b5ec4a37ddd8d0"
ARXIV_PDF_SHA256 = "ea8483bc1ab4e47fb3b0a824cf61f364e07287d5b98f0da6aafc90132cf4341f"

DATASETS = {
    "MEnvBench": {
        "repo": "ernie-research/MEnvBench",
        "revision": "4e312f11663e2ccdbd11f5cc3421de117ef4e118",
        "lfs_oid": "2f81e7edd4202eb0d3eda5598d22b64f43e2df29eda093a580ec87e58cbd93f4",
        "filename": "MEnvBench.jsonl",
    },
    "MEnvData-SWE": {
        "repo": "ernie-research/MEnvData-SWE",
        "revision": "edfaa7bf15ada849c3bd63f55a5e3ab9e85359c2",
        "lfs_oid": "e111fa1a8c4565f427d928652fddde7ce36a9d32973bb554beb60fba2c6055aa",
        "filename": "swe-images.jsonl",
    },
    "MEnvData-SWE-Trajectory": {
        "repo": "ernie-research/MEnvData-SWE-Trajectory",
        "revision": "7da0792504710d476297936765e40de3fd387097",
        "lfs_oid": "ec3d932bad7a81dcaeb29b9d53eff7042fa955c0999e23a5b32054ec5ab1cc46",
        "filename": "final_trajectories.jsonl",
    },
}


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


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def dataset_server_url(endpoint: str, **params: str) -> str:
    return "https://datasets-server.huggingface.co/" + endpoint + "?" + urllib.parse.urlencode(params)


def raw_dataset_url(dataset_name: str) -> str:
    record = DATASETS[dataset_name]
    return (
        f"https://huggingface.co/datasets/{record['repo']}/resolve/"
        f"{record['revision']}/{record['filename']}"
    )


def feature_names(dataset: str) -> list[str]:
    payload = fetch_json(
        dataset_server_url(
            "first-rows",
            dataset=dataset,
            config="default",
            split="train",
        )
    )
    return [feature["name"] for feature in payload["features"]]


def statistics_by_column(dataset: str) -> dict[str, dict[str, Any]]:
    payload = fetch_json(
        dataset_server_url(
            "statistics",
            dataset=dataset,
            config="default",
            split="train",
        )
    )
    return {item["column_name"]: item["column_statistics"] for item in payload["statistics"]}


def size_rows(dataset: str) -> int:
    payload = fetch_json(dataset_server_url("size", dataset=dataset))
    return int(payload["size"]["dataset"]["num_rows"])


def count_jsonl_dataset(dataset_name: str) -> dict[str, Any]:
    rows = 0
    repos: set[str] = set()
    languages: set[str] = set()
    fields: list[str] = []
    with urllib.request.urlopen(raw_dataset_url(dataset_name), timeout=180) as response:
        for line in response:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            repos.add(row["repo"])
            languages.add(row["language"])
            if not fields:
                fields = sorted(row)
    return {
        "num_rows": rows,
        "unique_repositories": len(repos),
        "unique_languages": len(languages),
        "schema_fields": fields,
    }


def collect_menvbench() -> dict[str, Any]:
    dataset = DATASETS["MEnvBench"]["repo"]
    stats = statistics_by_column(dataset)
    return {
        "num_rows": size_rows(dataset),
        "unique_repositories": stats["repo"]["n_unique"],
        "unique_languages": stats["language"]["n_unique"],
        "language_frequencies": stats["language"]["frequencies"],
        "schema_fields": feature_names(dataset),
    }


def collect_source_release(source_root: Path, arxiv_source: Path) -> dict[str, Any]:
    readme = (source_root / "README.md").read_text(encoding="utf-8")
    core_readme = (source_root / "menvagent" / "README.md").read_text(encoding="utf-8")
    core_note = next(
        line.strip()
        for line in core_readme.splitlines()
        if "currently being organized for public release" in line
    )
    curation_paths = [path.as_posix() for path in (source_root / "curation").rglob("*.py")]
    with tarfile.open(arxiv_source, "r:*") as archive:
        main_tex = archive.extractfile("main.tex")
        if main_tex is None:
            raise ValueError("main.tex missing from arXiv source")
        paper_source = main_tex.read().decode("utf-8", "replace")
    combined = "\n".join([readme, core_note, paper_source])
    return {
        "code_revision": run_git(["rev-parse", "HEAD"], source_root),
        "tracked_file_count": len(run_git(["ls-files"], source_root).splitlines()),
        "p_e_v_terms_present": all(term in combined for term in ["Planning", "Execution", "Verification"]),
        "environment_reuse_terms_present": "Environment Reuse" in combined and "retrieves" in combined,
        "env_patch_agent_present": "EnvPatchAgent" in combined,
        "curation_scripts_present": len(curation_paths) >= 10,
        "curation_script_count": len(curation_paths),
        "core_code_public_release_note": core_note,
    }


def build_claims(observations: dict[str, Any]) -> list[dict[str, Any]]:
    menvbench = observations["menvbench"]
    menvdata = observations["menvdata_swe"]
    trajectories = observations["menvdata_swe_trajectory"]
    source = observations["source_release"]
    return [
        {
            "claim_index": 1,
            "status": "toy",
            "summary": "The pinned source and arXiv source describe Planning-Execution-Verification and environment reuse, but the core runtime code is not fully released.",
            "evidence_basis": ["pinned_github_readme", "pinned_arxiv_source"],
            "observed": {
                "p_e_v_terms_present": source["p_e_v_terms_present"],
                "environment_reuse_terms_present": source["environment_reuse_terms_present"],
                "env_patch_agent_present": source["env_patch_agent_present"],
                "core_code_public_release_note": source["core_code_public_release_note"],
            },
        },
        {
            "claim_index": 2,
            "status": "verified"
            if menvbench["num_rows"] == 1000
            and menvbench["unique_repositories"] == 200
            and menvbench["unique_languages"] == 10
            else "falsified",
            "summary": "Released MEnvBench metadata and statistics show 1,000 rows, 200 repositories, and 10 languages.",
            "evidence_basis": ["hf_dataset_viewer_size", "hf_dataset_viewer_statistics"],
            "observed": menvbench,
        },
        {
            "claim_index": 3,
            "status": "unavailable",
            "summary": "Kimi-K2/Gemini-3-Flash versus SWE-Factory results require proprietary or paid model runs and raw logs not present in the release.",
            "evidence_basis": ["artifact_absence"],
        },
        {
            "claim_index": 4,
            "status": "unavailable",
            "summary": "EnvPatchAgent and reuse ablation metrics require benchmark reruns or raw ablation logs not present in the release.",
            "evidence_basis": ["artifact_absence"],
        },
        {
            "claim_index": 5,
            "status": "unavailable",
            "summary": "Fine-tuning claims require training/evaluation runs and checkpoints/logs that were not released as executable evidence.",
            "evidence_basis": ["artifact_absence"],
        },
        {
            "claim_index": 6,
            "status": "verified"
            if menvdata["num_rows"] == 3005
            and menvdata["unique_repositories"] == 942
            and menvdata["unique_languages"] == 10
            and trajectories["matches_claimed_trajectory_count"]
            else "falsified",
            "summary": "MEnvData-SWE matches 3,005 rows, 942 repositories, and 10 languages, but the trajectory viewer reports 3,918 rows rather than the claimed 3,872.",
            "evidence_basis": ["pinned_hf_jsonl_stream", "hf_dataset_viewer_size"],
            "observed": {
                "menvdata_swe": menvdata,
                "menvdata_swe_trajectory": trajectories,
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

    observations = {
        "menvbench": collect_menvbench(),
        "menvdata_swe": count_jsonl_dataset("MEnvData-SWE"),
        "menvdata_swe_trajectory": {
            "viewer_num_rows": size_rows(DATASETS["MEnvData-SWE-Trajectory"]["repo"]),
            "claimed_trajectory_count": 3872,
        },
        "source_release": collect_source_release(source_root, arxiv_source),
    }
    observations["menvdata_swe_trajectory"]["matches_claimed_trajectory_count"] = (
        observations["menvdata_swe_trajectory"]["viewer_num_rows"]
        == observations["menvdata_swe_trajectory"]["claimed_trajectory_count"]
    )

    bundle = {
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "challenge_snapshot_id": SNAPSHOT_ID,
        "generated_at": GENERATED_AT,
        "provenance": {
            "code": {
                "repo": CODE_REPO,
                "revision": CODE_REVISION,
            },
            "arxiv_source_sha256": sha256_file(arxiv_source),
            "arxiv_pdf_sha256": sha256_file(arxiv_pdf),
            "datasets": DATASETS,
            "license": "Apache-2.0",
        },
        "observations": observations,
        "claims": build_claims(observations),
    }
    if bundle["provenance"]["arxiv_source_sha256"] != ARXIV_SOURCE_SHA256:
        raise ValueError("arXiv source hash mismatch")
    if bundle["provenance"]["arxiv_pdf_sha256"] != ARXIV_PDF_SHA256:
        raise ValueError("arXiv PDF hash mismatch")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/tmp/menvagent-upstream-codex03"))
    parser.add_argument("--arxiv-source", type=Path, default=Path("/tmp/menvagent-2601.22859-src.tar"))
    parser.add_argument("--arxiv-pdf", type=Path, default=Path("/tmp/menvagent-2601.22859.pdf"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "evidence" / "bundle.json")
    args = parser.parse_args()
    build_bundle(args.source_root, args.arxiv_source, args.arxiv_pdf, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
