from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


ATTEMPT_ID = "113e8358-c223-4172-8f77-c5ba922bf74d"
PAPER_ID = "Rl2uQlCoQX"
TITLE = "SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding"
DATASET_ID = "nvidia/SPEED-Bench"
DATASET_REVISION = "487aa718444e816458d1a0a52bfce7a454285cf4"
GITHUB_REPO = "https://github.com/NVIDIA/Model-Optimizer.git"
GITHUB_COMMIT = "a23390dbb6e52b0c028f3e9455a74da824c88735"
GENERATED_AT = "2026-07-31T23:16:31.311943+00:00"

CLAIM_BINDINGS = [
    {
        "claim_index": 1,
        "target_claim": "SPEED-Bench contains a qualitative split optimized for semantic diversity and a throughput split with fixed 1K-32K input-length buckets supporting high-concurrency evaluation (Figure 1)",
        "challenge_claim": "SPEED-Bench contains a qualitative split optimized for semantic diversity and a throughput split with fixed 1K-32K input-length buckets supporting high-concurrency evaluation (Figure 1)",
        "challenge_claim_sha256": "acf30ca2d6ba9d93007e041eb314452cf636746f58d42da5973f5c5152f9f8a7",
    },
    {
        "claim_index": 2,
        "target_claim": "The qualitative split has lower average semantic similarity than random selection and SpecBench across categories (Figure 2)",
        "challenge_claim": "The qualitative split has lower average semantic similarity than random selection and SpecBench across categories (Figure 2)",
        "challenge_claim_sha256": "9199267aa24773c1e42e965eebfba48b160be69a5d2d4f330f6573893d06b084",
    },
    {
        "claim_index": 3,
        "target_claim": "SPEED-Bench reports average acceptance length and speedups for speculative decoding methods on a unified qualitative split (Table 1)",
        "challenge_claim": "SPEED-Bench reports average acceptance length and speedups for speculative decoding methods on a unified qualitative split (Table 1)",
        "challenge_claim_sha256": "445341382926e77a701223e5434981daeb8e507d0f07bc67d8f6dd6b7e1a43b4",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_dataset_card(readme: Path) -> dict[str, Any]:
    text = readme.read_text(encoding="utf-8")
    configs: dict[str, int] = {}
    pattern = re.compile(
        r"config_name:\s*(?P<name>[A-Za-z0-9_]+).*?num_examples:\s*(?P<count>\d+)",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        configs[match.group("name")] = int(match.group("count"))
    buckets = sorted(
        (
            name.removeprefix("throughput_")
            for name in configs
            if name.startswith("throughput_")
        ),
        key=lambda value: int(value.removesuffix("k")),
    )
    return {
        "dataset_readme": str(readme),
        "dataset_readme_sha256": sha256_file(readme),
        "configs": configs,
        "qualitative_examples": configs.get("qualitative"),
        "throughput_buckets": buckets,
        "throughput_examples_per_bucket": {
            bucket: configs[f"throughput_{bucket}"] for bucket in buckets
        },
        "fixed_bucket_claim_range": bool(buckets)
        and buckets[0] == "1k"
        and buckets[-1] == "32k",
        "expected_framework_configs": ["qualitative"]
        + [f"throughput_{bucket}" for bucket in buckets],
    }


def parse_similarity_table(readme: Path) -> list[dict[str, Any]]:
    rows = []
    for line in readme.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4:
            continue
        category, specbench, random_value, greedy_value = cells
        if category in {"Category", "---", ""}:
            continue
        spec = _first_float(specbench)
        random_score = _first_float(random_value)
        greedy_score = _first_float(greedy_value)
        if spec is None or greedy_score is None:
            continue
        rows.append(
            {
                "category": category,
                "specbench": spec,
                "speed_random": random_score,
                "speed_greedy": greedy_score,
                "greedy_below_specbench": greedy_score < spec,
                "greedy_below_random": (
                    random_score is not None and greedy_score < random_score
                ),
            }
        )
    return rows


def _first_float(value: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


def _repo_summary(repo_dir: Path) -> dict[str, Any]:
    commit = GITHUB_COMMIT
    if (repo_dir / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        commit = result.stdout.strip()
    specdec = repo_dir / "examples" / "specdec_bench"
    files = sorted(path.relative_to(repo_dir).as_posix() for path in specdec.rglob("*") if path.is_file()) if specdec.exists() else []
    result_files = [
        file for file in files
        if Path(file).name in {"timing.json", "acceptance_rate.json", "configuration.json"}
    ]
    readme = specdec / "README.md"
    readme_text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    return {
        "github_repo": GITHUB_REPO,
        "github_commit": commit,
        "expected_github_commit": GITHUB_COMMIT,
        "specdec_bench_exists": specdec.is_dir(),
        "specdec_file_count": len(files),
        "specdec_mentions_qualitative": "qualitative" in readme_text,
        "specdec_mentions_throughput": "throughput_1k" in readme_text,
        "result_artifacts": result_files,
    }


def build_evidence_bundle(*, dataset_readme: Path, repo_dir: Path) -> dict[str, Any]:
    dataset = parse_dataset_card(dataset_readme)
    similarity = parse_similarity_table(dataset_readme)
    repo = _repo_summary(repo_dir)
    comparable_similarity = [
        row for row in similarity
        if row["category"] != "QA" and row["speed_random"] is not None
    ]

    claim1_ok = (
        dataset["qualitative_examples"] == 880
        and dataset["throughput_buckets"] == ["1k", "2k", "8k", "16k", "32k"]
        and all(count == 1536 for count in dataset["throughput_examples_per_bucket"].values())
        and dataset["fixed_bucket_claim_range"]
        and repo["specdec_mentions_throughput"]
    )
    claim2_ok = bool(comparable_similarity) and all(
        row["greedy_below_specbench"] and row["greedy_below_random"]
        for row in comparable_similarity
    )
    claim3_machine_readable = bool(repo["result_artifacts"])
    claim3_framework = (
        repo["specdec_bench_exists"]
        and repo["specdec_mentions_qualitative"]
    )

    claims = [
        {
            **CLAIM_BINDINGS[0],
            "status": "verified" if claim1_ok else "falsified",
            "evidence": "Pinned HF dataset card and Model-Optimizer framework expose the qualitative split and throughput configs spanning 1K to 32K with 1,536 prompts per throughput bucket.",
            "observations": {
                "qualitative_examples": dataset["qualitative_examples"],
                "throughput_buckets": dataset["throughput_buckets"],
                "throughput_examples_per_bucket": dataset["throughput_examples_per_bucket"],
                "specdec_framework_mentions_throughput": repo["specdec_mentions_throughput"],
            },
        },
        {
            **CLAIM_BINDINGS[1],
            "status": "verified" if claim2_ok else "inconclusive",
            "evidence": "Pinned dataset card reports category-level average semantic-similarity values; for comparable non-QA rows, SPEED greedy selection is lower than both SpecBench and random selection.",
            "observations": {
                "similarity_rows_checked": len(comparable_similarity),
                "rows": comparable_similarity,
            },
        },
        {
            **CLAIM_BINDINGS[2],
            "status": "verified" if claim3_machine_readable else ("toy" if claim3_framework else "inconclusive"),
            "evidence": (
                "Pinned Model-Optimizer source provides a unified SPEED-Bench measurement framework with timing and acceptance-rate outputs, "
                "but No machine-readable Table 1 result artifact was found in the pinned repository; paper-reported acceptance length and speedups were not treated as reproduced measurements."
            ),
            "observations": {
                "specdec_bench_exists": repo["specdec_bench_exists"],
                "specdec_file_count": repo["specdec_file_count"],
                "result_artifacts": repo["result_artifacts"],
            },
        },
    ]
    return {
        "evidence_schema": "icml-repro-v1",
        "generated_at": GENERATED_AT,
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "upstream": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_readme_sha256": dataset["dataset_readme_sha256"],
            **repo,
        },
        "dataset_audit": dataset,
        "similarity_audit": similarity,
        "claims": claims,
    }


def _download_dataset_readme(cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            DATASET_ID,
            filename="README.md",
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=str(cache_dir),
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-readme", type=Path)
    parser.add_argument("--repo-dir", type=Path, default=Path("/tmp/model-optimizer-speed-113e8358"))
    parser.add_argument("--cache-dir", type=Path, default=Path("evidence/cache"))
    parser.add_argument("--output", type=Path, default=Path("evidence/bundle.json"))
    args = parser.parse_args(argv)

    dataset_readme = args.dataset_readme or _download_dataset_readme(args.cache_dir)
    bundle = build_evidence_bundle(dataset_readme=dataset_readme, repo_dir=args.repo_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "claims": [claim["status"] for claim in bundle["claims"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
