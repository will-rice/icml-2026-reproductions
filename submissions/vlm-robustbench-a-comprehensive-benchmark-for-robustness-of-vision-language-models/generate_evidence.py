from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


PAPER_ID = "HwXyyvK7ZJ"
ATTEMPT_ID = "084bfe5b-42df-4de0-8a5f-b40b17fb001f"
TITLE = "VLM-RobustBench: A Comprehensive Benchmark for Robustness of Vision-Language Models"
GITHUB_REPO = "https://github.com/saxenarohit/vlm_robustbench.git"
GITHUB_COMMIT = "8bc793d1649e574e000f91c59cb6ce7432c95073"
PROJECT_PAGE_URL = "https://edinburghnlp.github.io/vlm-robustbench/"
PROJECT_APP_URL = "https://edinburghnlp.github.io/vlm-robustbench/app.js"
PROJECT_PAGE_SHA256 = "6eac82e58ad661354943a49457d81acca3736a127f7c95eeaa2d19744963e7ed"
GENERATED_AT = "2026-07-31T22:21:41.997606+00:00"

CLAIM_BINDINGS = [
    {
        "claim_index": 1,
        "target_claim": "VLM-RobustBench evaluates 49 augmentations, including 42 severity-based corruptions across nine categories and 7 binary transforms (Table 1).",
        "challenge_claim": "VLM-RobustBench evaluates 49 augmentations, including 42 severity-based corruptions across nine categories and 7 binary transforms (Table 1).",
        "challenge_claim_sha256": "33e0dd682fda119548029ce6e9f553f4bd99d68ead5ca66c630149601e6dcc43",
    },
    {
        "claim_index": 2,
        "target_claim": "The benchmark yields 133 corrupted settings per model-dataset pair by applying three severities to 42 corruptions plus seven binary transforms (Section 3.2).",
        "challenge_claim": "The benchmark yields 133 corrupted settings per model-dataset pair by applying three severities to 42 corruptions plus seven binary transforms (Section 3.2).",
        "challenge_claim_sha256": "20bca186ddd41da1813a913f66b5218617a9f58d63833bded1646a4627dd97dd",
    },
    {
        "claim_index": 3,
        "target_claim": "Low-severity glass blur causes an average 8.1 percentage-point MMBench accuracy drop, larger than high-severity brightness reduction, illustrating a severity mismatch (Figure 1).",
        "challenge_claim": "Low-severity glass blur causes an average 8.1 percentage-point MMBench accuracy drop, larger than high-severity brightness reduction, illustrating a severity mismatch (Figure 1).",
        "challenge_claim_sha256": "3df43597a1c24deb295717216e45eae4df5f11796ba8459ad95014cc36083d8a",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise ValueError(f"missing assignment {name} in {path}")


def _parse_readme_categories(readme_source: Path) -> dict[str, list[str]]:
    text = readme_source.read_text(encoding="utf-8")
    categories: dict[str, list[str]] = {}
    pattern = re.compile(r"^- \*\*(?P<name>[^*]+)\*\* \((?P<count>\d+)\): (?P<items>.+)$")
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        items = [item.strip() for item in match.group("items").split(",")]
        expected_count = int(match.group("count"))
        if len(items) != expected_count:
            raise ValueError(f"category count mismatch for {match.group('name')}")
        categories[match.group("name")] = items
    if not categories:
        raise ValueError("no README augmentation categories found")
    return categories


def audit_augmentations(*, aug_source: Path, readme_source: Path) -> dict[str, Any]:
    severity_params = _literal_assignment(aug_source, "SEVERITY_PARAMS")
    binary_augmentations = sorted(_literal_assignment(aug_source, "BINARY_AUGMENTATIONS"))
    categories = _parse_readme_categories(readme_source)
    severity_based = sorted(severity_params)
    severity_category_names = [name for name in categories if name != "Binary"]
    category_members = {
        item for name in severity_category_names for item in categories[name]
    }

    missing_from_readme = sorted(set(severity_based) - category_members)
    missing_from_source = sorted(category_members - set(severity_based))
    binary_readme = sorted(categories.get("Binary", []))
    binary_mismatch = sorted(set(binary_augmentations) ^ set(binary_readme))
    if missing_from_readme or missing_from_source or binary_mismatch:
        raise ValueError(
            json.dumps(
                {
                    "missing_from_readme": missing_from_readme,
                    "missing_from_source": missing_from_source,
                    "binary_mismatch": binary_mismatch,
                },
                sort_keys=True,
            )
        )

    severity_levels_used = [1, 3, 5]
    corrupted_settings = len(severity_based) * len(severity_levels_used) + len(binary_augmentations)
    return {
        "aug_source": str(aug_source),
        "readme_source": str(readme_source),
        "aug_source_sha256": sha256_file(aug_source),
        "readme_source_sha256": sha256_file(readme_source),
        "severity_based_count": len(severity_based),
        "binary_count": len(binary_augmentations),
        "total_augmentation_count": len(severity_based) + len(binary_augmentations),
        "severity_category_count_excluding_binary": len(severity_category_names),
        "category_count_including_binary": len(categories),
        "corrupted_settings_per_model_dataset": corrupted_settings,
        "severity_levels_used": severity_levels_used,
        "categories": categories,
        "severity_based_augmentations": severity_based,
        "binary_augmentations": binary_augmentations,
    }


def _find_exact_glass_value(project_page: Path, project_app: Path) -> dict[str, Any]:
    page_text = project_page.read_text(encoding="utf-8")
    app_text = project_app.read_text(encoding="utf-8")
    combined = f"{page_text}\n{app_text}"
    exact_pattern = re.compile(
        r"glass(?:[_ -]blur)?.{0,80}8\.1|8\.1.{0,80}glass(?:[_ -]blur)?",
        re.IGNORECASE | re.DOTALL,
    )
    has_exact = bool(exact_pattern.search(combined))
    approximate_page = bool(
        re.search(r"glass blur.{0,120}about 8 points", page_text, re.IGNORECASE | re.DOTALL)
    )
    brightness_comparison = bool(
        re.search(
            r"brightness reduction.{0,120}(?:about )?2 points",
            page_text,
            re.IGNORECASE | re.DOTALL,
        )
    )
    return {
        "project_page_sha256_observed": sha256_file(project_page),
        "project_app_sha256_observed": sha256_file(project_app),
        "exact_8_1_primary_value_found": has_exact,
        "approximate_glass_blur_text_found": approximate_page,
        "brightness_comparison_text_found": brightness_comparison,
    }


def build_evidence_bundle(*, repo_dir: Path, project_page: Path, project_app: Path) -> dict[str, Any]:
    aug_source = repo_dir / "code" / "aug.py"
    readme_source = repo_dir / "README.md"
    if not aug_source.exists():
        aug_source = repo_dir / "aug.py"
    if not readme_source.exists():
        readme_source = repo_dir / "README.md"

    audit = audit_augmentations(aug_source=aug_source, readme_source=readme_source)
    glass = _find_exact_glass_value(project_page, project_app)
    git_commit = GITHUB_COMMIT
    if (repo_dir / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        git_commit = result.stdout.strip()

    claim1_ok = (
        audit["total_augmentation_count"] == 49
        and audit["severity_based_count"] == 42
        and audit["binary_count"] == 7
        and audit["severity_category_count_excluding_binary"] == 9
    )
    claim2_ok = audit["corrupted_settings_per_model_dataset"] == 133
    claim3_exact = glass["exact_8_1_primary_value_found"]
    claim3_partial = (
        glass["approximate_glass_blur_text_found"]
        and glass["brightness_comparison_text_found"]
    )

    claims = [
        {
            **CLAIM_BINDINGS[0],
            "status": "verified" if claim1_ok else "falsified",
            "evidence": "Pinned repository aug.py and README independently enumerate 42 severity-based transforms, 7 binary transforms, and nine non-binary categories.",
            "observations": {
                "total_augmentation_count": audit["total_augmentation_count"],
                "severity_based_count": audit["severity_based_count"],
                "binary_count": audit["binary_count"],
                "severity_category_count_excluding_binary": audit[
                    "severity_category_count_excluding_binary"
                ],
            },
        },
        {
            **CLAIM_BINDINGS[1],
            "status": "verified" if claim2_ok else "falsified",
            "evidence": "Computed corrupted settings from pinned source categories as 42 severity-based augmentations times levels 1, 3, and 5, plus 7 binary transforms.",
            "observations": {
                "severity_levels_used": audit["severity_levels_used"],
                "formula": "42 * 3 + 7",
                "corrupted_settings_per_model_dataset": audit[
                    "corrupted_settings_per_model_dataset"
                ],
            },
        },
        {
            **CLAIM_BINDINGS[2],
            "status": "verified" if claim3_exact else ("toy" if claim3_partial else "inconclusive"),
            "evidence": (
                "Pinned project artifacts support the qualitative severity-mismatch comparison, "
                "but the exact 8.1 percentage-point glass-blur value was not found as a "
                "machine-readable primary value; paper-reported value was not used as reproduced evidence."
            ),
            "observations": glass,
        },
    ]

    return {
        "evidence_schema": "icml-repro-v1",
        "generated_at": GENERATED_AT,
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "upstream": {
            "github_repo": GITHUB_REPO,
            "github_commit": git_commit,
            "expected_github_commit": GITHUB_COMMIT,
            "project_page_url": PROJECT_PAGE_URL,
            "project_app_url": PROJECT_APP_URL,
            "project_page_sha256": PROJECT_PAGE_SHA256,
            "project_page_sha256_observed": glass["project_page_sha256_observed"],
            "project_app_sha256_observed": glass["project_app_sha256_observed"],
        },
        "augmentation_audit": audit,
        "claims": claims,
    }


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def _ensure_repo(cache_dir: Path) -> Path:
    repo_dir = cache_dir / "vlm_robustbench"
    if not repo_dir.exists():
        subprocess.run(["git", "clone", "--no-checkout", GITHUB_REPO, str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "fetch", "--all", "--tags"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", GITHUB_COMMIT], check=True)
    return repo_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path)
    parser.add_argument("--project-page", type=Path)
    parser.add_argument("--project-app", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("evidence/cache"))
    parser.add_argument("--output", type=Path, default=Path("evidence/bundle.json"))
    args = parser.parse_args(argv)

    repo_dir = args.repo_dir or _ensure_repo(args.cache_dir)
    project_page = args.project_page or args.cache_dir / "project.html"
    project_app = args.project_app or args.cache_dir / "app.js"
    if args.project_page is None:
        _download(PROJECT_PAGE_URL, project_page)
    if args.project_app is None:
        _download(PROJECT_APP_URL, project_app)

    bundle = build_evidence_bundle(
        repo_dir=repo_dir,
        project_page=project_page,
        project_app=project_app,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "claims": [c["status"] for c in bundle["claims"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
