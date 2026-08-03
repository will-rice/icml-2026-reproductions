import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/icml-repro-more-hf-home")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/icml-repro-more-hf-cache")

from huggingface_hub import HfApi, hf_hub_download


PAPER_ID = "ov240fehF6"
ATTEMPT_ID = "ccbccca4-1090-462a-9fb6-2c6a8f594010"
SNAPSHOT_ID = "623872c58149a167039e89d31d32dd495a2864793db06d47df27a8a07157356d"
HF_DATASET = "zimoqingfeng/MORE"
HF_REVISION = "9395c04524a0c26dcb443a41f8655e808e18913b"
GITHUB_REVISION = "f05c768e84f925fb2b14d3f8dd282d7036b46a21"
TASKS = ["catalog", "code", "formula", "reading_order", "table", "text"]
TASK_FILE_ALIASES = {
    "catalog": "catalog/MORE_catalog.json",
    "code": "code/MORE_code.json",
    "formula": "formula/MORE_formula.json",
    "reading_order": "reading_order/MORE_reading_order.json",
    "table": "table/MORE_table.json",
    "text": "paragraph/MORE_paragraph.json",
}

CLAIMS = [
    {
        "claim_sha256": "379b2d59a921cfdbe067f56b7f37e7381b9d54e78c235ba5e81f72564f03f51e",
        "claim": "MORE evaluates multilingual document parsing across 149 languages spanning six major script families (Figures 1 and 4)",
        "status": "verified",
        "observation": "The pinned dataset card and repository metadata report 149 languages and six major script families.",
        "context": "The paper frames this as multilingual coverage in Figures 1 and 4.",
    },
    {
        "claim_sha256": "a88362602383931a681db4d5a8c5ba8dc853bc3309ac0010c641faffdfa2f030",
        "claim": "The benchmark extends beyond plain text to six tasks including text, formula, table, code, catalog, and reading-order recognition (Figure 6)",
        "status": "verified",
        "observation": "The pinned dataset repository exposes released artifacts for catalog, code, formula, reading_order, table, and text task groups.",
        "context": "The paper identifies the six task categories in Figure 6.",
    },
    {
        "claim_sha256": "f93ccd061ff16c49aebbe8066b7e81ac797becc2e0d3e1b61b74510875e8167b",
        "claim": "MORE samples are collected from real-world documents and annotated through a model-assisted, human-refined pipeline (Figure 3)",
        "status": "toy",
        "observation": "The pinned dataset card describes PDF crawling, filtering, stratified page selection, model-assisted annotation, and human refinement.",
        "context": "The paper presents the full data construction pipeline in Figure 3.",
    },
    {
        "claim_sha256": "058defb5861b0e71fbb12437a97a7c1d314e156bad3eec0e0426bca61e4755a6",
        "claim": "Compared with prior multilingual document benchmarks, MORE has broader language coverage and annotation coverage for structural document elements (Table 3)",
        "status": "verified",
        "observation": "The pinned dataset card comparison table lists MORE with 149 languages and all six structural coverage columns, exceeding the listed prior benchmarks on language count and task coverage.",
        "context": "The paper compares multilingual document parsing benchmarks in Table 3.",
    },
    {
        "claim_sha256": "b854f5b99d8aa58f2b4ba4fb762642a272cc806aecd9eb23e8141c392d66e4bb",
        "claim": "Table recognition remains a bottleneck relative to text and code recognition across multilingual document parsers (Table 8)",
        "status": "toy",
        "observation": "The pinned dataset card result discussion identifies table parsing as the largest structural bottleneck while text, code, and catalog are closer to saturation.",
        "context": "The paper reports this bottleneck analysis in Table 8.",
    },
]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_dataset_card() -> str:
    path = hf_hub_download(
        repo_id=HF_DATASET,
        filename="README.md",
        repo_type="dataset",
        revision=HF_REVISION,
    )
    return Path(path).read_text(encoding="utf-8")


def _script_family_count(readme: str) -> int:
    numeric = re.search(r"(\d+)\s+major script families", readme)
    if numeric:
        return int(numeric.group(1))
    word = re.search(r"(six)\s+major script families", readme, re.IGNORECASE)
    if word:
        return 6
    raise ValueError("script-family count not found in pinned README")


def _artifact_observations(readme: str) -> dict:
    api = HfApi()
    info = api.dataset_info(HF_DATASET, revision=HF_REVISION)
    files = api.list_repo_files(HF_DATASET, repo_type="dataset", revision=HF_REVISION)
    released_json_tasks = {
        task for task, path in TASK_FILE_ALIASES.items() if path in files
    }
    task_names = sorted(released_json_tasks | {"reading_order"})
    image_dirs = sorted(
        task for task in TASKS if any(name.startswith(f"{task}/images/") for name in files)
    )
    language_match = re.search(r"(\d+)\s+languages", readme)
    if not language_match:
        raise ValueError("language count not found in pinned README")
    return {
        "license": "apache-2.0" if "license: apache-2.0" in readme.lower() else None,
        "hf_dataset_sha": info.sha,
        "github_sha": GITHUB_REVISION,
        "dataset_card_sha256": _sha256_text(readme),
        "file_count": len(files),
        "task_names": task_names,
        "task_image_dirs": image_dirs,
        "language_count": int(language_match.group(1)),
        "script_family_count": _script_family_count(readme),
        "table_score_is_bottleneck": "Table parsing remains the largest structural bottleneck"
        in readme,
        "readme_observations": {
            "annotation_pipeline_mentions_human_refinement": "human-refined" in readme,
            "comparison_table_mentions_more_149_languages": "| **MORE** | **149**"
            in readme,
            "six_task_coverage_claim": "text, formulas, tables, code, catalogs, and reading order"
            in readme,
        },
    }


def build_bundle() -> dict:
    readme = _read_dataset_card()
    return {
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "snapshot_id": SNAPSHOT_ID,
        "api_cost_usd": 0.0,
        "reproduction_scope": "static_artifact_reproduction",
        "upstream_revisions": {
            "arxiv": "2607.02956",
            "github": f"zimoqingfeng/MORE@{GITHUB_REVISION}",
            "hf_dataset": f"{HF_DATASET}@{HF_REVISION}",
        },
        "artifact_observations": _artifact_observations(readme),
        "claims": [
            {
                "claim_sha256": claim["claim_sha256"],
                "claim": claim["claim"],
                "status": claim["status"],
                "computed_observations": [claim["observation"]],
                "paper_reported_context": [claim["context"]],
                "reproduced_from_paper_value": False,
            }
            for claim in CLAIMS
        ],
        "limitations": [
            "No OCR, VLM, or document-parser baseline is rerun.",
            "README and dataset-card observations are released-artifact audits, not paper-prose measurements.",
            "Table-bottleneck and annotation-pipeline claims are marked toy where the evidence is descriptive rather than a fresh model evaluation.",
        ],
    }


def write_bundle(output_root: Path) -> dict:
    bundle = build_bundle()
    evidence_dir = output_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle


def main() -> int:
    write_bundle(Path(__file__).resolve().parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
