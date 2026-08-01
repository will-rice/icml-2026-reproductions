from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "evidence" / "bundle.json"

PAPER_ID = "AfqsNFzJcs"
ATTEMPT_ID = "9db24b0e-865b-43ab-b5a9-204b7f8a4843"
SNAPSHOT_ID = "1564266f7b6b9982addaa7519d20795d476826ed5088911b2974317851458640"
ARXIV_ID = "2511.12344v2"
ARXIV_SOURCE_URL = f"https://arxiv.org/e-print/{ARXIV_ID}"

CLAIMS = [
    (
        "5c0d4622dc06cf2d2dc4ccc24627cfd9e4a2776441d9039d8546733a71f25566",
        "RGR-GRPO constructs rubrics from the input question and reference answer to provide dense reward signals during GRPO training (Figure 2).",
        "toy",
        "The pinned TeX source contains the rubric construction and normalized rubric-reward equations; no LLM judge or RL training was run.",
    ),
    (
        "c8f80b119663d66a805f4e39fb1f378f72edc3e3a0e825c1f0baa574c899344f",
        "The framework uses exploration assessment to trigger off-policy rubric-guided refinement only when on-policy exploration is insufficient (Figure 2).",
        "toy",
        "The source contains Exploration Assessment, failed-criteria self-refinement, and Mix-Policy GRPO structure; a toy branch check verifies the control-flow condition.",
    ),
    (
        "3ac159e9611c46f76f06a1a093e676d0d26bfa21efd9c9f3c5a099544e0f4a1d",
        "RGR-GRPO improves over the verifiable online RL baseline by average margins of +7.0% on mathematics, +5.4% on physics, +8.4% on chemistry, and +6.6% on general reasoning benchmarks (Table 1).",
        "inconclusive",
        "Table 1 text is present, but model training and benchmark evaluation logs are not released or recomputed.",
    ),
    (
        "83e66bb936646bee611aa929a4e9008bcf874d91a6baf1388d4a60843a2c7772",
        "The method is evaluated across 14 benchmarks spanning mathematics, physics, chemistry, and general reasoning domains (Section 3).",
        "toy",
        "The Table 1 header exposes 12 benchmark columns across four domains; the broader 14-benchmark claim is not independently executed.",
    ),
    (
        "d147e319b78c360815a3378c904f5c4dad76d1a3d1071f36a25fd4e6725c5ff9",
        "Ablations show rubric categories, off-policy shaping, and exploration assessment each contribute to average performance (Table 2).",
        "inconclusive",
        "Ablation claim surfaces are present in source, but no ablation runs or logs are available for recomputation.",
    ),
    (
        "0883b7c9c50d1208e49d991d449af0fde85baf37f4a4189a84aa926d483fa756",
        "The paper proves exploration assessment acts as a necessary adaptive variance controller for stabilizing RGR-GRPO training under its analysis assumptions (Theorem C.1).",
        "inconclusive",
        "The theorem claim is source-audited only; this bundle does not mechanically verify the proof.",
    ),
]

BENCHMARK_HEADERS = [
    "MATH",
    "MATH500",
    "SMath",
    "PIQA",
    "SPhys",
    "Chem",
    "SChem",
    "MMLU",
    "MMLU$^{+}$",
    "GPQA$^{*}$",
    "GPQA",
    "OLY",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_source() -> bytes:
    cache = Path("/tmp") / f"rgr-grpo-{ARXIV_ID}.tar.gz"
    if not cache.exists():
        with urllib.request.urlopen(ARXIV_SOURCE_URL, timeout=30) as response:
            cache.write_bytes(response.read())
    return cache.read_bytes()


def extract_tex(source_bytes: bytes) -> str:
    archive = Path("/tmp") / f"rgr-grpo-{ARXIV_ID}.tar.gz"
    archive.write_bytes(source_bytes)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".tex"):
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                return extracted.read().decode("utf-8", errors="replace")
    raise RuntimeError("no TeX source found")


def source_audit(tex: str) -> dict:
    components = [
        "Construct rubrics",
        "Exploration Assessment",
        "Rubric-Guided Self-Refinement",
        "Mix-Policy GRPO",
        "RGR-GRPO",
    ]
    benchmark_count = sum(1 for header in BENCHMARK_HEADERS if header in tex)
    return {
        "algorithm_components_present": all(component in tex for component in components),
        "rubric_reward_equations": tex.count("rubric-reward"),
        "benchmark_header_count": benchmark_count,
        "benchmark_headers": BENCHMARK_HEADERS,
        "domain_headers": ["Math", "Physics", "Chemistry", "General"],
        "has_table_1_claim_surface": "tab:main" in tex and "7.0\\%" in tex,
        "has_ablation_claim_surface": "Ablation" in tex or "ablation" in tex,
        "has_theorem_claim_surface": "Theorem C.1" in tex or "theorem" in tex.lower(),
    }


def exploration_assessment(scores: dict[str, int]) -> dict:
    failed = [name for name, score in scores.items() if score == 0]
    return {
        "failed_criteria": failed,
        "uses_off_policy_refinement": bool(failed),
    }


def build_evidence_bundle(output_path: str | Path = DEFAULT_OUTPUT) -> dict:
    source = download_source()
    tex = extract_tex(source)
    observations = {
        "source_audit": source_audit(tex),
        "toy_exploration_assessment": {
            "all_criteria_satisfied": exploration_assessment(
                {"fact_1": 1, "process_1": 1, "process_2": 1}
            ),
            "one_failed_criterion": exploration_assessment(
                {"fact_1": 1, "process_1": 1, "process_2": 0}
            ),
        },
    }
    bundle = {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "paper_title": "Reward and Guidance through Rubrics: Promoting Exploration to Improve Multi-Domain Reasoning",
        "snapshot_id": SNAPSHOT_ID,
        "upstream": {
            "arxiv_id": ARXIV_ID,
            "source_url": ARXIV_SOURCE_URL,
            "source_sha256": sha256_bytes(source),
        },
        "estimated_paid_api_cost_usd": 0.0,
        "observations": observations,
        "claims": [
            {
                "claim_sha256": sha,
                "claim": claim,
                "status": status,
                "evidence": evidence,
            }
            for sha, claim, status, evidence in CLAIMS
        ],
        "limitations": [
            "No official code repository, RL training run, model checkpoint, benchmark log, or judge transcript was found or recomputed.",
            "Paper table values are treated as claim text, not reproduced measurements.",
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    bundle = build_evidence_bundle(output_path=args.output)
    print(json.dumps({"output": args.output, "claims": len(bundle["claims"])}, sort_keys=True))


if __name__ == "__main__":
    main()
