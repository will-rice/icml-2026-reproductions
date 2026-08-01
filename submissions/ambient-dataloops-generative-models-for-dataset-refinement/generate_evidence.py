from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CLAIMS = [
    {
        "id": 1,
        "claim": "Ambient Dataloops iteratively trains a diffusion model on a noisy dataset and uses posterior sampling to refine that same dataset for subsequent loops.",
        "status_if_found": "verified",
        "patterns": ["Algorithm 1", "posterior sampling", "restored point"],
        "reason": "The released paper text and model card describe the loop structure and posterior-sampling restoration step.",
    },
    {
        "id": 2,
        "claim": "On CIFAR-10 with 90% corrupted and 10% clean data, one Ambient Dataloops iteration improves unconditional and conditional generation metrics over baselines.",
        "status_if_found": "inconclusive",
        "patterns": ["Table 1", "CIFAR-10", "90%"],
        "reason": "The table is located in the paper, but the audit did not find released executable result artifacts or checkpoints sufficient to recompute the FID/C-FID values on CPU.",
    },
    {
        "id": 3,
        "claim": "The denoising rate per loop has a non-monotonic effect and can enter a madness regime.",
        "status_if_found": "inconclusive",
        "patterns": ["Figure 3", "madness regime", "Table 3"],
        "reason": "The ablation is located in the paper text, but the training/evaluation sweep was not recomputed.",
    },
    {
        "id": 4,
        "claim": "Ambient Dataloops improves COCO zero-shot text-to-image generation and GenEval scores relative to using the initial synthetic dataset as clean.",
        "status_if_found": "inconclusive",
        "patterns": ["Table 2", "COCO", "GenEval"],
        "reason": "The HF model card and Space support the text-to-image artifact lineage, but the COCO/GenEval metrics are not recomputed here.",
    },
    {
        "id": 5,
        "claim": "In de novo protein backbone design, one Ambient Dataloops iteration produces a new Pareto point with a 14.3% diversity increase for a 0.2% designability decrease.",
        "status_if_found": "inconclusive",
        "patterns": ["Figure 4", "14.3%", "0.2%", "designability"],
        "reason": "The protein metric claim is located in the paper, but no released CPU-executable protein evaluation artifact was found.",
    },
    {
        "id": 6,
        "claim": "The paper provides theoretical justification that idealized dataset refinement can reduce estimation error after looping.",
        "status_if_found": "verified",
        "patterns": ["Theoretical Modeling", "estimation error", "Proof of Lemma 1"],
        "reason": "The extracted paper text contains the theory section and proof structure for the idealized refinement claim.",
    },
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context(text: str, pattern: str, window: int = 140) -> str:
    index = text.lower().find(pattern.lower())
    if index < 0:
        return ""
    start = max(0, index - window)
    end = min(len(text), index + len(pattern) + window)
    return " ".join(text[start:end].split())


def find_evidence(path: Path, patterns: Iterable[str]) -> dict:
    text = path.read_text(encoding="utf-8")
    matched = [pattern for pattern in patterns if pattern.lower() in text.lower()]
    snippets = {pattern: _context(text, pattern) for pattern in matched}
    return {
        "path": str(path),
        "found": len(matched) == len(list(patterns)),
        "matched_patterns": matched,
        "snippets": snippets,
        "sha256": sha256_file(path),
    }


def build_evidence(
    *,
    paper_text: Path,
    model_card: Path,
    space_app: Path,
    model_revision: str,
    space_revision: str,
    generated_at: str | None = None,
) -> dict:
    sources = {
        "paper_text": paper_text,
        "model_card": model_card,
        "space_app": space_app,
    }
    claims = []
    combined_text_path = _combined_source_text(sources)
    for spec in CLAIMS:
        evidence = find_evidence(combined_text_path, spec["patterns"])
        status = spec["status_if_found"] if evidence["found"] else "inconclusive"
        claims.append(
            {
                "id": spec["id"],
                "claim": spec["claim"],
                "status": status,
                "reason": spec["reason"],
                "patterns": spec["patterns"],
                "evidence": evidence,
            }
        )

    return {
        "paper_id": "Li5ki5Dopo",
        "title": "Ambient Dataloops: Generative Models for Dataset Refinement",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "artifact_revisions": {
            "arxiv": "2601.15417",
            "model": model_revision,
            "space": space_revision,
        },
        "source_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "claims": claims,
        "environment": {
            "python_requirement": ">=3.10,<3.13",
        },
        "limitations": [
            "No diffusion training was run.",
            "No FID, C-FID, COCO, GenEval, or protein designability/diversity metrics were recomputed.",
            "Paper-reported metric values are reported only as located claims, not reproduced measurements.",
        ],
    }


def _combined_source_text(sources: dict[str, Path]) -> Path:
    target = Path("/tmp/ambient_dataloops_combined_sources.txt")
    parts = []
    for name, path in sources.items():
        parts.append(f"\n--- {name}: {path} ---\n")
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    target.write_text("\n".join(parts), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-text", type=Path, required=True)
    parser.add_argument("--model-card", type=Path, required=True)
    parser.add_argument("--space-app", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--space-revision", required=True)
    parser.add_argument("--generated-at")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence = build_evidence(
        paper_text=args.paper_text,
        model_card=args.model_card,
        space_app=args.space_app,
        model_revision=args.model_revision,
        space_revision=args.space_revision,
        generated_at=args.generated_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
