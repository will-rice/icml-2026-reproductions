from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PAPER_ID = "RNuC8Nj6rD"
TITLE = "TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation"
SNAPSHOT_ID = "d32beb9e79859f40a37e565155ef84fb3bdc6bf3679e8f79e8f5414cc3f60600"
CHALLENGE_REVISION = "81166abbeb76e5f79ff87e51061b5a0306507203"
UPSTREAM_REPO = "ChatterjeeLab/TD3B"
UPSTREAM_REVISION = "7d3c9bfe171a1db77e7b5431c572dadce8520bb5"

CHECKPOINT_LFS = {
    "checkpoints/pretrained.ckpt": {
        "sha256": "b259f022c21121f5c755fed61230d6fdf2626ee4ab8a23df479b3cf553fd4aef",
        "size": 1386966244,
    },
    "checkpoints/td3b.ckpt": {
        "sha256": "9b8aeecbfe29b4652860028135c2d7abd2688cfa51aa939b419dd3aec41495d4",
        "size": 231462144,
    },
    "checkpoints/direction_oracle.pt": {
        "sha256": "5ee476c8100752caab069d17569beaece06728d3c8a92223b603c3cba6a9246d",
        "size": 2850095568,
    },
}

CLAIM_TEXTS = {
    1: "TD3B formulates allosteric binder design as control over sequence-conditioned transition operators rather than optimization toward static conformations (Section 4.2).",
    2: "The framework combines a target-aware Direction Oracle, a soft binding-affinity gate, and amortized fine-tuning of a masked discrete diffusion model (Figure 2).",
    3: "The Direction Oracle achieves 0.93 accuracy, 0.90 precision, 0.91 recall, and 0.90 F1 for binary direction classification (Table 1).",
    4: "TD3B obtains the highest gated reward among compared pre-trained, classifier guidance, SMC, TDS, and PepTune baselines (Table 2).",
    5: "For targeted transition control, generated binders achieve 61% success for forward transitions and 100% success for reverse transitions under the paper's success definition (Table 3).",
    6: "The paper evaluates TD3B-designed agonist and antagonist binders on GLP-1R and OX1R case studies (Figures 4 and 5).",
}

REQUIRED_SOURCE_PATTERNS = {
    "direction_oracle": ("td3b/direction_oracle.py", ["DirectionalOracle|DirectionOracle"]),
    "soft_affinity_gate": ("td3b/td3b_scoring.py", ["affinity_predictor|binding_affinity", "direction_oracle"]),
    "gated_reward": ("td3b/td3b_scoring.py", ["gated_reward"]),
    "td3b_losses": ("td3b/td3b_losses.py", ["L_WDCE", "L_ctr", "L_KL"]),
    "amortized_finetuning": ("td3b/td3b_finetune.py", ["td3b_finetune|amortized", "fine"]),
    "masked_diffusion": ("models/diffusion.py", ["MASK", "Diffusion"]),
    "mcts": ("mcts/peptide_mcts.py", ["MCTS"]),
    "inference_resampling": ("inference.py", ["direction_oracle", "affinity", "resample"]),
}

REQUIRED_RESULT_ARTIFACTS = [
    "data/test.csv",
    "data/train.csv",
    "data/td3b_data_new.csv",
    "generated_binders/agonist",
    "generated_binders/antagonist.tar.gz",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _source_fact(source_root: Path, relative_path: str, patterns: list[str]) -> dict:
    path = source_root / relative_path
    text = _read(path)
    lowered = text.lower()
    matches = []
    for pattern in patterns:
        alternatives = pattern.split("|")
        if any(alternative.lower() in lowered for alternative in alternatives):
            matches.append(pattern)
    return {
        "path": relative_path,
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "required_patterns": patterns,
        "matched_patterns": matches,
        "passed": path.exists() and len(matches) == len(patterns),
    }


def _all_pass(facts: dict[str, dict], names: list[str]) -> bool:
    return all(facts[name]["passed"] for name in names)


def build_evidence(*, source_root: Path, generated_at: str | None = None) -> dict:
    facts = {
        name: _source_fact(source_root, relative_path, patterns)
        for name, (relative_path, patterns) in REQUIRED_SOURCE_PATTERNS.items()
    }
    missing_artifacts = [
        relative_path
        for relative_path in REQUIRED_RESULT_ARTIFACTS
        if not (source_root / relative_path).exists()
    ]
    readme = _read(source_root / "README.md")
    has_case_study_mentions = "glp-1r" in readme.lower() and "ox1r" in readme.lower()

    claim_specs = {
        1: {
            "status": "verified" if _all_pass(facts, ["gated_reward", "masked_diffusion"]) else "inconclusive",
            "reason": "Pinned source implements directional reward over a masked discrete diffusion policy.",
            "evidence_keys": ["gated_reward", "masked_diffusion"],
        },
        2: {
            "status": "verified"
            if _all_pass(
                facts,
                [
                    "direction_oracle",
                    "soft_affinity_gate",
                    "td3b_losses",
                    "amortized_finetuning",
                    "masked_diffusion",
                    "mcts",
                    "inference_resampling",
                ],
            )
            else "inconclusive",
            "reason": "Pinned source contains the Direction Oracle, affinity gate, TD3B losses, MCTS fine-tuning path, and inference resampling path.",
            "evidence_keys": [
                "direction_oracle",
                "soft_affinity_gate",
                "td3b_losses",
                "amortized_finetuning",
                "masked_diffusion",
                "mcts",
                "inference_resampling",
            ],
        },
        3: {
            "status": "unavailable" if missing_artifacts else "inconclusive",
            "reason": "Direction Oracle metrics require primary test labels or evaluation outputs; paper table values are not recomputed here.",
            "evidence_keys": ["direction_oracle"],
        },
        4: {
            "status": "unavailable" if missing_artifacts else "inconclusive",
            "reason": "Baseline gated-reward comparison requires generated results for TD3B and baselines; no table-scale result artifact is present.",
            "evidence_keys": ["gated_reward"],
        },
        5: {
            "status": "unavailable" if missing_artifacts else "inconclusive",
            "reason": "Forward/reverse transition success requires generated binders and success labels; the bundled generated_binders artifacts are absent.",
            "evidence_keys": [],
        },
        6: {
            "status": "toy" if has_case_study_mentions else "unavailable",
            "reason": "The pinned README documents GLP-1R and OX1R case-study artifacts, but absent generated binders prevent independent case-study evaluation.",
            "evidence_keys": [],
        },
    }

    claims = []
    for claim_id, claim_text in CLAIM_TEXTS.items():
        spec = claim_specs[claim_id]
        claims.append(
            {
                "id": claim_id,
                "claim": claim_text,
                "challenge_claim": claim_text,
                "challenge_claim_sha256": sha256_text(claim_text),
                "status": spec["status"],
                "reason": spec["reason"],
                "evidence_keys": spec["evidence_keys"],
            }
        )

    source_files = {}
    for path in sorted(source_root.rglob("*")):
        if path.is_file():
            source_files[str(path.relative_to(source_root))] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }

    return {
        "paper_id": PAPER_ID,
        "title": TITLE,
        "generated_at": generated_at,
        "challenge": {
            "snapshot_id": SNAPSHOT_ID,
            "challenge_revision": CHALLENGE_REVISION,
        },
        "upstream": {
            "repo_id": UPSTREAM_REPO,
            "revision": UPSTREAM_REVISION,
            "source_root": str(source_root),
        },
        "checkpoint_lfs": CHECKPOINT_LFS,
        "source_facts": facts,
        "source_files": source_files,
        "missing_artifacts": missing_artifacts,
        "claims": claims,
        "limitations": [
            "The default evidence path avoids multi-GB checkpoint downloads.",
            "Training, baseline sweeps, Direction Oracle metrics, and transition-success rates are not recomputed without the missing primary data/generated-binder artifacts.",
            "Paper-reported table values are not used as reproduced measurements.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)

    evidence = build_evidence(source_root=args.source_root, generated_at=args.generated_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
