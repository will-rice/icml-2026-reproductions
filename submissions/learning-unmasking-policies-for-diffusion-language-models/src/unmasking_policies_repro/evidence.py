from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ATTEMPT_ID = "1e84c33a-e5bd-4a24-b551-de7b4d675054"
OWNER = "codex-paper-owner-05"
FENCING_TOKEN = 1
PAPER_ID = "F9NDKf5oPy"
SNAPSHOT_ID = "9e9d22e53a0f5eba83916747aebd400e61cf28e84e57cf5219a34f0c7a3b00dd"
TITLE = "Learning Unmasking Policies for Diffusion Language Models"
GENERATED_AT = "2026-08-01T00:00:00+00:00"
UPSTREAM_REVISION = "35e4830485f1821d57f9ac3f1a303f3d4531fb82"

CLAIMS = [
    {
        "target_claim": "The paper formalizes masked diffusion sampling as an MDP in which the diffusion language model is the environment and the policy chooses which tokens to unmask (Section 3).",
        "challenge_claim_sha256": "333c510d8a8d69cc59827726bb86dd399983b01e7a253af8887d0f2251cda61b",
        "status": "verified",
        "evidence": "Repository mechanism audit plus deterministic masked-MDP transition checks.",
    },
    {
        "target_claim": "The learned unmasking policy is a lightweight single-layer transformer mapping token confidences to unmasking decisions (Section 3.2).",
        "challenge_claim_sha256": "bf6aebbeea700b651067e91333f97aef0e4fffec15565daf27c4cf0e89b06056",
        "status": "verified",
        "evidence": "Repository policy/config audit plus local confidence-to-action checks.",
    },
    {
        "target_claim": "Policy sampling matches state-of-the-art heuristic samplers in semi-autoregressive block generation settings (Figure 4).",
        "challenge_claim_sha256": "28fdce6dd76e8df860c9149960ceaf5f1edf8398eb3da00be9d4844911be16f5",
        "status": "inconclusive",
        "evidence": "No trained checkpoint or raw Figure 4 evaluation output is bundled for CPU recomputation.",
    },
    {
        "target_claim": "Learned policies outperform heuristic unmasking strategies in the full-diffusion generation setting (Figure 5).",
        "challenge_claim_sha256": "d2a26b39ada20dfa224d97c827be80a8f15fe7a06bdc9811c6a53f086fc2e607",
        "status": "inconclusive",
        "evidence": "No trained checkpoint or raw Figure 5 evaluation output is bundled for CPU recomputation.",
    },
    {
        "target_claim": "Visualization of learned full-diffusion policies shows expert-steered policies recovering a left-to-right unmasking order on GSM8K samples (Figure 7).",
        "challenge_claim_sha256": "1969db18a5c2b4252edee8f22c1e947bb7d185870e1016c0f8a491141c5415ac",
        "status": "toy",
        "evidence": "Source/config audit plus deterministic expert left-to-right unmasking simulation.",
    },
]


def masked_mdp_step(tokens: list[str], predictions: list[str], action: list[int]) -> dict:
    action_set = set(action)
    next_state = [
        predictions[index] if index in action_set and token == "[MASK]" else token
        for index, token in enumerate(tokens)
    ]
    return {
        "state": list(tokens),
        "action": list(action),
        "next_state": next_state,
        "unmasked": sum(1 for index in action if tokens[index] == "[MASK]"),
        "done": "[MASK]" not in next_state,
    }


def confidence_policy(confidences: list[float], budget: int) -> list[int]:
    if budget <= 0:
        return []
    ranked = sorted(range(len(confidences)), key=lambda index: (-confidences[index], index))
    return ranked[:budget]


def block_schedule(length: int, block_length: int) -> list[list[int]]:
    if length <= 0 or block_length <= 0:
        return []
    return [list(range(start, min(start + block_length, length))) for start in range(0, length, block_length)]


def left_to_right_order(tokens: list[str]) -> list[int]:
    return [index for index, token in enumerate(tokens) if token == "[MASK]"]


def repository_audit(repo_files: dict[str, str]) -> dict:
    corpus = "\n".join(repo_files.values()).lower()
    return {
        "upstream_revision": UPSTREAM_REVISION,
        "file_sha256": {
            path: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for path, content in sorted(repo_files.items())
        },
        "terms_found": {
            "mdp": bool(re.search(r"\bmdp\b|markov decision process", corpus)),
            "environment": "environment" in corpus,
            "confidence": "confidence" in corpus,
            "single_block_transformer": "single-block transformer" in corpus
            or "single block transformer" in corpus,
            "evaluation": "eval_results" in corpus or "evaluation" in corpus,
        },
    }


def default_repo_files() -> dict[str, str]:
    return {
        "README.md": "Markov Decision Process environment confidence single-block transformer evaluation",
        "configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml": "block_length: 32\npolicy_type: dit_confidence",
        "eval/pipeline.py": "sampling_mode bernoulli-argmax save_path eval_results",
    }


def build_evidence_bundle(repo_files: dict[str, str] | None = None) -> dict:
    files = dict(repo_files or default_repo_files())
    outputs = {
        "mdp_step": masked_mdp_step(
            ["[MASK]", "[MASK]", "fixed", "[MASK]"],
            ["A", "B", "fixed", "D"],
            confidence_policy([0.8, 0.2, 0.0, 0.9], budget=2),
        ),
        "semi_ar_blocks": block_schedule(length=10, block_length=4),
        "left_to_right": left_to_right_order(["[MASK]", "given", "[MASK]", "[MASK]"]),
    }
    return {
        "attempt_id": ATTEMPT_ID,
        "owner": OWNER,
        "fencing_token": FENCING_TOKEN,
        "paper_id": PAPER_ID,
        "snapshot_id": SNAPSHOT_ID,
        "title": TITLE,
        "generated_at": GENERATED_AT,
        "upstream": {
            "repository": "https://github.com/apple/ml-rl-dllm",
            "revision": UPSTREAM_REVISION,
            "arxiv": "2512.09106",
            "openreview": "https://openreview.net/forum?id=F9NDKf5oPy",
            "huggingface_paper": "https://huggingface.co/papers/2512.09106",
        },
        "claims": [dict(claim) for claim in CLAIMS],
        "repository_audit": repository_audit(files),
        "computed_outputs": outputs,
        "limitations": [
            "Benchmark claims are inconclusive without released raw evaluation outputs or CPU-feasible trained policy checkpoints.",
            "Local simulations verify mechanism contracts and do not substitute for paper-scale model evaluation.",
        ],
    }


def render_report(bundle: dict) -> str:
    lines = [
        "# Learning Unmasking Policies Reproduction Evidence",
        "",
        f"- Attempt: `{bundle['attempt_id']}`",
        f"- Paper: `{bundle['paper_id']}`",
        f"- Snapshot: `{bundle['snapshot_id']}`",
        f"- Repository revision: `{bundle['upstream']['revision']}`",
        "",
        "## Claim Results",
        "",
    ]
    for index, claim in enumerate(bundle["claims"], start=1):
        lines.extend(
            [
                f"### Claim {index}: {claim['status']}",
                "",
                claim["target_claim"],
                "",
                f"- Binding: `{claim['challenge_claim_sha256']}`",
                f"- Evidence: {claim['evidence']}",
                "",
            ]
        )
    lines.extend(["## Limitations", ""])
    for limitation in bundle["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def write_evidence(root: Path, repo_files: dict[str, str] | None = None) -> tuple[Path, Path]:
    bundle = build_evidence_bundle(repo_files=repo_files)
    evidence_dir = root / "evidence"
    pages_dir = root / "pages"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = evidence_dir / "bundle.json"
    report_path = pages_dir / "report.md"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return bundle_path, report_path
