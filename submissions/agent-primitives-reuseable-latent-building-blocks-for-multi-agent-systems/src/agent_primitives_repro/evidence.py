from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


ATTEMPT_ID = "c2270ea7-fabd-4292-a117-3b7181c0c5fa"
OWNER = "codex-paper-owner-05"
FENCING_TOKEN = 1
PAPER_ID = "CzShhpY2qU"
SNAPSHOT_ID = "262de4b8f7a83b9fa6af23efd0755d5e77522b789239db644faefa5ba4cf9d30"
TITLE = "Agent Primitives: Reuseable Latent Building Blocks for Multi-Agent Systems"
GENERATED_AT = "2026-08-01T00:00:00+00:00"

CLAIMS = [
    {
        "target_claim": "Agent Primitives instantiates three reusable MAS building blocks: Review, Voting and Selection, and Planning and Execution (Section 3).",
        "challenge_claim_sha256": "8115a999582028cd40604b3d6dd9ee69546547d4babd00b02e8c1abb1b719ff1",
        "status": "toy",
        "evidence": "Source-term audit plus deterministic Review, Voting/Selection, and Planning/Execution simulations.",
    },
    {
        "target_claim": "The primitives communicate internally through KV-cache states rather than only natural-language message passing (Section 3).",
        "challenge_claim_sha256": "5b9aaefaf80d29d999f70165b70dde7f2e9cb90f080dffee3c41a8f4d17228b5",
        "status": "toy",
        "evidence": "Source-term audit plus local KV-cache tensor shape invariant checks.",
    },
    {
        "target_claim": "An Organizer agent selects and composes primitives for each query using a lightweight pool of previously successful configurations (Section 3).",
        "challenge_claim_sha256": "3ff1e2007a7b9f313467db56636bd1e60598d8dfad968e2cec21aeff157168bc",
        "status": "toy",
        "evidence": "Source-term audit plus deterministic overlap-based primitive-pool selection.",
    },
    {
        "target_claim": "Primitive-based MAS improve average accuracy by 12.0-16.5% over single-agent baselines across evaluated tasks (Section 4).",
        "challenge_claim_sha256": "7eadb6ed2e71e12ea036c70a7f6c4e4ef52c30861df2d99f613d30d1b0012129",
        "status": "inconclusive",
        "evidence": "No released raw benchmark outputs or executable evaluation artifacts were available in this package.",
    },
    {
        "target_claim": "Compared with text-based MAS, Agent Primitives reduce token usage and inference latency by about 3-4x while adding only 1.3-1.6x overhead over single-agent inference (Appendix E).",
        "challenge_claim_sha256": "67bd705462e640c09cc50efc1020c8eb02746643ecc5cfa913ef7f6d80da91a1",
        "status": "inconclusive",
        "evidence": "No released raw token or latency traces were available in this package.",
    },
]


def simulate_review(initial: str, critiques: Iterable[str]) -> dict:
    critique_list = list(critiques)
    revision = initial
    for index, critique in enumerate(critique_list, start=1):
        revision = f"{revision} | review-{index}: {critique}"
    return {
        "initial": initial,
        "critiques": critique_list,
        "revision": revision,
        "rounds": len(critique_list),
    }


def select_vote(candidates: Iterable[str], scores: dict[str, float]) -> dict:
    ranked = sorted(((scores[name], name) for name in candidates), key=lambda item: (-item[0], item[1]))
    score, selected = ranked[0]
    return {"selected": selected, "score": score, "tie_break": "lexicographic"}


def plan_and_execute(goal: str, steps: Iterable[str]) -> dict:
    step_list = list(steps)
    trace = []
    for index, step in enumerate(step_list):
        phase = "plan" if index == 0 else "verify" if index == len(step_list) - 1 else "execute"
        trace.append(f"{phase}:{step}")
    return {"goal": goal, "steps": step_list, "trace": trace, "completed": bool(step_list)}


def kv_cache_shape_valid(*, layers: int, tokens: int, heads: int, dim: int) -> dict:
    valid = all(value > 0 for value in (layers, tokens, heads, dim))
    shape = [2, layers, tokens, heads, dim] if valid else []
    elements = 1
    for value in shape:
        elements *= value
    return {"valid": valid, "shape": shape, "elements": elements if valid else 0}


def organizer_select(query: str, pool: Iterable[dict]) -> dict:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    names = []
    for primitive in pool:
        name = primitive["name"]
        names.append(name)
        tags = set(primitive.get("tags", []))
        scored.append((-(len(query_terms & tags)), name, len(query_terms & tags)))
    scored.sort()
    _, selected, overlap = scored[0]
    return {"selected": selected, "overlap": overlap, "available": sorted(names)}


def arxiv_audit(arxiv_text: str = "", arxiv_files: Iterable[str] | None = None) -> dict:
    text = arxiv_text.lower()
    files = sorted(arxiv_files or [])
    return {
        "text_sha256": hashlib.sha256(arxiv_text.encode("utf-8")).hexdigest(),
        "source_files": files,
        "terms_found": {
            "review": "review" in text,
            "voting_selection": "voting" in text and "selection" in text,
            "planning_execution": "planning" in text and "execution" in text,
            "kv_cache": "kv-cache" in text or "kv cache" in text,
            "organizer": "organizer" in text,
            "appendix_e": "appendix e" in text or any("appendix_e" in file.lower() for file in files),
        },
    }


def build_evidence_bundle(arxiv_text: str = "", arxiv_files: Iterable[str] | None = None) -> dict:
    primitive_outputs = {
        "review": simulate_review("candidate solution", ["find missing assumption", "add verification"]),
        "voting_selection": select_vote(
            ["single", "critic", "tool"],
            {"single": 0.2, "critic": 0.7, "tool": 0.7},
        ),
        "planning_execution": plan_and_execute("solve query", ["decompose", "execute", "check"]),
        "kv_cache": kv_cache_shape_valid(layers=2, tokens=16, heads=4, dim=64),
        "organizer": organizer_select(
            "decompose execute check",
            [
                {"name": "review", "tags": ["critique", "revise"]},
                {"name": "planning", "tags": ["decompose", "execute"]},
                {"name": "voting", "tags": ["rank", "select"]},
            ],
        ),
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
            "arxiv": "2602.03695",
            "openreview": "https://openreview.net/forum?id=CzShhpY2qU",
            "huggingface_paper": "https://huggingface.co/papers/2602.03695",
            "executable_repository": None,
            "upstream_revision": "arxiv:2602.03695",
        },
        "claims": [dict(claim) for claim in CLAIMS],
        "arxiv_audit": arxiv_audit(arxiv_text, arxiv_files),
        "computed_outputs": primitive_outputs,
        "limitations": [
            "Toy statuses are limited to local mechanism checks and source-term audit.",
            "Accuracy, token, and latency claims remain inconclusive without released raw benchmark outputs.",
            "Paper-reported values are recorded only as claim text, not as reproduced measurements.",
        ],
    }


def render_report(bundle: dict) -> str:
    lines = [
        "# Agent Primitives Reproduction Evidence",
        "",
        f"- Attempt: `{bundle['attempt_id']}`",
        f"- Paper: `{bundle['paper_id']}`",
        f"- Snapshot: `{bundle['snapshot_id']}`",
        f"- Generated: `{bundle['generated_at']}`",
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


def write_evidence(root: Path, arxiv_text: str = "", arxiv_files: Iterable[str] | None = None) -> tuple[Path, Path]:
    bundle = build_evidence_bundle(arxiv_text=arxiv_text, arxiv_files=arxiv_files)
    evidence_dir = root / "evidence"
    pages_dir = root / "pages"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = evidence_dir / "bundle.json"
    report_path = pages_dir / "report.md"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return bundle_path, report_path
