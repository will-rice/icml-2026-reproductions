from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tau2_bench_repro.evidence import build_evidence


def _upstream_root() -> Path:
    return PROJECT_ROOT / "vendor" / "tau2-bench"


def _evidence() -> dict:
    evidence_path = PROJECT_ROOT / "evidence.json"
    if evidence_path.exists():
        return json.loads(evidence_path.read_text(encoding="utf-8"))
    return build_evidence(_upstream_root())


def summary_markdown() -> str:
    evidence = _evidence()
    lines = [
        "# tau2-Bench Reproduction Evidence",
        "",
        f"Paper: `{evidence['paper_id']}`",
        f"Upstream: `{evidence['upstream']['revision']}`",
        "",
    ]
    for claim in evidence["claims"]:
        lines.extend(
            [
                f"## {claim['claim_id']}: {claim['status']}",
                "",
                claim["claim"],
                "",
                f"Claim SHA-256: `{claim['challenge_claim_sha256']}`",
                "",
                f"Provenance: {claim['provenance']}",
                "",
            ]
        )
    return "\n".join(lines)


def evidence_json() -> str:
    return json.dumps(_evidence(), indent=2)


with gr.Blocks(title="tau2-Bench ICML 2026 Reproduction Evidence") as demo:
    gr.Markdown(summary_markdown)
    gr.Code(evidence_json, language="json", label="evidence.json")


if __name__ == "__main__":
    demo.launch()
