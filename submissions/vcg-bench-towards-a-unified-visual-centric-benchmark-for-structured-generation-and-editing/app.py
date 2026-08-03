from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from generate_evidence import build_evidence


def render_summary() -> str:
    bundle_path = Path("evidence/bundle.json")
    bundle = (
        json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle_path.exists()
        else build_evidence()
    )
    lines = [
        f"# {bundle['paper_title']}",
        "",
        f"Paper ID: `{bundle['paper_id']}`",
        "",
        "## Claim Results",
    ]
    for claim_id, result in bundle["claim_results"].items():
        lines.append(f"- `{claim_id}`: **{result['status']}**")
        lines.append(f"  {result['claim']}")
        lines.append(f"  {result['observation']}")
    lines.extend(["", "## Unreplicated"])
    lines.extend(f"- {item}" for item in bundle["unreplicated"])
    return "\n".join(lines)


demo = gr.Interface(
    fn=render_summary,
    inputs=None,
    outputs=gr.Markdown(),
    title="VCG-Bench Reproduction",
)


if __name__ == "__main__":
    demo.launch()
