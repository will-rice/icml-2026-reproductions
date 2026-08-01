"""Hugging Face Space app for the 3ViewSense reproduction evidence."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent


def load_summary() -> str:
    evidence_path = PROJECT_ROOT / "evidence.json"
    if not evidence_path.exists():
        return "Evidence bundle has not been generated."
    bundle = json.loads(evidence_path.read_text(encoding="utf-8"))
    lines = [
        "# 3ViewSense Reproduction Evidence",
        "",
        f"Paper: `{bundle['paper_id']}`",
        f"Upstream revision: `{bundle['upstream']['revision']}`",
        "",
    ]
    for claim in bundle["claims"]:
        lines.extend(
            [
                f"## {claim['claim_id']}: `{claim['status']}`",
                claim["observation"],
                "",
            ]
        )
    return "\n".join(lines)


demo = gr.Interface(fn=load_summary, inputs=None, outputs=gr.Markdown(), title="3ViewSense")


if __name__ == "__main__":
    demo.launch()
