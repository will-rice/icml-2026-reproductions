from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "evidence" / "bundle.json"


def load_summary() -> str:
    if not BUNDLE.exists():
        return "Evidence bundle has not been generated yet."
    bundle = json.loads(BUNDLE.read_text())
    lines = [
        f"# LM-CC reproduction evidence",
        f"Attempt: `{bundle['attempt_id']}`",
        f"Paper: `{bundle['paper_id']}`",
        f"Upstream: `{bundle['upstream_revision']}`",
        "",
        "| Claim | Status |",
        "| --- | --- |",
    ]
    for claim in bundle["claims"]:
        if claim["selected"]:
            lines.append(f"| {claim['challenge_claim']} | `{claim['status']}` |")
    return "\n".join(lines)


with gr.Blocks() as demo:
    gr.Markdown(load_summary())


if __name__ == "__main__":
    demo.launch()
