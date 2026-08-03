from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "evidence" / "bundle.json"


def load_summary() -> str:
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    lines = [
        f"# {bundle['paper_title']}",
        "",
        f"Attempt: `{bundle['attempt_id']}`",
        "",
        "| Claim | Status | Evidence |",
        "|---|---|---|",
    ]
    for claim in bundle["claims"]:
        lines.append(
            f"| {claim['claim_index']} | {claim['status']} | {claim['evidence']} |"
        )
    return "\n".join(lines)


with gr.Blocks(title="SPEED-Bench Reproduction") as demo:
    gr.Markdown(load_summary())


if __name__ == "__main__":
    demo.launch()
