from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from generate_evidence import build_evidence


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "evidence" / "bundle.json"


def load_bundle() -> dict:
    if BUNDLE.exists():
        return json.loads(BUNDLE.read_text(encoding="utf-8"))
    return build_evidence()


def rows() -> list[list[str]]:
    return [
        [claim_id, result["status"], result["summary"]]
        for claim_id, result in load_bundle()["claim_results"].items()
    ]


with gr.Blocks(title="WarmServe Reproduction") as demo:
    gr.Markdown("# WarmServe Reproduction")
    gr.Markdown("Pinned source evidence for WarmServe. GPU cluster performance claims remain unreproduced.")
    gr.Dataframe(headers=["Claim", "Status", "Summary"], value=rows, interactive=False, wrap=True)
    gr.Code(value=lambda: json.dumps(load_bundle(), indent=2, sort_keys=True), language="json", label="Evidence bundle")


if __name__ == "__main__":
    demo.launch()
