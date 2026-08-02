from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "evidence" / "bundle.json"


def load_bundle() -> tuple[list[list[str]], str]:
    if not BUNDLE_PATH.exists():
        return [], "Evidence bundle has not been generated."
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    rows = [
        [str(item["claim_index"]), item["status"], item["observation"], item["limitation"]]
        for item in bundle["claim_results"]
    ]
    summary = json.dumps(
        {
            "attempt_id": bundle["attempt_id"],
            "paper_id": bundle["paper_id"],
            "snapshot_id": bundle["snapshot_id"],
            "upstream_pins": bundle["upstream_pins"],
        },
        indent=2,
    )
    return rows, summary


with gr.Blocks(title="d2 reproduction evidence") as demo:
    gr.Markdown("# d2 reproduction evidence")
    table = gr.Dataframe(headers=["Claim", "Status", "Observation", "Limitation"], datatype=["str", "str", "str", "str"])
    summary = gr.Code(language="json")
    demo.load(load_bundle, outputs=[table, summary])


if __name__ == "__main__":
    demo.launch()
