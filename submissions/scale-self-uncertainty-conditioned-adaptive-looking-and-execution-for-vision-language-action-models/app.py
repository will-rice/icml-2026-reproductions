from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from generate_evidence import build_evidence


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "evidence" / "bundle.json"


def load_bundle() -> dict:
    if BUNDLE_PATH.exists():
        return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    return build_evidence()


def claim_table() -> list[list[str]]:
    bundle = load_bundle()
    return [
        [claim_id, result["status"], result["summary"]]
        for claim_id, result in bundle["claim_results"].items()
    ]


def bundle_json() -> str:
    return json.dumps(load_bundle(), indent=2, sort_keys=True)


with gr.Blocks(title="SCALE Reproduction") as demo:
    gr.Markdown("# SCALE Reproduction")
    gr.Markdown(
        "CPU evidence for the pinned official SCALE implementation. Benchmark success-rate claims are not counted as reproduced measurements."
    )
    gr.Dataframe(
        headers=["Claim", "Status", "Summary"],
        value=claim_table,
        interactive=False,
        wrap=True,
    )
    gr.Code(value=bundle_json, language="json", label="Evidence bundle")


if __name__ == "__main__":
    demo.launch()
