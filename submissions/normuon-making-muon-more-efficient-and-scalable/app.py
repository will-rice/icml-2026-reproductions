from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "evidence" / "bundle.json"


def load_bundle() -> str:
    if not BUNDLE_PATH.exists():
        return "Evidence bundle has not been generated yet."
    return json.dumps(json.loads(BUNDLE_PATH.read_text(encoding="utf-8")), indent=2)


with gr.Blocks(title="NorMuon Reproduction Evidence") as demo:
    gr.Markdown("# NorMuon Reproduction Evidence")
    gr.Code(value=load_bundle, language="json", label="evidence/bundle.json")


if __name__ == "__main__":
    demo.launch()
