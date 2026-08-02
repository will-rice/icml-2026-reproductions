from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
BUNDLE = json.loads((ROOT / "evidence" / "bundle.json").read_text(encoding="utf-8"))
SUMMARY = (ROOT / "pages" / "00-summary.md").read_text(encoding="utf-8")


with gr.Blocks(title="To Grok Grokking Reproduction") as demo:
    gr.Markdown(SUMMARY)
    gr.JSON(value=BUNDLE, label="Evidence bundle")


if __name__ == "__main__":
    demo.launch()
