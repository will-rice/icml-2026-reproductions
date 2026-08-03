from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _bundle() -> dict:
    return json.loads((ROOT / "evidence" / "bundle.json").read_text(encoding="utf-8"))


with gr.Blocks(title="Neural Thickets Reproduction") as demo:
    gr.Markdown(_read("README.md"))
    gr.Markdown(_read("pages/report.md"))
    gr.JSON(value=_bundle(), label="Evidence bundle")


if __name__ == "__main__":
    demo.launch()
