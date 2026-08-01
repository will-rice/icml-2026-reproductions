from __future__ import annotations

from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent


def _load_markdown() -> str:
    summary = ROOT / "pages" / "00-summary.md"
    if summary.exists():
        return summary.read_text(encoding="utf-8")
    return (
        "# Memorization-capacity reproduction\n\n"
        "Run `python generate_evidence.py` to build the evidence bundle."
    )


with gr.Blocks(title="Memorization-capacity Reproduction") as demo:
    gr.Markdown(_load_markdown())
    gr.File(value=str(ROOT / "evidence" / "bundle.json"), label="Evidence bundle")


if __name__ == "__main__":
    demo.launch()
