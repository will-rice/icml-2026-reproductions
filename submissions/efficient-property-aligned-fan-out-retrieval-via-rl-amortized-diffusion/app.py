from __future__ import annotations

from pathlib import Path

import gradio as gr


PROJECT = Path(__file__).resolve().parent
REPORT = PROJECT / "pages" / "report.md"
EVIDENCE = PROJECT / "evidence" / "bundle.json"


def load_report() -> str:
    if REPORT.exists():
        return REPORT.read_text(encoding="utf-8")
    return "# R4T Reproduction\n\nRun `python generate_evidence.py` to create the evidence report."


with gr.Blocks(title="R4T Reproduction") as demo:
    gr.Markdown(load_report())
    if EVIDENCE.exists():
        gr.File(value=str(EVIDENCE), label="Evidence bundle")


if __name__ == "__main__":
    demo.launch()
