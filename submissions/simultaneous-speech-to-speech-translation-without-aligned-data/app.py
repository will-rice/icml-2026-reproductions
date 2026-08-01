"""Gradio Space application for Hibiki-Zero reproduction."""

import json
from pathlib import Path
import gradio as gr


def load_evidence():
    evidence_file = Path(__file__).parent / "evidence_summary.json"
    if evidence_file.exists():
        with open(evidence_file, "r") as f:
            return json.dumps(json.load(f), indent=2)
    return "Evidence file not found."


def load_report():
    report_file = Path(__file__).parent / "pages" / "report.md"
    if report_file.exists():
        with open(report_file, "r") as f:
            return f.read()
    return "Report file not found."


with gr.Blocks(title="Hibiki-Zero Reproduction") as demo:
    gr.Markdown("# Simultaneous Speech-to-Speech Translation Without Aligned Data (Hibiki-Zero)")
    with gr.Tab("Report"):
        gr.Markdown(load_report())
    with gr.Tab("Evidence Summary"):
        gr.Code(value=load_evidence(), language="json")

if __name__ == "__main__":
    demo.launch()
