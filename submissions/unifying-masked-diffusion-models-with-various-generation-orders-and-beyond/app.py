"""Gradio Space application for Unifying Masked Diffusion Models reproduction."""

import gradio as gr
import json
from pathlib import Path


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


with gr.Blocks(title="Unifying Masked Diffusion Models Reproduction") as demo:
    gr.Markdown("# Unifying Masked Diffusion Models Reproduction (ICML 2026)")
    with gr.Tab("Report"):
        gr.Markdown(load_report())
    with gr.Tab("Evidence Summary"):
        gr.Code(value=load_evidence(), language="json")

if __name__ == "__main__":
    demo.launch()
