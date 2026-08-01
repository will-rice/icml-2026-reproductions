"""
Gradio Web Application for LiME reproduction logbook and interactive evaluation.
"""

import os
import json
import gradio as gr
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
LOGBOOK_PATH = BASE_DIR / "pages" / "logbook.md"
EVIDENCE_PATH = BASE_DIR / "evidence" / "evidence.json"

def get_logbook():
    if LOGBOOK_PATH.exists():
        return LOGBOOK_PATH.read_text()
    return "# Logbook Not Found"

def get_evidence():
    if EVIDENCE_PATH.exists():
        return json.dumps(json.loads(EVIDENCE_PATH.read_text()), indent=2)
    return "{}"

with gr.Blocks(title="LiME Reproduction Logbook") as demo:
    gr.Markdown("# LiME: Lightweight Mixture of Experts for Efficient Multimodal Multi-task Learning Reproduction")

    with gr.Tabs():
        with gr.TabItem("Logbook"):
            gr.Markdown(get_logbook())
        with gr.TabItem("Evidence JSON"):
            gr.Code(get_evidence(), language="json")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
