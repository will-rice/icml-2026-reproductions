from __future__ import annotations

import gradio as gr

from wire_graph_rope.claims import build_evidence_bundle


def load_bundle() -> dict:
    return build_evidence_bundle()


with gr.Blocks(title="WIRE Reproduction Evidence") as demo:
    gr.Markdown("# Rotary Position Encodings for Graphs")
    gr.JSON(value=load_bundle(), label="Evidence bundle")


if __name__ == "__main__":
    demo.launch()
