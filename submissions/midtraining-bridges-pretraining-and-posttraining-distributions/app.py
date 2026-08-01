# Midtraining Bridges Reproduction Space
import json
import gradio as gr
from midtraining_bridges.core import run_full_reproduction


def evaluate_ui():
    results = run_full_reproduction()
    return json.dumps(results, indent=2)


with gr.Blocks(title="Midtraining Bridges Reproduction Space") as demo:
    gr.Markdown("# ICML 2026 Reproduction: Midtraining Bridges")
    gr.Markdown("**Paper Title**: Midtraining Bridges Pretraining and Posttraining Distributions")
    gr.Markdown("**Paper ID**: `5PfEQzE9bf` | **arXiv**: `2510.14865`")

    run_btn = gr.Button("Run Reproduction Evaluation", variant="primary")
    output_json = gr.Code(label="Verified Evidence Output (JSON)", language="json")

    run_btn.click(evaluate_ui, outputs=output_json)


if __name__ == "__main__":
    demo.launch()
