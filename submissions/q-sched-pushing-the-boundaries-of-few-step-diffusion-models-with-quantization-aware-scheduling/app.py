import json
import gradio as gr
from qsched.eval import run_evaluation


def evaluate_ui():
    results = run_evaluation()
    return json.dumps(results, indent=2)


with gr.Blocks(title="Q-Sched Reproduction space") as demo:
    gr.Markdown("# ICML 2026 Reproduction: Q-Sched")
    gr.Markdown("**Paper Title**: Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling")
    gr.Markdown("**Paper ID**: `4yzY0GFIJj` | **arXiv**: `2509.01624`")
    
    run_btn = gr.Button("Run Reproduction Evaluation", variant="primary")
    output_json = gr.Code(label="Verified Evidence Output (JSON)", language="json")
    
    run_btn.click(evaluate_ui, outputs=output_json)


if __name__ == "__main__":
    demo.launch()
