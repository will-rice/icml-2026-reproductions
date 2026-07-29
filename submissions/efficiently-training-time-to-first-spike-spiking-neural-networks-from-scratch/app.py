import json
from pathlib import Path
import gradio as gr
import torch

from ettfs_snn.ettfs import (
    evaluate_pooling_constraints,
    run_fashion_mnist_ablation,
    run_decoder_comparison_benchmark,
)


def load_summary_markdown() -> str:
    summary_path = Path(__file__).parent / "pages" / "00-summary.md"
    if summary_path.exists():
        return summary_path.read_text(encoding="utf-8")
    return "Summary page missing."


def run_decoder_demo():
    metrics = run_decoder_comparison_benchmark()
    return json.dumps(metrics, indent=2)


def run_ablation_demo():
    results = run_fashion_mnist_ablation()
    return json.dumps(results, indent=2)


def run_pooling_demo():
    constraints = evaluate_pooling_constraints()
    return json.dumps(constraints, indent=2)


with gr.Blocks(title="ETTFS SNN Reproduction") as demo:
    gr.Markdown(load_summary_markdown())

    with gr.Tab("Decoder Step Reduction"):
        gr.Markdown("Compare inference time-step metrics between TQ-TTFS and TWD decoders across benchmarks.")
        btn_decoder = gr.Button("Run Decoder Comparison Benchmark")
        out_decoder = gr.Code(label="Decoder Benchmark Metrics (JSON)", language="json")
        btn_decoder.click(fn=run_decoder_demo, inputs=[], outputs=[out_decoder])

    with gr.Tab("Fashion-MNIST Ablation"):
        gr.Markdown("Evaluate accuracy progression across ETTFS architectural components.")
        btn_ablation = gr.Button("Run Fashion-MNIST Ablation Evaluation")
        out_ablation = gr.Code(label="Ablation Accuracy Results (%)", language="json")
        btn_ablation.click(fn=run_ablation_demo, inputs=[], outputs=[out_ablation])

    with gr.Tab("Pooling Constraints"):
        gr.Markdown("Verify TTFS single-spike preservation under Max-Pooling vs Average-Pooling.")
        btn_pooling = gr.Button("Verify Pooling Constraints")
        out_pooling = gr.Code(label="Pooling Constraint Verification", language="json")
        btn_pooling.click(fn=run_pooling_demo, inputs=[], outputs=[out_pooling])

if __name__ == "__main__":
    demo.launch()
