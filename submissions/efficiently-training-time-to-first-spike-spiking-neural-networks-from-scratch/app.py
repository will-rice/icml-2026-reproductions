import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import json

import gradio as gr

from ettfs_snn.ettfs import (
    evaluate_pooling_constraints,
    run_component_ablation,
    run_decoder_comparison_benchmark,
    run_init_signal_propagation_test,
)

PAGES_DIR = Path(__file__).parent / "pages"


def load_page(name: str) -> str:
    path = PAGES_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"Page {name} missing."


def run_propagation_demo():
    return json.dumps(run_init_signal_propagation_test(), indent=2)


def run_decoder_demo():
    return json.dumps(run_decoder_comparison_benchmark(), indent=2)


def run_ablation_demo():
    return json.dumps(run_component_ablation(), indent=2)


def run_pooling_demo():
    return json.dumps(evaluate_pooling_constraints(), indent=2)


with gr.Blocks(title="ETTFS SNN Reproduction") as demo:
    gr.Markdown(load_page("00-summary.md"))

    with gr.Tab("Claim 1 — Init signal propagation"):
        gr.Markdown(load_page("01-claim-1-init-signal-propagation.md"))
        btn_prop = gr.Button("Re-run ETTFS-init vs Kaiming propagation")
        out_prop = gr.Code(label="Signal propagation measurements (JSON)", language="json")
        btn_prop.click(fn=run_propagation_demo, inputs=[], outputs=[out_prop])

    with gr.Tab("Claim 2 — Decoder steps"):
        gr.Markdown(load_page("02-claim-2-temporal-weighting-decoder.md"))
        btn_decoder = gr.Button("Re-run decoder comparison")
        out_decoder = gr.Code(label="Decoder benchmark metrics (JSON)", language="json")
        btn_decoder.click(fn=run_decoder_demo, inputs=[], outputs=[out_decoder])

    with gr.Tab("Claim 3 — Pooling constraints"):
        gr.Markdown(load_page("03-claim-3-pooling-constraints.md"))
        btn_pooling = gr.Button("Re-run pooling commutation check")
        out_pooling = gr.Code(label="Pooling constraint measurements (JSON)", language="json")
        btn_pooling.click(fn=run_pooling_demo, inputs=[], outputs=[out_pooling])

    with gr.Tab("Claim 4 — Dataset accuracies"):
        gr.Markdown(load_page("04-claim-4-dataset-accuracies-not-reproduced.md"))

    with gr.Tab("Claim 5 — Component ablation"):
        gr.Markdown(load_page("05-claim-5-component-ablation.md"))
        btn_ablation = gr.Button("Re-train ablation configurations")
        out_ablation = gr.Code(label="Ablation accuracies (JSON)", language="json")
        btn_ablation.click(fn=run_ablation_demo, inputs=[], outputs=[out_ablation])

if __name__ == "__main__":
    demo.launch()
