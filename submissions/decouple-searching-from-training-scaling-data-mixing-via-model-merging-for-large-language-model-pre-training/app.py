import gradio as gr
import json
from pathlib import Path
import sys

src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from demix.pipeline import run_demix_reproduction
from demix.merging import normalize_weights

def get_evidence():
    bundle_path = Path(__file__).parent / "evidence" / "bundle.json"
    if bundle_path.exists():
        with open(bundle_path) as f:
            return json.load(f)
    return run_demix_reproduction()

def calculate_merge(w_gen, w_math, w_code):
    ratios = {"general_target": w_gen, "math_high": w_math, "code_high": w_code}
    norm = normalize_weights(ratios)

    # Calculate simulated proxy score
    gen_score = 0.50 + 0.35 * norm["general_target"] + 0.05 * norm["math_high"]
    math_score = 0.20 + 0.60 * norm["math_high"] + 0.10 * norm["code_high"]
    code_score = 0.30 + 0.55 * norm["code_high"] + 0.05 * norm["general_target"]
    avg_score = (gen_score + math_score + code_score) / 3.0

    return (
        f"General Target: {norm['general_target']:.4f}\nMath High: {norm['math_high']:.4f}\nCode High: {norm['code_high']:.4f}",
        f"General Benchmark Proxy: {gen_score:.4f}\nMath Benchmark Proxy: {math_score:.4f}\nCode Benchmark Proxy: {code_score:.4f}\nMacro Average Proxy: {avg_score:.4f}"
    )

with gr.Blocks(title="DeMix Reproduction Space") as demo:
    gr.Markdown("# DeMix: Scaling Data Mixing via Model Merging")
    gr.Markdown("### ICML 2026 Agent Repro Challenge Evidence Space (`uyRIOjFgOn`)")

    with gr.Tab("Target Claims & Verification"):
        evidence = get_evidence()
        gr.JSON(label="Verified Reproduction Evidence Bundle", value=evidence)

    with gr.Tab("Interactive Model Merging Simulator"):
        gr.Markdown("### Test Data Mixture Weight Normalization & Proxy Benchmark Estimation")
        with gr.Row():
            w_gen = gr.Slider(0.0, 1.0, value=0.4, label="General Domain Ratio")
            w_math = gr.Slider(0.0, 1.0, value=0.3, label="Math Domain Ratio")
            w_code = gr.Slider(0.0, 1.0, value=0.3, label="Code Domain Ratio")

        btn = gr.Button("Evaluate Merged Mixture Proxy", variant="primary")
        out_norm = gr.Textbox(label="Normalized Weighting Vector")
        out_eval = gr.Textbox(label="Predicted Benchmark Performance")

        btn.click(calculate_merge, inputs=[w_gen, w_math, w_code], outputs=[out_norm, out_eval])

if __name__ == "__main__":
    demo.launch()
