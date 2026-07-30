"""Gradio Space app for High-accuracy sampling reproduction."""

import json
from pathlib import Path
import gradio as gr


def load_bundle() -> dict:
    bundle_path = Path(__file__).parent / "evidence" / "bundle.json"
    if bundle_path.exists():
        return json.loads(bundle_path.read_text())
    return {"error": "Evidence bundle not found"}


bundle_data = load_bundle()

with gr.Blocks(title="High-Accuracy Diffusion Sampling Reproduction") as demo:
    gr.Markdown("# Reproduction: High-accuracy sampling for diffusion models and log-concave distributions")
    gr.Markdown("Paper ID: `71132` | Upstream Revision: `arxiv:2602.01338v2` | ICML 2026 Agent Repro Challenge")

    with gr.Tab("Claims & Evidence"):
        gr.JSON(bundle_data)

    with gr.Tab("Summary"):
        gr.Markdown("""
        ### Target Claims
        1. **Theorem 4.3**: Diffusion sampler achieves $\\delta$-error in $\\tilde{O}(\\text{polylog}(1/\\delta))$ steps given accurate score estimates.
        2. **Corollary 4.4**: Complexity reduces to $\\tilde{O}(d^* \\text{polylog}(1/\\delta))$ under intrinsic dimension $d^*$.
        3. **Section 5**: Polylogarithmic accuracy sampler for log-concave distributions using first-order gradient queries.
        """)

if __name__ == "__main__":
    demo.launch()
