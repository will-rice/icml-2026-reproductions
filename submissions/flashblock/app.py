import json
import os
import sys
from pathlib import Path

# Add src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gradio as gr
import torch

from flashblock_repro.attention import (
    scaled_dot_product_attention,
    log_space_attention_composition,
    BlockCausalAttentionCache,
)
from flashblock_repro.block_diffusion import BlockDiffusionModel, BlockDiffusionGenerator
from flashblock_repro.metrics import (
    compute_cross_step_stability,
    compute_composition_error,
    compute_speedup_and_flops,
)

def run_simulation(context_length, block_size, update_threshold):
    # Calculate speedup metrics
    speedup_res = compute_speedup_and_flops(
        batch_size=1,
        num_heads=16,
        d_k=64,
        context_len=int(context_length),
        block_size=int(block_size),
        num_steps=10,
        update_threshold=int(update_threshold),
    )

    # Run test block diffusion generator
    model = BlockDiffusionModel(vocab_size=100, embed_dim=64, num_heads=4, num_layers=2)
    generator = BlockDiffusionGenerator(model=model, block_size=int(block_size), update_threshold=int(update_threshold))
    gen_result = generator.generate(num_blocks=3, num_steps_per_block=4, use_flashblock=True)

    stab = gen_result["stability_metrics"]

    report = f"""
### Simulation Results
- **Context Length**: {context_length} tokens
- **Block Size**: {block_size} tokens
- **Update Threshold (τ)**: {update_threshold} tokens

#### Attention Stability Discrepancy:
- **Block-External Cosine Similarity**: `{stab['external_cosine_similarity']:.6f}` (High Stability $\ge 0.95$)
- **Block-Internal Cosine Similarity**: `{stab['internal_cosine_similarity']:.6f}` (Low Stability $\le 0.70$)

#### Attention Caching FLOPs & Speedup:
- **Standard Dense FLOPs**: `{speedup_res['dense_flops']:,.0f}`
- **FlashBlock Cached FLOPs**: `{speedup_res['flashblock_flops']:,.0f}`
- **Theoretical Speedup**: `{speedup_res['theoretical_speedup']:.2f}x`
"""
    return report

def load_evidence_json():
    summary_path = SCRIPT_DIR / "evidence_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return f.read()
    return "evidence_summary.json not found."

with gr.Blocks(title="FlashBlock ICML 2026 Reproduction") as demo:
    gr.Markdown("# FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion")
    gr.Markdown("**ICML 2026 Paper Reproduction** (`4jfuNNghPS` | arXiv:2602.05305)")

    with gr.Tab("Interactive Simulation"):
        gr.Markdown("### Test FlashBlock Attention Caching Speedups & Stability")
        with gr.Row():
            ctx_slider = gr.Slider(minimum=256, maximum=8192, value=2048, step=256, label="Context Length (tokens)")
            block_slider = gr.Slider(minimum=4, maximum=32, value=8, step=4, label="Block Size (B)")
            thresh_slider = gr.Slider(minimum=1, maximum=8, value=2, step=1, label="Update Threshold (τ)")

        sim_btn = gr.Button("Run FlashBlock Simulation", variant="primary")
        sim_output = gr.Markdown()
        sim_btn.click(run_simulation, inputs=[ctx_slider, block_slider, thresh_slider], outputs=[sim_output])

    with gr.Tab("Evidence Summary Artifact"):
        gr.Markdown("### Machine-Readable Verified Claims Artifact")
        evidence_box = gr.Code(value=load_evidence_json(), language="json", label="evidence_summary.json")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
