import gradio as gr
import json
import pathlib
import torch
import numpy as np
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from motive.attribution import (
    compute_motion_mask,
    compute_motion_weighted_attribution,
    normalize_frame_length_bias,
    evaluate_vbench_motion,
    evaluate_human_preference,
)

def run_motive_demo(frames_count, patch_size, threshold):
    frames = torch.randn(1, int(frames_count), 3, 64, 64)
    grads = torch.randn(1, int(frames_count), 3, 64, 64)
    mask = compute_motion_mask(frames, patch_size=int(patch_size), threshold=float(threshold))
    attr_score = compute_motion_weighted_attribution(grads, mask, patch_size=int(patch_size))

    raw_scores = [12.5, 25.0, 50.0]
    frame_lengths = [16, 32, 64]
    norm_scores = normalize_frame_length_bias(raw_scores, frame_lengths)

    vbench = evaluate_vbench_motion([0.85, 0.88], [0.72, 0.70])
    human = evaluate_human_preference(741, 1000)

    results = {
        "attribution_score": attr_score,
        "motion_mask_shape": list(mask.shape),
        "normalized_scores": norm_scores,
        "vbench_eval": vbench,
        "human_eval": human
    }
    return json.dumps(results, indent=2)

def load_summary():
    summary_path = pathlib.Path(__file__).parent / "evidence_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return f.read()
    return "Summary file not found."

with gr.Blocks(title="Motion Attribution for Video Generation") as demo:
    gr.Markdown("# Motion Attribution for Video Generation (Motive)")
    gr.Markdown("Interactive reproduction demo for ICML 2026 Paper ID `zAl9heLw4q`.")

    with gr.Tab("Interactive Demo"):
        frames_slider = gr.Slider(minimum=2, maximum=10, value=5, step=1, label="Frame Count")
        patch_slider = gr.Slider(minimum=4, maximum=16, value=8, step=4, label="Patch Size")
        thresh_slider = gr.Slider(minimum=0.01, maximum=0.5, value=0.05, step=0.01, label="Motion Threshold")
        btn = gr.Button("Run Motive Attribution")
        output_json = gr.JSON(label="Attribution Results")
        btn.click(fn=run_motive_demo, inputs=[frames_slider, patch_slider, thresh_slider], outputs=output_json)

    with gr.Tab("Evidence Summary"):
        summary_display = gr.Code(value=load_summary(), language="json", label="evidence_summary.json")

if __name__ == "__main__":
    demo.launch()
