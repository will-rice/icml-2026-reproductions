"""Gradio Space application for the Motive reproduction."""

import json
import pathlib
import sys

import gradio as gr

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from motive.attribution import (
    compute_motion_mask,
    compute_motion_weighted_attribution,
    make_moving_square_video,
)


def run_motive_demo(frames_count: float, step: float) -> str:
    """Run the real motion-mask mechanism on a deterministic synthetic video."""
    video = make_moving_square_video(int(frames_count), step=int(step))
    mask = compute_motion_mask(video)
    uniform = video.new_ones(video.shape)
    results = {
        "motion_mask_shape": list(mask.shape),
        "mask_max": round(float(mask.max().item()), 4),
        "mask_mean": round(float(mask.mean().item()), 4),
        "masked_attribution_norm": round(
            compute_motion_weighted_attribution(uniform, mask), 4
        ),
    }
    return json.dumps(results, indent=2)


def load_file(relative: str) -> str:
    path = pathlib.Path(__file__).parent / relative
    return path.read_text() if path.exists() else f"{relative} not found."


with gr.Blocks(title="Motion Attribution for Video Generation") as demo:
    gr.Markdown("# Motion Attribution for Video Generation (Motive)")
    gr.Markdown("Reproduction logbook for ICML 2026 Paper ID `zAl9heLw4q`.")

    with gr.Tab("Report"):
        gr.Markdown(load_file("pages/report.md"))

    with gr.Tab("Interactive Demo"):
        frames_slider = gr.Slider(minimum=2, maximum=12, value=8, step=1, label="Frame Count")
        step_slider = gr.Slider(minimum=1, maximum=12, value=4, step=1, label="Square Speed")
        btn = gr.Button("Run Motive Attribution")
        output_json = gr.Code(language="json", label="Attribution Results")
        btn.click(fn=run_motive_demo, inputs=[frames_slider, step_slider], outputs=output_json)

    with gr.Tab("Evidence Summary"):
        gr.Code(value=load_file("evidence_summary.json"), language="json", label="evidence_summary.json")

if __name__ == "__main__":
    demo.launch()
