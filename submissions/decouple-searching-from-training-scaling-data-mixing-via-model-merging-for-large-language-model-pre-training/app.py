"""Read-only Gradio viewer for committed DeMix artifact evidence."""

import json
from pathlib import Path

import gradio as gr


SUBMISSION_ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = SUBMISSION_ROOT / "evidence" / "bundle.json"


def get_evidence():
    """Read the committed deterministic evidence bundle."""
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def get_mixture_observation(mixture_id):
    """Return one already-computed observation from the pinned release."""
    observations = get_evidence()["released_artifact_observations"]
    if mixture_id not in observations["mixture_ids"]:
        raise KeyError(f"unknown released mixture {mixture_id!r}")
    return {
        "mixture_id": mixture_id,
        "raw_weight_sum": observations["raw_weight_sums"][mixture_id],
        "normalized_weights": observations["normalized_weights"][mixture_id],
    }


evidence = get_evidence()
mixture_ids = evidence["released_artifact_observations"]["mixture_ids"]

with gr.Blocks(title="DeMix Released-Artifact Audit") as demo:
    gr.Markdown("# DeMix released-artifact audit")
    gr.Markdown(
        "ICML 2026 Agent Repro Challenge evidence for paper `uyRIOjFgOn`. "
        "Overall status: **partial**. Benchmark correlation and mixture "
        "comparison claims are **unavailable** from released evaluation inputs."
    )

    with gr.Tab("Evidence bundle"):
        gr.JSON(
            label="Conservative reproduction evidence",
            value=evidence,
        )

    with gr.Tab("Released mixture inspector"):
        gr.Markdown(
            "Select one entry from the exact pinned mixture manifest. "
            "The values below are read from the committed evidence bundle."
        )
        mixture = gr.Dropdown(
            choices=mixture_ids,
            value=mixture_ids[0],
            label="Released mixture ID",
        )
        mixture_observation = gr.JSON(
            label="Released weights and deterministic normalization",
            value=get_mixture_observation(mixture_ids[0]),
        )
        mixture.change(
            fn=get_mixture_observation,
            inputs=mixture,
            outputs=mixture_observation,
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
