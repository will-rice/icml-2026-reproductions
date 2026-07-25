from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "evidence" / "results.json"
POSTER_PATH = PROJECT_ROOT / "poster_embed.html"
TITLE = "EEG-FM-Bench released-artifact audit"

RESULTS = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
POSTER_HTML = POSTER_PATH.read_text(encoding="utf-8")


def evidence_summary() -> dict[str, object]:
    """Return the committed EEG-FM-Bench evidence summary."""
    return {
        "scope": RESULTS["scope"],
        "claims": [
            {
                "claim_id": claim["claim_id"],
                "kind": claim["kind"],
                "status": claim["status"],
            }
            for claim in RESULTS["claims"]
        ],
        "unavailable_claims": RESULTS["unavailable_claims"],
        "source_revisions": {
            name: item["revision"]
            for name, item in RESULTS["provenance"]["inputs"].items()
        },
    }


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=TITLE, fill_width=True) as blocks:
        gr.Markdown(
            """
# EEG-FM-Bench released-artifact audit

Deterministic CPU evidence from the pinned released repository. This audit
does not reproduce the GPU leaderboard or use gated raw EEG datasets.
"""
        )
        with gr.Tabs():
            with gr.Tab("Evidence"):
                gr.JSON(
                    value=RESULTS,
                    label="Committed evidence/results.json",
                    open=True,
                )
            with gr.Tab("Poster"):
                gr.HTML(value=POSTER_HTML)
            with gr.Tab("Summary API"):
                summary_output = gr.JSON(
                    value=evidence_summary(),
                    label="Evidence summary",
                    open=True,
                )
                refresh = gr.Button("Return committed evidence summary")
                refresh.click(
                    fn=evidence_summary,
                    inputs=None,
                    outputs=summary_output,
                    api_name="evidence_summary",
                    api_description=evidence_summary.__doc__,
                    show_progress="hidden",
                )
    return blocks


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
