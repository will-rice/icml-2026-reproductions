from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "evidence" / "bundle.json"


def _load_bundle() -> dict:
    if BUNDLE_PATH.exists():
        return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    return {"claims": [], "observations": {}, "upstream_revision": "not generated"}


def _claim_rows(bundle: dict) -> list[list[str]]:
    return [
        [
            claim["challenge_claim_sha256"][:12],
            claim["status"],
            claim["summary"],
        ]
        for claim in bundle.get("claims", [])
    ]


bundle = _load_bundle()

with gr.Blocks(title="TRM Complex Reasoning Reproduction") as demo:
    gr.Markdown("# TRM Complex Reasoning Reproduction")
    gr.Markdown(f"`{bundle.get('upstream_revision', 'not generated')}`")
    gr.Dataframe(
        headers=["Claim", "Status", "Evidence Summary"],
        value=_claim_rows(bundle),
        wrap=True,
        interactive=False,
    )
    gr.JSON(value=bundle.get("observations", {}), label="Artifact observations")


if __name__ == "__main__":
    demo.launch()
