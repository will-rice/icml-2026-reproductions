from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = PROJECT_ROOT / "evidence" / "bundle.json"


def _load_bundle() -> dict:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def _claim_table(bundle: dict) -> list[list[str]]:
    rows = []
    for claim_id, result in bundle["claim_results"].items():
        rows.append([
            claim_id,
            result["status"],
            result["claim"],
            result["evidence"],
        ])
    return rows


def build_app() -> gr.Blocks:
    bundle = _load_bundle()
    observations = json.dumps(bundle["observations"], indent=2, sort_keys=True)
    pins = json.dumps(bundle["upstream_pins"], indent=2, sort_keys=True)
    with gr.Blocks(title="RoboMME Reproduction") as demo:
        gr.Markdown(f"# {bundle['paper_title']}")
        gr.Markdown(f"Paper ID: `{bundle['paper_id']}`")
        gr.Dataframe(
            headers=["Claim ID", "Status", "Claim", "Evidence"],
            value=_claim_table(bundle),
            datatype=["str", "str", "str", "str"],
            wrap=True,
            interactive=False,
        )
        with gr.Tab("Observations"):
            gr.Code(observations, language="json")
        with gr.Tab("Pins"):
            gr.Code(pins, language="json")
        with gr.Tab("Excluded"):
            gr.Code(
                json.dumps(bundle["excluded_claims"], indent=2, sort_keys=True),
                language="json",
            )
    return demo


demo = build_app()


if __name__ == "__main__":
    demo.launch()
