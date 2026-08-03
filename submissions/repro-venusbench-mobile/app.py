from __future__ import annotations

import json

import gradio as gr

from venusbench_mobile_repro.evidence import load_evidence_bundle, render_summary_markdown


def _bundle() -> dict:
    return load_evidence_bundle()


def summary_markdown() -> str:
    return render_summary_markdown(_bundle())


def evidence_json() -> str:
    return json.dumps(_bundle(), indent=2)


with gr.Blocks(title="VenusBench-Mobile ICML 2026 Reproduction Evidence") as demo:
    gr.Markdown(summary_markdown())
    gr.Code(evidence_json(), language="json", label="evidence/bundle.json")


if __name__ == "__main__":
    demo.launch()
