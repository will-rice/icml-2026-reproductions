from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from d3llm_repro.evidence import generate_bundle


def load_bundle() -> str:
    bundle_path = Path("evidence/bundle.json")
    if not bundle_path.exists():
        generate_bundle(bundle_path.parent)
    return json.dumps(json.loads(bundle_path.read_text(encoding="utf-8")), indent=2)


with gr.Blocks(title="d3LLM Reproduction Evidence") as demo:
    gr.Markdown("# d3LLM Reproduction Evidence")
    gr.Markdown("CPU-only audit for ICML 2026 Repro Challenge attempt `063c65c5-a8aa-4679-a184-fc83b92a820f`.")
    gr.Code(value=load_bundle, language="json", label="evidence/bundle.json")


if __name__ == "__main__":
    demo.launch()
