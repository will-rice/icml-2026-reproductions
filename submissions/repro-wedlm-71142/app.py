from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wedlm_repro.evidence import build_evidence_bundle


def load_summary() -> str:
    path = ROOT / "evidence_summary.json"
    if path.exists():
        return path.read_text()
    return json.dumps(build_evidence_bundle(timestamp="not-generated", git_commit="unknown"), indent=2)


with gr.Blocks(title="WeDLM Reproduction") as demo:
    gr.Markdown("# WeDLM Reproduction")
    gr.Markdown(
        "CPU-only evidence for topological reordering, strict causal reachability, "
        "and streaming parallel decoding. GPU vLLM speedups are marked unreplicated."
    )
    gr.Code(value=load_summary, language="json", label="Evidence Summary")


if __name__ == "__main__":
    demo.launch()
