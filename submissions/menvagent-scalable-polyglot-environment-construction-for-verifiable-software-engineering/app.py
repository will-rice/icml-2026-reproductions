from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from generate_evidence import build_bundle


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "evidence" / "bundle.json"


def load_bundle() -> dict:
    if BUNDLE.exists():
        return json.loads(BUNDLE.read_text(encoding="utf-8"))
    return build_bundle(
        source_root=Path("/tmp/menvagent-upstream-codex03"),
        arxiv_source=Path("/tmp/menvagent-2601.22859-src.tar"),
        arxiv_pdf=Path("/tmp/menvagent-2601.22859.pdf"),
        output=BUNDLE,
    )


def rows() -> list[list[str]]:
    return [
        [claim["claim_index"], claim["status"], claim["summary"]]
        for claim in load_bundle()["claims"]
    ]


with gr.Blocks(title="MEnvAgent Reproduction") as demo:
    gr.Markdown("# MEnvAgent Reproduction")
    gr.Markdown("CPU-only evidence from pinned code, arXiv source, and Hugging Face dataset releases.")
    gr.Dataframe(headers=["Claim", "Status", "Summary"], value=rows, interactive=False, wrap=True)
    gr.Code(value=lambda: json.dumps(load_bundle(), indent=2, sort_keys=True), language="json", label="Evidence bundle")


if __name__ == "__main__":
    demo.launch()
