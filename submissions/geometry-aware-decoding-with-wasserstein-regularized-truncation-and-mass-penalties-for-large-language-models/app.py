from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
PAGES_DIR = ROOT / "pages"
BUNDLE_PATH = ROOT / "evidence" / "bundle.json"

PAGE_TITLES = {
    "00-summary.md": "Executive summary",
    "01-claim-1-wasserstein-objective.md": "Claim 1: Objective audit",
    "02-claim-2-exact-subset-update.md": "Claim 2: Exact S-step",
    "03-claim-3-gsm8k-not-reproduced.md": "Claim 3: GSM8K (not reproduced)",
    "04-claim-4-gpqa-not-reproduced.md": "Claim 4: GPQA (not reproduced)",
    "05-claim-5-openended-not-reproduced.md": "Claim 5: AlpacaEval/MT-Bench (not reproduced)",
    "06-methods-and-provenance.md": "Methods & provenance",
}

with gr.Blocks(title="Top-W Geometry-Aware Decoding Reproduction") as demo:
    for filename, title in PAGE_TITLES.items():
        with gr.Tab(title):
            gr.Markdown((PAGES_DIR / filename).read_text())
    with gr.Tab("Raw evidence bundle"):
        gr.Code(
            value=json.dumps(
                json.loads(BUNDLE_PATH.read_text()), indent=2, sort_keys=True
            ),
            language="json",
        )

if __name__ == "__main__":
    demo.launch()
