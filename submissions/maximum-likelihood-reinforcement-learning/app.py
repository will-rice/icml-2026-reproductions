from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "evidence" / "results.json"
REPORT = ROOT / "pages" / "report.md"


def load_report() -> tuple[str, str]:
    if RESULTS.exists():
        data = json.loads(RESULTS.read_text(encoding="utf-8"))
        summary = "\n".join(f"- {c['id']}: {c['status']}" for c in data["claims"])
    else:
        summary = "Evidence has not been generated."
    report = REPORT.read_text(encoding="utf-8") if REPORT.exists() else summary
    return summary, report


with gr.Blocks(title="MaxRL Reproduction") as demo:
    gr.Markdown("# Maximum Likelihood Reinforcement Learning")
    summary = gr.Markdown()
    report = gr.Markdown()
    demo.load(load_report, outputs=[summary, report])


if __name__ == "__main__":
    demo.launch()
