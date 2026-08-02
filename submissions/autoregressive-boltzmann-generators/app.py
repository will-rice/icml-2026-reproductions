import json
from pathlib import Path

import gradio as gr


PROJECT = Path(__file__).resolve().parent
REPORT = PROJECT / "pages" / "report.md"
EVIDENCE = PROJECT / "evidence" / "autobg_results.json"


def load_report() -> str:
    return REPORT.read_text(encoding="utf-8")


def load_evidence() -> str:
    if not EVIDENCE.exists():
        return "Evidence has not been generated yet."
    return json.dumps(json.loads(EVIDENCE.read_text(encoding="utf-8")), indent=2, sort_keys=True)


with gr.Blocks(title="Autoregressive Boltzmann Generators evidence") as demo:
    gr.Markdown(load_report())
    gr.Code(load_evidence(), language="json", label="Evidence JSON")


if __name__ == "__main__":
    demo.launch()
