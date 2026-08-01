from pathlib import Path

import gradio as gr


REPORT = Path(__file__).resolve().parent / "pages" / "report.md"


def load_report() -> str:
    if REPORT.exists():
        return REPORT.read_text()
    return "Run `python generate_evidence.py` to create the report."


with gr.Blocks(title="FAC Synthesis Reproduction") as demo:
    gr.Markdown(load_report())


if __name__ == "__main__":
    demo.launch()
