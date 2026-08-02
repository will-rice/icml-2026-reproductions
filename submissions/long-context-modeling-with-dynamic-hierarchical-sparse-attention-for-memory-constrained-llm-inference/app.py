from pathlib import Path

import gradio as gr


PROJECT = Path(__file__).resolve().parent
REPORT = PROJECT / "pages" / "report.md"
MEASUREMENTS = PROJECT / "pages" / "01-measurements.md"


def _read(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "Run `python generate_evidence.py` to create this page."


with gr.Blocks(title="DHSA Reproduction Evidence") as demo:
    gr.Markdown(_read(REPORT))
    gr.Markdown(_read(MEASUREMENTS))


if __name__ == "__main__":
    demo.launch()
