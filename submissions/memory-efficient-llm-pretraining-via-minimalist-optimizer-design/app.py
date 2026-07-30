import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "evidence" / "results.json"


def load_results():
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {"error": "evidence/results.json has not been generated"}


with gr.Blocks(title="SCALE Reproduction") as demo:
    gr.Markdown("# SCALE Reproduction")
    gr.JSON(value=load_results, label="Evidence bundle")


if __name__ == "__main__":
    demo.launch()
