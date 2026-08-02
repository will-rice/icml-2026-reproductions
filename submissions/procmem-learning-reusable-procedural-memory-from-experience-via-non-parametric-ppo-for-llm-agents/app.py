import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr

from procmem_repro.core import generate_evidence_bundle


def load_bundle() -> str:
    return json.dumps(generate_evidence_bundle(), indent=2, sort_keys=True)


with gr.Blocks(title="ProcMEM Reproduction Evidence") as demo:
    gr.Markdown("# ProcMEM Reproduction Evidence")
    gr.JSON(value=generate_evidence_bundle(), label="Evidence bundle")
    gr.Textbox(value=load_bundle, label="Machine-readable JSON", lines=18)


if __name__ == "__main__":
    demo.launch()
