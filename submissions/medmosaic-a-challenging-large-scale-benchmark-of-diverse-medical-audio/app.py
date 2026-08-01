import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr

from medmosaic_repro.core import build_evidence_bundle


def load_bundle() -> str:
    return json.dumps(build_evidence_bundle(), indent=2, sort_keys=True)


with gr.Blocks(title="MedMosaic Reproduction Evidence") as demo:
    gr.Markdown("# MedMosaic Reproduction Evidence")
    gr.JSON(value=build_evidence_bundle(), label="Evidence bundle")
    gr.Textbox(value=load_bundle, label="Machine-readable JSON", lines=18)


if __name__ == "__main__":
    demo.launch()
