import json
from pathlib import Path

import gradio as gr


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "evidence" / "bundle.json"


def load_bundle() -> str:
    if not BUNDLE_PATH.exists():
        return "Evidence bundle has not been generated."
    return json.dumps(json.loads(BUNDLE_PATH.read_text(encoding="utf-8")), indent=2)


with gr.Blocks(title="DiReCT Reproduction") as demo:
    gr.Markdown("# DiReCT Reproduction")
    gr.Markdown("Paper ID: `2UH01A9Za0` | arXiv: `2605.31175`")
    gr.Code(value=load_bundle, language="json", label="Evidence Bundle")


demo.launch()
