import json
from pathlib import Path
import gradio as gr

APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "evidence" / "bundle.json"


def load_evidence():
    if not BUNDLE_PATH.exists():
        return "Evidence bundle.json not found."
    return json.dumps(json.loads(BUNDLE_PATH.read_text()), indent=2)


with gr.Blocks(title="FlexRank Reproduction") as demo:
    gr.Markdown("# FlexRank Reproduction Demonstration")
    gr.Markdown("Paper ID: `DK0kvnNelx` | ArXiv: `2602.02680` | ICML 2026")

    with gr.Row():
        evidence_output = gr.Code(label="Verified Evidence Bundle", value=load_evidence, language="json")

demo.launch()
