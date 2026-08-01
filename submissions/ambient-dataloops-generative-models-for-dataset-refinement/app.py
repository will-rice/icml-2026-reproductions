import json
from pathlib import Path

import gradio as gr


EVIDENCE_PATH = Path(__file__).parent / "evidence" / "ambient_dataloops_results.json"


def load_summary() -> tuple[str, str]:
    data = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    rows = []
    for claim in data["claims"]:
        rows.append(
            f"{claim['id']}. {claim['status'].upper()}: {claim['claim']}\n\n{claim['reason']}"
        )
    return data["title"], "\n\n".join(rows)


with gr.Blocks() as demo:
    title, summary = load_summary()
    gr.Markdown(f"# {title}")
    gr.Markdown(summary)
    gr.JSON(value=json.loads(EVIDENCE_PATH.read_text(encoding="utf-8")), label="Evidence JSON")


demo.launch()
