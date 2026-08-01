import json
from pathlib import Path

import gradio as gr


EVIDENCE_PATH = Path(__file__).parent / "evidence" / "td3b_results.json"


def load_evidence() -> dict:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def claim_summary(data: dict) -> str:
    rows = []
    for claim in data["claims"]:
        rows.append(f"{claim['id']}. {claim['status'].upper()}: {claim['claim']}\n\n{claim['reason']}")
    return "\n\n".join(rows)


with gr.Blocks() as demo:
    evidence = load_evidence()
    gr.Markdown(f"# {evidence['title']}")
    gr.Markdown(claim_summary(evidence))
    gr.JSON(value=evidence, label="Evidence JSON")


demo.launch()
