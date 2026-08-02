import json
from pathlib import Path

import gradio as gr


def _load_bundle() -> dict:
    path = Path("evidence/bundle.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"claim_results": [], "audits": {}}


def render_report() -> str:
    bundle = _load_bundle()
    rows = [
        "# AVGen-Bench Reproduction",
        "",
        f"Attempt: `{bundle.get('attempt_id', 'missing')}`",
        "",
        "| # | Status | Observation |",
        "|---:|---|---|",
    ]
    for result in bundle.get("claim_results", []):
        rows.append(f"| {result['claim_index']} | {result['status']} | {result['observation']} |")
    return "\n".join(rows)


with gr.Blocks(title="AVGen-Bench Reproduction") as demo:
    gr.Markdown(render_report())


if __name__ == "__main__":
    demo.launch()
