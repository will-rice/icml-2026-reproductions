from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "evidence" / "textatlas5m_results.json"


def load_json() -> dict:
    if BUNDLE.exists():
        return json.loads(BUNDLE.read_text(encoding="utf-8"))
    return {"error": "Run generate_evidence.py to create evidence/textatlas5m_results.json"}


def load_page(name: str) -> str:
    path = ROOT / "pages" / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"{name} is not available."


with gr.Blocks(title="TextAtlas5M Reproduction Evidence") as demo:
    gr.Markdown(load_page("00-summary.md"))
    with gr.Tab("Claim Evidence"):
        gr.Markdown(load_page("01-claims.md"))
    with gr.Tab("Evidence JSON"):
        gr.JSON(value=load_json())


if __name__ == "__main__":
    demo.launch()
