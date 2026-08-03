from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).parent
BUNDLE = ROOT / "evidence" / "bundle.json"


def load_summary() -> str:
    if not BUNDLE.exists():
        return "Evidence bundle has not been generated."
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    lines = [f"# {bundle['paper_title']}", ""]
    for claim in bundle["claims"]:
        lines.append(f"- **{claim['status']}**: {claim['claim']}")
    totals = bundle["observations"]["dataset_totals"]
    lines.extend(
        [
            "",
            f"Named benchmark rows: `{totals['named_rows']}`",
            f"Rows including `toksuite_general`: `{totals['with_general_rows']}`",
        ]
    )
    return "\n".join(lines)


with gr.Blocks(title="TokSuite Reproduction") as demo:
    gr.Markdown(load_summary())


if __name__ == "__main__":
    demo.launch()
