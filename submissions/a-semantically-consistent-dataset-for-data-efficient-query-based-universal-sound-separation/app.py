from __future__ import annotations

import json
from pathlib import Path

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None


BUNDLE_PATH = Path(__file__).parent / "evidence" / "bundle.json"
REPORT_PATH = Path(__file__).parent / "pages" / "report.md"


def load_bundle() -> tuple[str, list[list[str]]]:
    if not BUNDLE_PATH.exists():
        return "Evidence bundle has not been generated.", []
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    rows = [
        [str(result["claim_index"]), result["status"], result["claim_sha256"], result["evidence"]]
        for result in bundle["claim_results"]
    ]
    summary = (
        f"{bundle['title']}\n\n"
        f"Attempt: {bundle['attempt_id']}\n"
        f"Paper: {bundle['paper_id']}\n"
        f"Snapshot: {bundle['snapshot_id']}"
    )
    return summary, rows


def create_demo():
    if gr is None:
        raise RuntimeError("gradio is required to launch the Space app")
    summary, rows = load_bundle()
    with gr.Blocks(title="Hive Reproduction Evidence") as demo:
        gr.Markdown("# Hive Reproduction Evidence")
        gr.Textbox(value=summary, label="Bundle", lines=5, interactive=False)
        gr.Dataframe(
            value=rows,
            headers=["Claim", "Status", "Claim SHA-256", "Evidence"],
            datatype=["str", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )
        pages_dir = Path(__file__).parent / "pages"
        if pages_dir.exists():
            for page_path in sorted(pages_dir.glob("*.md")):
                gr.Markdown(page_path.read_text(encoding="utf-8"))
    return demo


if __name__ == "__main__":
    create_demo().launch()
