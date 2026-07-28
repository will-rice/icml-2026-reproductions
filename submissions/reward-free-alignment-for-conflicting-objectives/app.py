import json
from pathlib import Path
import gradio as gr

PROJECT_ROOT = Path(__file__).parent
PAGES_DIR = PROJECT_ROOT / "pages"
EVIDENCE_PATH = PROJECT_ROOT / "evidence/results.json"

PAGE_PATHS = sorted(list(PAGES_DIR.glob("*.md")))

if EVIDENCE_PATH.is_file():
    EVIDENCE = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
else:
    EVIDENCE = {"paper_id": "vSzRJyg6k0", "claims": [], "audits": {}}


def build_app():
    with gr.Blocks(title="RACO Score Reproduction Evidence") as app:
        gr.Markdown("# RACO: Reward-Free Alignment for Conflicting Objectives")
        gr.Markdown(f"**Paper ID:** `{EVIDENCE.get('paper_id', 'vSzRJyg6k0')}` | **Snapshot:** `{EVIDENCE.get('snapshot_id', 'N/A')[:12]}`")

        with gr.Tabs():
            for page_path in PAGE_PATHS:
                title = page_path.stem.replace("-", " ").title()
                content = page_path.read_text(encoding="utf-8")
                with gr.TabItem(title):
                    gr.Markdown(content)

            with gr.TabItem("Raw Evidence JSON"):
                gr.JSON(value=EVIDENCE)

    return app


app = build_app()

if __name__ == "__main__":
    app.launch()
