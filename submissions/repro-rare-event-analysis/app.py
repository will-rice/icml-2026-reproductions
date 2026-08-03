from pathlib import Path
import sys

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rare_event_llm.evidence import build_evidence_bundle  # noqa: E402


def load_bundle() -> dict:
    return build_evidence_bundle()


with gr.Blocks(title="Rare Event Analysis Reproduction") as demo:
    gr.Markdown("# Rare Event Analysis of Large Language Models")
    gr.Markdown(
        "CPU-only rare-event estimator checks on an exactly enumerable "
        "stochastic text process."
    )
    gr.JSON(value=load_bundle(), label="Evidence bundle")


if __name__ == "__main__":
    demo.launch()
