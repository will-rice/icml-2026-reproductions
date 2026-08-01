import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

from unmasking_policies_repro.evidence import write_evidence


def load_report() -> str:
    _, report_path = write_evidence(ROOT)
    return report_path.read_text(encoding="utf-8")


if gr is None:  # pragma: no cover
    print(load_report())
else:
    with gr.Blocks(title="Learning Unmasking Policies Evidence") as demo:
        gr.Markdown(load_report())

    if __name__ == "__main__":
        demo.launch()
