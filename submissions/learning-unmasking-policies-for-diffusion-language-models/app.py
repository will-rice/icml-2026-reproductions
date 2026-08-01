from pathlib import Path

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

from unmasking_policies_repro.evidence import write_evidence


ROOT = Path(__file__).resolve().parent


def load_report() -> str:
    _, report_path = write_evidence(ROOT)
    return report_path.read_text(encoding="utf-8")


if gr is None:  # pragma: no cover
    print(load_report())
else:
    demo = gr.Interface(
        fn=load_report,
        inputs=None,
        outputs=gr.Markdown(),
        title="Learning Unmasking Policies Evidence",
        allow_flagging="never",
    )

    if __name__ == "__main__":
        demo.launch()
