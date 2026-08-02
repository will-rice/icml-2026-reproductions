import json
from pathlib import Path

import gradio as gr


RESULTS = Path(__file__).resolve().parent / "evidence" / "scenesmith_results.json"


def load_rows():
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    return [
        [claim["id"], claim["status"], claim["claim"]]
        for claim in payload["claims"]
    ]


with gr.Blocks(title="SceneSmith Reproduction Evidence") as demo:
    gr.Markdown("# SceneSmith Reproduction Evidence")
    gr.Dataframe(
        headers=["Claim", "Status", "Summary"],
        value=load_rows,
        wrap=True,
    )


if __name__ == "__main__":
    demo.launch()
