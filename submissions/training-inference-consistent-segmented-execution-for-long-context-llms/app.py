import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence_summary.json"


def load_report():
    if not EVIDENCE.exists():
        return "Run generate_evidence.py first.", {}
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    rows = [
        [
            claim["status"],
            claim["challenge_claim_sha256"][:12],
            claim["claim"],
            "\n".join(claim["observations"]),
        ]
        for claim in data["claims"]
    ]
    return json.dumps(data["checks"], indent=2), rows


with gr.Blocks(title="Segmented Execution Evidence") as demo:
    gr.Markdown("# Training-Inference Consistent Segmented Execution")
    checks = gr.Code(label="Checks", language="json")
    claims = gr.Dataframe(
        headers=["Status", "Claim SHA", "Claim", "Observations"],
        datatype=["str", "str", "str", "str"],
        wrap=True,
    )
    demo.load(load_report, outputs=[checks, claims])


if __name__ == "__main__":
    demo.launch()
