import json
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))

from udm_grpo_repro import build_evidence_bundle


def load_bundle():
    bundle = build_evidence_bundle()
    rows = [
        [
            claim["sha256"][:12],
            claim["status"],
            claim["target_claim"],
            claim["observation"],
        ]
        for claim in bundle["claims"]
    ]
    return rows, json.dumps(bundle["source_audit"], indent=2), json.dumps(bundle, indent=2)


with gr.Blocks(title="UDM-GRPO Reproduction Audit") as demo:
    gr.Markdown("# UDM-GRPO Reproduction Audit")
    gr.Markdown("CPU-only source/config audit for ICML 2026 paper `WJcFtJriqv`.")
    table = gr.Dataframe(
        headers=["Claim SHA", "Status", "Claim", "Observation"],
        datatype=["str", "str", "str", "str"],
        wrap=True,
    )
    audit = gr.Code(label="Source audit", language="json")
    bundle = gr.Code(label="Evidence bundle", language="json")
    demo.load(load_bundle, outputs=[table, audit, bundle])


if __name__ == "__main__":
    demo.launch()
