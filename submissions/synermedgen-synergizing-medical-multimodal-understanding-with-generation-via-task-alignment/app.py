import json
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))

from synermedgen_repro import build_evidence_bundle


def load_bundle():
    bundle = build_evidence_bundle()
    rows = [
        [
            claim["challenge_claim_sha256"][:12],
            claim["status"],
            claim["target_claim"],
            claim["observation"],
        ]
        for claim in bundle["claims"]
    ]
    return rows, json.dumps(bundle["artifact_audit"], indent=2), json.dumps(bundle, indent=2)


with gr.Blocks(title="SynerMedGen Reproduction Audit") as demo:
    gr.Markdown("# SynerMedGen Reproduction Audit")
    gr.Markdown("CPU-only artifact availability and claim-status audit for ICML 2026 paper `Tyv61ZKb9s`.")
    status_table = gr.Dataframe(
        headers=["Claim SHA", "Status", "Claim", "Observation"],
        datatype=["str", "str", "str", "str"],
        wrap=True,
    )
    artifact_json = gr.Code(label="Artifact audit", language="json")
    bundle_json = gr.Code(label="Evidence bundle", language="json")
    demo.load(load_bundle, outputs=[status_table, artifact_json, bundle_json])


if __name__ == "__main__":
    demo.launch()
