from __future__ import annotations

import json
from pathlib import Path
import gradio as gr

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "evidence" / "results.json"
PROVENANCE_PATH = ROOT / "evidence" / "provenance.json"


def load_data():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    return results, provenance


results, provenance = load_data()

with gr.Blocks(title="Know More, Know Clearer Reproduction") as demo:
    gr.Markdown(
        f"# {results['title']}\n\n"
        f"**Paper ID**: `{results['paper_id']}` | **Execution**: `{provenance['execution_environment']}` | **API Cost**: `${provenance['actual_api_cost_usd']:.2f}`\n\n"
        f"**Upstream Revision**: `{results['upstream_revision']}`\n"
    )

    claims_data = [
        [c["claim"], c["status"], c["observation"]]
        for c in results["target_claims"]
    ]

    gr.Dataframe(
        headers=["Target Claim", "Status", "Observation"],
        value=claims_data,
        wrap=True,
        interactive=False,
    )

    gr.JSON(value=results.get("metrics", {}), label="Metrics & Statistical Fit")

if __name__ == "__main__":
    demo.launch()
