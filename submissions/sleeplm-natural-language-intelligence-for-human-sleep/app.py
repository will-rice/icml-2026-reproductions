from __future__ import annotations

from pathlib import Path
import json

import gradio as gr


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "evidence" / "bundle.json"


def load_bundle() -> dict:
    return json.loads(BUNDLE_PATH.read_text())


def claim_table(bundle: dict) -> list[list[str]]:
    rows = []
    results = bundle["claim_results"]
    for claim in bundle["target_claims"]:
        result = results[claim["id"]]
        rows.append(
            [
                claim["id"],
                claim["challenge_claim_sha256"],
                result["status"],
                result["evidence"],
                result.get("limitations", ""),
            ]
        )
    return rows


def source_table(bundle: dict) -> list[list[str]]:
    rows = []
    for name, source in bundle["source_files"].items():
        rows.append(
            [
                name,
                source["sha256"],
                source["url"],
                ", ".join(source["observed_facts"]),
            ]
        )
    return rows


bundle = load_bundle()

with gr.Blocks(title="SleepLM Reproduction Evidence") as demo:
    gr.Markdown(
        f"# {bundle['paper_title']}\n"
        f"Paper `{bundle['paper_id']}`. Attempt `{bundle['attempt_id']}`. "
        f"CPU-only evidence; estimated paid API cost USD "
        f"{bundle['estimated_api_cost_usd']:.2f}."
    )
    gr.Markdown(f"**Pinned upstream:** `{bundle['upstream_revision']}`")
    gr.Dataframe(
        headers=["Claim", "Challenge SHA-256", "Status", "Evidence", "Limitations"],
        value=claim_table(bundle),
        row_count=(len(bundle["target_claims"]), "fixed"),
        col_count=(5, "fixed"),
        wrap=True,
        interactive=False,
    )
    gr.Markdown("## Source Files")
    gr.Dataframe(
        headers=["Source", "SHA-256", "URL", "Observed facts"],
        value=source_table(bundle),
        row_count=(len(bundle["source_files"]), "fixed"),
        col_count=(4, "fixed"),
        wrap=True,
        interactive=False,
    )
    gr.JSON(value=bundle["observations"], label="Structured observations")
    gr.Markdown(
        "Dataset-scale evidence is primary-artifact documentation, not a "
        "raw-data recount. Performance claims from Section 4 are excluded."
    )


if __name__ == "__main__":
    demo.launch()
