"""Read-only Gradio browser for the committed RBench evidence bundle."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
EVIDENCE = json.loads((ROOT / "evidence" / "results.json").read_text())
COMMANDS = json.loads((ROOT / "evidence" / "commands.json").read_text())


def claim_rows() -> list[list[str]]:
    return [
        [
            claim["claim"],
            claim["status"],
            " | ".join(item["summary"] for item in claim["observations"]),
            "; ".join(claim["limitations"]),
        ]
        for claim in EVIDENCE["claims"]
    ]


def census_rows() -> list[list[object]]:
    return [
        [
            item["partition"],
            item["leaderboard_column"],
            item["record_count"],
            item["prompt_path"],
        ]
        for item in EVIDENCE["census"]["categories"]
    ]


def cohort_rows() -> list[list[object]]:
    return [
        [
            item["cohort"],
            item["raw_count"],
            item["valid_count"],
            item["unique_exact_count"],
            len(item["discrepancies"]),
            item["ordered_name_hash"],
        ]
        for item in EVIDENCE["leaderboards"]
    ]


def failure_rows() -> list[list[object]]:
    return [
        [
            item["label"],
            item["status"],
            item["parser_path"] or "not found",
            item["aggregation_path"] or "not found",
        ]
        for item in EVIDENCE["failure_modes"]
    ]


with gr.Blocks() as demo:
    gr.Markdown(
        "# RBench artifact reproduction\n"
        "CPU-only inspection of immutable released artifacts. "
        "No video model or human study runs are represented here."
    )
    gr.Dataframe(
        headers=["Claim", "Status", "Computed evidence", "Limitations"],
        value=claim_rows(),
        interactive=False,
        wrap=True,
    )
    with gr.Tab("Prompt census"):
        gr.Dataframe(
            headers=["Partition", "Category", "Records", "Pinned path"],
            value=census_rows(),
            interactive=False,
        )
    with gr.Tab("Leaderboard cohorts"):
        gr.Dataframe(
            headers=[
                "Cohort",
                "Raw",
                "Valid",
                "Unique",
                "Discrepancies",
                "Ordered-name SHA-256",
            ],
            value=cohort_rows(),
            interactive=False,
        )
        gr.JSON(
            value=EVIDENCE["comparison"],
            label="Cross-revision comparison",
        )
    with gr.Tab("Failure-mode routes"):
        gr.Dataframe(
            headers=["Exact phrase", "Status", "Parser", "Aggregation"],
            value=failure_rows(),
            interactive=False,
        )
    with gr.Tab("Provenance"):
        gr.Code(
            value="\n".join(COMMANDS["commands"]),
            language="shell",
            label="Recorded commands",
        )
        gr.File(
            value=[
                str(ROOT / "evidence" / "results.json"),
                str(ROOT / "evidence" / "input-manifest.json"),
                str(ROOT / "evidence" / "commands.json"),
                str(ROOT / "evidence" / "validation.json"),
            ],
            label="Committed machine-readable evidence",
        )


if __name__ == "__main__":
    demo.launch()
