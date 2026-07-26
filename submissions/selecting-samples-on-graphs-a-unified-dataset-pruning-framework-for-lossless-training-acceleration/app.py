"""Offline Gradio Space for the accepted graph-pruning evidence bundle."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from graph_pruning_repro.evidence import build_evidence, validate_evidence


PROJECT_ROOT = Path(__file__).resolve().parent
EVIDENCE_PATH = PROJECT_ROOT / "evidence" / "evidence.json"
SCHEMA_PATH = PROJECT_ROOT / "evidence" / "schema.json"

validate_evidence(EVIDENCE_PATH, SCHEMA_PATH, PROJECT_ROOT)
ACCEPTED_EVIDENCE = json.loads(EVIDENCE_PATH.read_text())
STARTUP_VALIDATED = True

PANEL_NAMES = (
    "Summary",
    "Variants",
    "Witnesses",
    "Proof ledger",
    "Unavailable claims",
)

DOWNLOAD_PATHS = (
    Path("evidence/evidence.json"),
    *tuple(
        Path(witness["artifact_path"])
        for witness in ACCEPTED_EVIDENCE["witnesses"]
    ),
    Path("NOTICE.md"),
    Path("LICENSE"),
    Path("LICENSES/CC-BY-NC-SA-4.0.txt"),
)


def _summary_markdown() -> str:
    claims = "\n".join(
        f"- {claim}" for claim in ACCEPTED_EVIDENCE["target_claims"]
    )
    return (
        "# Graph Dataset Pruning Formal Evidence\n\n"
        "This offline Space presents accepted formal and bounded finite "
        "evidence. Paper-reported training measurements remain unavailable.\n\n"
        "## Target claims\n\n"
        f"{claims}\n\n"
        f"**Pinned paper:** {ACCEPTED_EVIDENCE['paper']['revision']}  \n"
        f"**Evidence source revision:** "
        f"`{ACCEPTED_EVIDENCE['source_revision']}`"
    )


def _variant_rows() -> list[list[object]]:
    return [
        [
            result["model_variant"],
            result["audit"],
            result["evidence_kind"],
            result["status"],
        ]
        for result in ACCEPTED_EVIDENCE["claim_results"]
    ]


def _witness_rows() -> list[list[object]]:
    return [
        [
            witness["id"],
            witness["property"],
            witness["artifact_path"],
        ]
        for witness in ACCEPTED_EVIDENCE["witnesses"]
    ]


def _proof_rows() -> list[list[object]]:
    rows: list[list[object]] = []
    for variant, ledger in ACCEPTED_EVIDENCE["proof_ledger"]["symbolic"][
        "ledgers"
    ].items():
        for equation in ledger:
            for conclusion in equation["conclusions"]:
                rows.append(
                    [
                        variant,
                        equation["equation"],
                        conclusion["check_id"],
                        conclusion["status"],
                    ]
                )
    return rows


def _unavailable_rows() -> list[list[object]]:
    return [
        [record["id"], record["status"], record["reason"]]
        for record in ACCEPTED_EVIDENCE["unavailable_claims"]
    ]


def recompute(output_dir: str | Path | None = None) -> tuple[str, Path]:
    """Run the same bounded CPU evidence builder into an isolated directory."""

    destination = (
        Path(tempfile.mkdtemp(prefix="graph-pruning-recompute-"))
        if output_dir is None
        else Path(output_dir)
    )
    evidence = build_evidence(
        destination,
        source_revision=ACCEPTED_EVIDENCE["source_revision"],
    )
    command = next(
        record
        for record in evidence["commands"]
        if record["id"] == "recompute"
    )
    status = (
        f"PASS actual={command['actual']} ceiling={command['ceiling']}"
    )
    return status, destination / "evidence.json"


def _recompute_for_ui() -> tuple[str, str]:
    status, evidence_path = recompute()
    return status, str(evidence_path)


with gr.Blocks(title="Graph Dataset Pruning Formal Evidence") as demo:
    with gr.Tab(PANEL_NAMES[0]):
        gr.Markdown(_summary_markdown())

    with gr.Tab(PANEL_NAMES[1]):
        gr.Dataframe(
            headers=["Variant", "Audit", "Evidence kind", "Status"],
            value=_variant_rows(),
            interactive=False,
        )

    with gr.Tab(PANEL_NAMES[2]):
        gr.Dataframe(
            headers=["Witness", "Property", "Artifact"],
            value=_witness_rows(),
            interactive=False,
        )

    with gr.Tab(PANEL_NAMES[3]):
        gr.Dataframe(
            headers=["Variant", "Equation", "Check", "Status"],
            value=_proof_rows(),
            interactive=False,
        )

    with gr.Tab(PANEL_NAMES[4]):
        gr.Markdown(
            "Paper-reported accuracy, training-time, acceleration, detection, "
            "and segmentation measurements were not recomputed."
        )
        gr.Dataframe(
            headers=["Claim", "Status", "Reason"],
            value=_unavailable_rows(),
            interactive=False,
        )

    with gr.Tab("Downloads and licenses"):
        gr.Markdown(
            "The seven-author attribution is in `NOTICE.md`. Executable code "
            "uses the MIT `LICENSE`; transcriptions and evidence use "
            "`LICENSES/CC-BY-NC-SA-4.0.txt`."
        )
        for relative_path in DOWNLOAD_PATHS:
            gr.DownloadButton(
                label=f"Download {relative_path.as_posix()}",
                value=str(PROJECT_ROOT / relative_path),
            )

    with gr.Tab("Bounded CPU recomputation"):
        gr.Markdown(
            "Run the canonical exact-rational evidence builder on CPU. "
            "Output is written to an isolated temporary directory."
        )
        recompute_button = gr.Button("Recompute bounded evidence")
        recompute_status = gr.Textbox(label="Status", interactive=False)
        recompute_file = gr.File(label="Recomputed evidence")
        recompute_button.click(
            fn=_recompute_for_ui,
            inputs=None,
            outputs=[recompute_status, recompute_file],
        )


if __name__ == "__main__":
    demo.launch()
