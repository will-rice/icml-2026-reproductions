"""Read-only Gradio explorer for canonical AGoQ evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gradio as gr

from agoq_repro.evidence import build_evidence, canonical_json_bytes
from agoq_repro.memory_accounting import (
    audit_table_1,
    fraction_text,
    project_model,
)
from agoq_repro.provenance import load_verified_transcription


def load_committed_evidence(
    path: Path = PROJECT_ROOT / "evidence.json",
) -> dict[str, object]:
    expected = build_evidence(PROJECT_ROOT)
    if path.read_bytes() != canonical_json_bytes(expected):
        raise ValueError("evidence.json is not the canonical pinned evidence bundle")
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_summary() -> tuple[list[list[str]], list[list[str]], str]:
    evidence = load_committed_evidence()
    claims = [
        [
            claim["claim_id"],
            claim["status"],
            claim["challenge_claim_sha256"],
            ", ".join(claim["evidence_basis"]) or "No executable evidence",
            claim["limitation"],
        ]
        for claim in evidence["claims"]
    ]
    table = evidence["reproduced_observations"]["table_1"]
    memory = [
        [method, table[f"{method}_total_u"]] for method in ("bf16", "coat", "agoq")
    ]
    pipeline = evidence["reproduced_observations"]["pipeline"]
    limitation = (
        "The paper-reported integer allocation has a one-unit overshoot: "
        f"target {pipeline['target_storage_units']}, observed maximum "
        f"{pipeline['maximum_reported_storage_units']}. The rounding rule is "
        "not available."
    )
    return claims, memory, limitation


def model_projection(
    batch: object,
    sequence: object,
    hidden: object,
    layers: object,
) -> dict[str, object]:
    raw_dimensions = (batch, sequence, hidden, layers)
    if any(type(value) is not int or value <= 0 for value in raw_dimensions):
        raise ValueError("model dimensions must be positive integers")
    result = project_model(
        audit_table_1(load_verified_transcription(PROJECT_ROOT)),
        batch=batch,
        sequence=sequence,
        hidden=hidden,
        layers=layers,
    )
    return {
        "bytes_per_u": result.bytes_per_u,
        "totals_bytes": {
            method: fraction_text(total)
            for method, total in result.totals_bytes.items()
        },
        "notice": (
            "Optional arithmetic projection only; this is not runtime memory "
            "measurement or claim evidence."
        ),
    }


def create_demo() -> gr.Blocks:
    evidence = load_committed_evidence()
    claims, memory, pipeline_limitation = evidence_summary()
    pipeline = evidence["reproduced_observations"]["pipeline"]
    source_rows = evidence["reproduced_observations"]["source_audit"]
    with gr.Blocks(title="AGoQ Evidence Audit") as blocks:
        gr.Markdown(
            "# AGoQ pinned evidence audit\n"
            "Deterministic CPU arithmetic and source tracing. Paper training "
            "tables are explicitly unavailable."
        )
        with gr.Tab("Claim Status"):
            gr.Dataframe(
                value=claims,
                headers=[
                    "Claim",
                    "Status",
                    "Claim SHA-256",
                    "Evidence basis",
                    "Limitation",
                ],
                interactive=False,
            )
        with gr.Tab("Exact Memory Algebra"):
            gr.Dataframe(
                value=memory,
                headers=["Method", "Total U per layer"],
                interactive=False,
            )
            gr.JSON(value=evidence["reproduced_observations"]["table_1"])
            gr.Markdown(
                "Optional exact model projection. This calls the audited "
                "`U = B*S*H*2 bytes` algebra and is not claim evidence."
            )
            with gr.Row():
                batch = gr.Number(value=1, precision=0, label="Batch")
                sequence = gr.Number(value=4096, precision=0, label="Sequence")
                hidden = gr.Number(value=8192, precision=0, label="Hidden")
                layers = gr.Number(value=32, precision=0, label="Layers")
            projection_output = gr.JSON(label="Projected activation bytes")
            gr.Button("Project").click(
                fn=model_projection,
                inputs=[batch, sequence, hidden, layers],
                outputs=projection_output,
            )
        with gr.Tab("Pipeline Audit"):
            gr.Markdown(pipeline_limitation)
            gr.JSON(value=pipeline)
        with gr.Tab("Pinned Source Trace"):
            gr.Markdown(
                f"Official repository commit: `{evidence['upstream']['commit']}`"
            )
            gr.Dataframe(
                value=[
                    [
                        row["observation_id"],
                        row["disposition"],
                        ", ".join(row["files"]),
                        row["detail"],
                    ]
                    for row in source_rows
                ],
                headers=["Observation", "Disposition", "Pinned files", "Detail"],
                interactive=False,
            )
            gr.JSON(value=evidence["upstream"]["files"])
    return blocks


demo = create_demo()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
