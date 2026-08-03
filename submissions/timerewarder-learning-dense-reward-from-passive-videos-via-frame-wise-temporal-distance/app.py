"""Offline Space UI backed only by committed evidence and a tiny fixture."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr

from timerewarder_repro.fixture import run_fixture
from timerewarder_repro.presentation import claim_rows, load_verified_evidence


BUNDLE = load_verified_evidence(ROOT / "artifacts" / "evidence.json")


def claim_records() -> dict[str, object]:
    return {
        "measurement_sha256": BUNDLE["measurement_sha256"],
        "claims": claim_rows(BUNDLE),
    }


def evidence_summary() -> dict[str, object]:
    representative = BUNDLE["measurements"]["representative"]
    return {
        "measurement_sha256": BUNDLE["measurement_sha256"],
        "protocol": BUNDLE["protocol"],
        "pooled_metrics": representative["pooled_metrics"],
        "mean_voc": representative["mean_voc"],
    }


def rerun_fixture() -> dict[str, object]:
    result = run_fixture()
    payload = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        "diagnostic_only": True,
        "measurement_sha256": hashlib.sha256(payload).hexdigest(),
        "result": result,
    }


with gr.Blocks(title="TimeRewarder Reproduction Evidence") as demo:
    gr.Markdown(
        "# TimeRewarder Reproduction Evidence\n"
        "Canonical released-artifact measurements and a deterministic, "
        "diagnostic-only fixture."
    )
    summary_output = gr.JSON(label="Evidence summary")
    claims_output = gr.JSON(label="Six claim records")
    fixture_output = gr.JSON(label="Diagnostic fixture")
    with gr.Row():
        summary_button = gr.Button("Load evidence summary")
        claims_button = gr.Button("Load claim records")
        fixture_button = gr.Button("Rerun deterministic fixture")
    summary_button.click(
        evidence_summary, outputs=summary_output, api_name="evidence_summary"
    )
    claims_button.click(claim_records, outputs=claims_output, api_name="claim_records")
    fixture_button.click(
        rerun_fixture, outputs=fixture_output, api_name="rerun_fixture"
    )


if __name__ == "__main__":
    demo.launch()
