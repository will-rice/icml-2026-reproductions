from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

import generate_evidence


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "evidence" / "bundle.json"


def _load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        return generate_evidence.build_evidence(ROOT)
    return json.loads(BUNDLE_PATH.read_text())


def _markdown() -> str:
    bundle = _load_bundle()
    rows = []
    for claim in bundle["claims"]:
        rows.append(
            f"- `{claim['status']}` `{claim['claim_sha256'][:12]}`: {claim['claim']}"
        )
    return "\n".join(
        [
            "# MemoryBench Reproduction",
            "",
            f"Attempt: `{bundle['attempt_id']}`",
            f"Snapshot: `{bundle['snapshot_id']}`",
            f"API cost: `${bundle['api_cost_usd']:.2f}`",
            "",
            "## Claims",
            "",
            *rows,
            "",
            "Full machine-readable evidence is in `evidence/bundle.json`.",
        ]
    )


with gr.Blocks(title="MemoryBench Reproduction") as demo:
    gr.Markdown(_markdown())


if __name__ == "__main__":
    demo.launch()
