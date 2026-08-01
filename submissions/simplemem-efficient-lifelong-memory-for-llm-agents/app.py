from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = PROJECT_ROOT / "evidence" / "bundle.json"


def load_bundle(bundle_path: Path = DEFAULT_BUNDLE) -> dict[str, Any]:
    if not bundle_path.exists():
        return {
            "attempt_id": "8d3d77be-6e6a-48a0-b50e-3a078786181d",
            "paper_id": "oBgLvd5YC6",
            "claims": [],
            "provenance": {},
        }
    return json.loads(bundle_path.read_text(encoding="utf-8"))


def build_markdown(bundle_path: Path = DEFAULT_BUNDLE) -> str:
    bundle = load_bundle(bundle_path)
    rows = "\n".join(
        f"| {claim['claim_index']} | {claim['status']} | {claim['summary']} |"
        for claim in bundle.get("claims", [])
    )
    if not rows:
        rows = "| - | unavailable | Evidence bundle has not been generated in this Space image. |"
    return "\n".join(
        [
            "# SimpleMem: Efficient Lifelong Memory for LLM Agents",
            "",
            f"Attempt `{bundle['attempt_id']}` for paper `{bundle['paper_id']}`.",
            "",
            "No paper-reported table value is treated as a reproduced measurement.",
            "",
            "## Claim Evidence",
            "",
            "| Claim | Status | Summary |",
            "| --- | --- | --- |",
            rows,
        ]
    )


demo = gr.Markdown(build_markdown())


if __name__ == "__main__":
    demo.launch()
