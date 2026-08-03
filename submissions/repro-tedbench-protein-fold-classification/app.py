from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr


PROJECT = Path(__file__).resolve().parent
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tedbench_repro.evidence import build_bundle


def bundle_json() -> str:
    return json.dumps(build_bundle(), indent=2, sort_keys=True)


def summary_markdown() -> str:
    bundle = build_bundle()
    rows = [
        f"- `{result['status']}`: {result['claim']}"
        for result in bundle["claim_results"].values()
    ]
    return "\n".join(
        [
            f"# {bundle['paper_title']}",
            "",
            "CPU-only evidence for four selected TEDBench metadata and architecture claims.",
            "",
            *rows,
        ]
    )


with gr.Blocks(title="TEDBench Reproduction Evidence") as demo:
    gr.Markdown(summary_markdown())
    gr.Code(value=bundle_json(), language="json", label="Evidence bundle")


if __name__ == "__main__":
    demo.launch()
