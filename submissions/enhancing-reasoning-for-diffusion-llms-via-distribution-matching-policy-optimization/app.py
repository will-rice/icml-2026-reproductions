from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from dmpo_repro.evidence import build_bundle


BUNDLE_PATH = ROOT / "evidence" / "bundle.json"


def load_bundle() -> dict:
    if BUNDLE_PATH.exists():
        return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    return build_bundle()


def summary_markdown() -> str:
    bundle = load_bundle()
    rows = []
    for claim in bundle["claims"]:
        rows.append(f"- `{claim['status']}`: {claim['text']}")
    limitations = "\n".join(f"- {item}" for item in bundle["limitations"])
    return (
        f"# {bundle['title']}\n\n"
        f"Attempt `{bundle['attempt_id']}` verifies CPU-only implementation evidence from "
        f"`{bundle['upstream']['github']['repo']}@{bundle['upstream']['github']['commit']}`.\n\n"
        "## Claim Status\n\n"
        + "\n".join(rows)
        + "\n\n## Limitations\n\n"
        + limitations
    )


def bundle_json() -> str:
    return json.dumps(load_bundle(), indent=2, sort_keys=True)


with gr.Blocks(title="DMPO Reproduction") as demo:
    gr.Markdown(summary_markdown())
    gr.Code(value=bundle_json(), language="json", label="Evidence Bundle")


if __name__ == "__main__":
    demo.launch()
