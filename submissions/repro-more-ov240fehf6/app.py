import json
from pathlib import Path

import gradio as gr


PROJECT = Path(__file__).resolve().parent
BUNDLE_PATH = PROJECT / "evidence" / "bundle.json"


def _load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        import generate_evidence

        generate_evidence.write_bundle(PROJECT)
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def render_summary() -> str:
    bundle = _load_bundle()
    lines = [
        "# MORE Reproduction Evidence",
        f"Paper ID: `{bundle['paper_id']}`",
        f"Attempt ID: `{bundle['attempt_id']}`",
        f"Snapshot ID: `{bundle['snapshot_id']}`",
        "",
        "## Claims",
    ]
    for claim in bundle["claims"]:
        lines.extend(
            [
                f"### {claim['status'].title()}: `{claim['claim_sha256']}`",
                claim["claim"],
                "",
                "**Computed observation:** "
                + "; ".join(claim["computed_observations"]),
                "",
            ]
        )
    lines.extend(["## Limitations", *[f"- {item}" for item in bundle["limitations"]]])
    return "\n".join(lines)


with gr.Blocks(title="MORE Reproduction Evidence") as demo:
    gr.Markdown(render_summary())
    gr.JSON(value=_load_bundle(), label="bundle.json")


if __name__ == "__main__":
    demo.launch()
