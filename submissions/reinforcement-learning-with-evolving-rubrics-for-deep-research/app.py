import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence" / "dr_tulu_results.json"


def load_summary() -> str:
    bundle = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    lines = [
        f"# {bundle['paper_title']}",
        "",
        f"Attempt: `{bundle['attempt_id']}`",
        f"Paper: `{bundle['paper_id']}`",
        "",
        "## Claim Statuses",
    ]
    for claim in bundle["claims"]:
        lines.extend(
            [
                "",
                f"### Claim {claim['claim_index']}: {claim['status']}",
                claim["claim"],
                "",
                claim["evidence"],
            ]
        )
    lines.extend(
        [
            "",
            "## Provenance",
            f"Upstream: `{bundle['provenance']['upstream_repo']}@{bundle['provenance']['upstream_commit']}`",
        ]
    )
    return "\n".join(lines)


with gr.Blocks(title="DR Tulu Reproduction") as demo:
    gr.Markdown(load_summary())
    gr.JSON(value=json.loads(EVIDENCE.read_text(encoding="utf-8")), label="Evidence bundle")


if __name__ == "__main__":
    demo.launch()
