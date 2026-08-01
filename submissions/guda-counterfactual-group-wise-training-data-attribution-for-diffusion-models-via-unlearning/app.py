from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = PROJECT_ROOT / "evidence" / "bundle.json"


def load_bundle() -> dict:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def load_summary() -> dict:
    bundle = load_bundle()
    return {
        "paper_id": bundle["paper_id"],
        "attempt_id": bundle["attempt_id"],
        "claim_count": len(bundle["claims"]),
        "upstream_commit": bundle["upstream"]["commit"],
        "estimated_paid_api_cost_usd": bundle["estimated_paid_api_cost_usd"],
        "statuses": {claim["claim_sha256"]: claim["status"] for claim in bundle["claims"]},
    }


def render_markdown() -> str:
    bundle = load_bundle()
    lines = [
        f"# {bundle['paper_title']}",
        "",
        f"- Paper ID: `{bundle['paper_id']}`",
        f"- Attempt ID: `{bundle['attempt_id']}`",
        f"- Upstream: `{bundle['upstream']['repository']}@{bundle['upstream']['commit']}`",
        f"- Paid API cost: `${bundle['estimated_paid_api_cost_usd']:.2f}`",
        "",
        "## Claim Results",
    ]
    for claim in bundle["claims"]:
        lines.extend(
            [
                "",
                f"### `{claim['status']}`",
                claim["claim"],
                "",
                claim["evidence"],
            ]
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in bundle["limitations"])
    return "\n".join(lines)


try:
    import gradio as gr

    demo = gr.Interface(fn=render_markdown, inputs=None, outputs=gr.Markdown(), title="GUDA Evidence Bundle")
except Exception:  # pragma: no cover - permits local tests without Gradio installed
    demo = None


if __name__ == "__main__":
    if demo is None:
        print(render_markdown())
    else:
        demo.launch()
