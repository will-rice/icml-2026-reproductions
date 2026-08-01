from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = PROJECT_ROOT / "evidence" / "bundle.json"


def load_bundle() -> dict:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def render_markdown() -> str:
    bundle = load_bundle()
    lines = [
        f"# {bundle['paper_title']}",
        "",
        f"- Paper ID: `{bundle['paper_id']}`",
        f"- Attempt ID: `{bundle['attempt_id']}`",
        f"- Upstream: `{bundle['upstream']['arxiv_id']}`",
        "",
        "## Claim Results",
    ]
    for claim in bundle["claims"]:
        lines.extend(["", f"### `{claim['status']}`", claim["claim"], "", claim["evidence"]])
    return "\n".join(lines)


try:
    import gradio as gr

    demo = gr.Interface(fn=render_markdown, inputs=None, outputs=gr.Markdown(), title="RGR-GRPO Evidence")
except Exception:  # pragma: no cover
    demo = None


if __name__ == "__main__":
    if demo is None:
        print(render_markdown())
    else:
        demo.launch()
