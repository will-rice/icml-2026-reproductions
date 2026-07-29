from __future__ import annotations

import json
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    gr = None


def load_summary() -> str:
    bundle_path = Path("evidence/bundle.json")
    if not bundle_path.exists():
        return "# h1 Reproduction\n\nEvidence bundle has not been generated yet."
    evidence = json.loads(bundle_path.read_text(encoding="utf-8"))
    upstream = evidence.get("upstream", {})
    lines = [
        f"# {evidence['paper_title']}",
        "",
        f"GitHub revision: `{upstream.get('github_revision', 'not recorded')}`",
        f"arXiv identifier: `{upstream.get('arxiv_identifier', 'not recorded')}`",
        f"License: `{upstream.get('license', 'not recorded')}`",
        "",
        "## Claim Results",
    ]
    for key, result in evidence.get("claim_results", {}).items():
        lines.append(f"- `{key}`: **{result['status']}**")
        lines.append(f"  {result['claim']}")
        lines.append(f"  {result['observation']}")
    urls = evidence.get("provenance", {}).get("source_urls", [])
    if urls:
        lines.extend(["", "## Provenance", *[f"- {url}" for url in urls]])
    return "\n".join(lines)


if gr is not None:
    demo = gr.Interface(
        fn=load_summary,
        inputs=None,
        outputs=gr.Markdown(),
        title="h1 Reproduction",
    )
else:
    demo = None


if __name__ == "__main__":
    if demo is None:
        raise RuntimeError("gradio is required to launch the app")
    demo.launch()
