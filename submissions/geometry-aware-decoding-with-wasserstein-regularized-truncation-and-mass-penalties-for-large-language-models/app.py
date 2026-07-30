from __future__ import annotations

import json
from pathlib import Path
import gradio as gr

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "evidence" / "bundle.json"


def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        import sys
        SRC = ROOT / "src"
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        from top_w_repro.evidence import build_bundle
        return build_bundle()
    return json.loads(BUNDLE_PATH.read_text())


def claim_table(bundle: dict) -> list[list[str]]:
    rows = []
    results = bundle["claim_results"]
    for claim in bundle["target_claims"]:
        result = results[claim["id"]]
        rows.append(
            [
                claim["id"],
                claim["challenge_claim_sha256"],
                result["status"],
                result["evidence"],
                result.get("limitations", ""),
            ]
        )
    return rows


def metric_table(bundle: dict) -> list[list[str]]:
    rows = []
    for temp_key, metrics in bundle.get("metrics", {}).items():
        rows.append(
            [
                temp_key,
                f"{metrics['entropy_top_w']:.4f}",
                f"{metrics['entropy_min_p']:.4f}",
                f"{metrics['entropy_top_p']:.4f}",
                f"{metrics['entropy_top_h']:.4f}",
                str(int(metrics['subset_size_top_w'])),
                str(int(metrics['subset_size_top_p'])),
            ]
        )
    return rows


bundle = load_bundle()

with gr.Blocks(title="Top-W Geometry-Aware Decoding Reproduction") as demo:
    gr.Markdown(
        f"# {bundle['paper_title']}\n"
        f"Paper ID: `{bundle['paper_id']}` | Attempt ID: `{bundle['attempt_id']}`\n"
        f"**Pinned Upstream:** `{bundle['upstream_revision']}` | **Estimated Paid API Cost:** USD {bundle['estimated_api_cost_usd']:.2f}"
    )

    with gr.Tab("Target Claims & Verification Status"):
        gr.Dataframe(
            headers=["Claim ID", "Challenge Claim SHA-256", "Verdict Status", "Reproduced Evidence", "Limitations"],
            value=claim_table(bundle),
            interactive=False,
        )

    with gr.Tab("Decoder Temperature & Entropy Evaluation"):
        gr.Dataframe(
            headers=["Temp Reg", "Top-W H", "Min-p H", "Top-p H", "Top-H H", "Top-W Pool", "Top-p Pool"],
            value=metric_table(bundle),
            interactive=False,
        )

if __name__ == "__main__":
    demo.launch()
