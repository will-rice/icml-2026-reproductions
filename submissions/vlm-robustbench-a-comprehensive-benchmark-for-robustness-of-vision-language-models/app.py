from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


PROJECT = Path(__file__).resolve().parent
BUNDLE_PATH = PROJECT / "evidence" / "bundle.json"


def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        return {
            "paper_title": "VLM-RobustBench",
            "claims": [],
            "upstream": {},
            "augmentation_audit": {},
            "missing_bundle": str(BUNDLE_PATH),
        }
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def claim_table(bundle: dict) -> list[list[str]]:
    rows = []
    for claim in bundle.get("claims", []):
        rows.append(
            [
                str(claim["claim_index"]),
                claim["status"],
                claim["target_claim"],
                claim["evidence"],
            ]
        )
    return rows


def summary_markdown(bundle: dict) -> str:
    audit = bundle.get("augmentation_audit", {})
    upstream = bundle.get("upstream", {})
    return "\n".join(
        [
            f"# {bundle.get('paper_title', 'VLM-RobustBench')}",
            "",
            f"Paper ID: `{bundle.get('paper_id', 'HwXyyvK7ZJ')}`",
            f"Attempt ID: `{bundle.get('attempt_id', '')}`",
            f"GitHub commit: `{upstream.get('github_commit', '')}`",
            f"Project page SHA-256: `{upstream.get('project_page_sha256_observed', '')}`",
            "",
            "## Audit",
            "",
            f"- Severity-based augmentations: `{audit.get('severity_based_count', 'missing')}`",
            f"- Binary transforms: `{audit.get('binary_count', 'missing')}`",
            f"- Total augmentations: `{audit.get('total_augmentation_count', 'missing')}`",
            f"- Corrupted settings per model-dataset pair: `{audit.get('corrupted_settings_per_model_dataset', 'missing')}`",
        ]
    )


bundle = load_bundle()

with gr.Blocks(title="VLM-RobustBench Reproduction") as demo:
    gr.Markdown(summary_markdown(bundle))
    gr.Dataframe(
        headers=["#", "Status", "Claim", "Evidence"],
        value=claim_table(bundle),
        wrap=True,
        interactive=False,
    )
    gr.JSON(value=bundle, label="Evidence bundle")


if __name__ == "__main__":
    demo.launch()
