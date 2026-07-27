"""Read-only Gradio viewer for committed DeMix artifact evidence."""

import json
from pathlib import Path
import sys
from typing import Any

import gradio as gr


SUBMISSION_ROOT = Path(__file__).resolve().parent
SRC_PATH = SUBMISSION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from demix.artifacts import analyze_manifest, load_pinned_manifest


BUNDLE_PATH = SUBMISSION_ROOT / "evidence" / "bundle.json"
MANIFEST_PATH = (
    SUBMISSION_ROOT / "evidence" / "inputs" / "sampled_mixture.json"
)


class EvidenceMismatchError(ValueError):
    """Raised when committed observations differ from recomputation."""


def get_evidence(
    bundle_path: Path = BUNDLE_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    """Read evidence only after independently validating its observations."""
    bundle, _ = _load_verified_state(bundle_path, manifest_path)
    return bundle


def get_mixture_observation(
    mixture_id: str,
    bundle_path: Path = BUNDLE_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    """Recompute and return one observation from the pinned release."""
    _, observations = _load_verified_state(bundle_path, manifest_path)
    if mixture_id not in observations["mixture_ids"]:
        raise KeyError(f"unknown released mixture {mixture_id!r}")
    return {
        "mixture_id": mixture_id,
        "raw_weight_sum": observations["raw_weight_sums"][mixture_id],
        "normalized_weights": observations["normalized_weights"][mixture_id],
    }


def _load_verified_state(
    bundle_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_pinned_manifest(manifest_path)
    observations = analyze_manifest(manifest)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("released_artifact_observations") != observations:
        raise EvidenceMismatchError(
            "committed released artifact observations differ from "
            "pinned-manifest recomputation"
        )
    return bundle, observations


evidence = get_evidence()
mixture_ids = evidence["released_artifact_observations"]["mixture_ids"]

with gr.Blocks(title="DeMix Released-Artifact Audit") as demo:
    gr.Markdown("# DeMix released-artifact audit")
    gr.Markdown(
        "ICML 2026 Agent Repro Challenge evidence for paper `uyRIOjFgOn`. "
        "Overall status: **partial**. Benchmark correlation and mixture "
        "comparison claims are **unavailable** from released evaluation inputs."
    )

    with gr.Tab("Evidence bundle"):
        gr.JSON(
            label="Conservative reproduction evidence",
            value=evidence,
        )

    with gr.Tab("Released mixture inspector"):
        gr.Markdown(
            "Select one entry from the exact pinned mixture manifest. "
            "The values below are read from the committed evidence bundle."
        )
        mixture = gr.Dropdown(
            choices=mixture_ids,
            value=mixture_ids[0],
            label="Released mixture ID",
        )
        mixture_observation = gr.JSON(
            label="Released weights and deterministic normalization",
            value=get_mixture_observation(mixture_ids[0]),
        )
        mixture.change(
            fn=get_mixture_observation,
            inputs=mixture,
            outputs=mixture_observation,
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
