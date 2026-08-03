"""Build the machine-readable WeDLM evidence bundle."""

from __future__ import annotations

import os
import platform

from .causal_diffusion import causal_reachability, simulate_streaming_decode, topological_reorder

PAPER_ID = "71142"
ATTEMPT_ID = "64537525-54cc-4b47-b721-72979fa954dd"
SNAPSHOT_ID = "a0ba41c847d742fd8ae3c5b36542ffd1dfe5c7dd0fc1bdbd82cf631d933dd1d1"
CHALLENGE_REVISION = "81166abbeb76e5f79ff87e51061b5a0306507203"
SPACE_ID = "wrice/repro-wedlm-71142"
UPSTREAM_REVISION = (
    "arxiv:2512.22737+github:tencent/WeDLM@8d3f66b18f6a00467f8f87d3ec1f091b1da9204e"
    "+hf:tencent/WeDLM-8B-Instruct@c1e0373ec7e11bc27321a548ea54ea9728b0d9c0"
)

CLAIM_BINDINGS = [
    {
        "claim_id": "causal-diffusion-prefix-cache",
        "challenge_claim_sha256": "c863e0bbfdce6fe5e9f4e69ce8a845a814d954b3a0f1f1c2b39c751dd4b44798",
        "challenge_claim": "WeDLM implements diffusion-style parallel decoding entirely with standard causal attention to preserve prefix KV-cache compatibility (Section 3).",
    },
    {
        "claim_id": "topological-reordering",
        "challenge_claim_sha256": "b28ec2d483ef39319347cc164e3f8322750f95f3dfb0e595b4dc616d29cda187",
        "challenge_claim": "Topological Reordering lets each masked position condition on observed tokens under a strict causal mask by moving observed tokens into the physical prefix while preserving logical positions (Section 3).",
    },
    {
        "claim_id": "streaming-parallel-decoding",
        "challenge_claim_sha256": "cafcce7c1211fbf898f3a997b5dc99cbad90191f1878d86180b43ac2ba979f60",
        "challenge_claim": "WeDLM introduces a streaming decoding procedure that commits confident tokens into a growing left-to-right prefix while maintaining a fixed parallel workload (Section 3).",
    },
    {
        "claim_id": "gpu-vllm-speedup",
        "challenge_claim_sha256": "b33475be6926404f68c6a286deef74b56d8c09c291c9cdbd093423e427b463ef",
        "challenge_claim": "Against vLLM-served autoregressive baselines under matched deployment settings, WeDLM approaches 3x speedups on reasoning benchmarks and up to 10x speedups in low-entropy generation regimes (Section 4).",
    },
]


def build_evidence_bundle(*, timestamp: str, git_commit: str) -> dict:
    reorder = topological_reorder(["The", "<mask>", "sky", "<mask>"], observed=[0, 2])
    reachability = causal_reachability(reorder)
    trace = simulate_streaming_decode(
        prompt_tokens=["Solve:"],
        planned_tokens=["2", "+", "2#2", "=", "4"],
        confidence_steps=[
            {"2": 0.95, "+": 0.91, "2#2": 0.55},
            {"2#2": 0.93, "=": 0.94},
            {"4": 0.97},
        ],
        window_size=3,
        threshold=0.9,
    )

    first_mask = reorder.physical_index_by_logical[1]
    second_mask = reorder.physical_index_by_logical[3]
    return {
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "snapshot_id": SNAPSHOT_ID,
        "challenge_revision": CHALLENGE_REVISION,
        "space_id": SPACE_ID,
        "paper_title": "WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference",
        "upstream_revision": UPSTREAM_REVISION,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "estimated_api_cost_usd": 0.0,
        "cpu_only": True,
        "claim_bindings": CLAIM_BINDINGS,
        "claims": [
            {
                "claim_id": "causal-diffusion-prefix-cache",
                "status": "toy",
                "measured_value": {
                    "uses_strict_lower_triangular_mask": bool(not reachability[0, 1:].any()),
                    "observed_prefix_count": reorder.observed_count,
                },
                "provenance": "CPU topological-order and causal-mask check mirroring WeDLM Section 3 mechanics.",
            },
            {
                "claim_id": "topological-reordering",
                "status": "toy",
                "measured_value": {
                    "physical_tokens": reorder.physical_tokens,
                    "logical_positions": reorder.logical_positions,
                    "first_mask_reachability": reachability[first_mask].tolist(),
                    "second_mask_reachability": reachability[second_mask].tolist(),
                },
                "provenance": "Deterministic fixture showing observed logical futures moved to a causal physical prefix.",
            },
            {
                "claim_id": "streaming-parallel-decoding",
                "status": "toy",
                "measured_value": {
                    "final_tokens": trace.final_tokens,
                    "commits_by_step": [step.committed for step in trace.steps],
                    "max_active_window": max(len(step.active_window) for step in trace.steps),
                },
                "provenance": "CPU streaming decoder simulator for confidence-gated prefix commits and window refill.",
            },
            {
                "claim_id": "gpu-vllm-speedup",
                "status": "unreplicated",
                "measured_value": None,
                "provenance": "No GPU serving benchmark was run; this CPU-only evidence does not reproduce vLLM latency.",
            },
        ],
        "limitations": [
            "The vLLM-served 3x to 10x speedup claim is unreplicated because this bundle runs CPU-only toy scheduling checks.",
            "No Tencent WeDLM model weights are loaded, and no full benchmark datasets are evaluated.",
        ],
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count() or 1,
        },
        "commands": [
            "uv run python -m pytest tests -q",
            "uv run python generate_evidence.py --check",
        ],
    }
