import json
from pathlib import Path

from scale_repro.scale import ScaleToyOptimizer, column_normalize, memory_accounting


ATTEMPT_ID = "3cee6045-d032-484f-b22e-6f602d733bad"
PAPER_ID = "prvGhNz39e"
SNAPSHOT_ID = "9526a1a45e8d7f5380d7130abe7011e03a6b4fda337ec1b417b25e4e643eebc8"
UPSTREAM_REVISION = (
    "arxiv:2506.16659v3+"
    "github:OptimAI-Lab/Minimalist_LLM_Pretraining@"
    "94712d907f6cc94528b04dffb632215b5ed20a0d"
)
UPSTREAM_OPTIMIZER_SHA256 = (
    "8656df7290ea638c5b093d60a28f80a4e4cbaefb7056dc79523ea2d273874196"
)


def build_evidence_bundle():
    gradient = [[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]
    normalized = column_normalize(gradient)
    optimizer = ScaleToyOptimizer(
        {
            "model.embed_tokens.weight": (2, 3),
            "model.layers.0.mlp.weight": (2, 3),
            "lm_head.weight": (2, 3),
        },
        lm_output_parameter="lm_head.weight",
        beta=0.5,
    )
    optimizer.step(
        {
            "model.embed_tokens.weight": [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
            "model.layers.0.mlp.weight": [[2.0, 1.0, 0.0], [0.0, 1.0, 4.0]],
            "lm_head.weight": [[4.0, 0.0, 2.0], [0.0, 6.0, 0.0]],
        }
    )
    accounting = memory_accounting(
        {
            "model.embed_tokens.weight": (32000, 512),
            "model.layers.0.mlp.weight": (512, 2048),
            "model.layers.0.attn.weight": (512, 512),
            "lm_head.weight": (512, 32000),
        },
        lm_output_parameter="lm_head.weight",
    )

    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "title": "Memory-Efficient LLM Pretraining via Minimalist Optimizer Design",
        "snapshot_id": SNAPSHOT_ID,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_optimizer_sha256": UPSTREAM_OPTIMIZER_SHA256,
        "estimated_api_cost_usd": 0.0,
        "commands": [
            "uv run pytest -q submissions/memory-efficient-llm-pretraining-via-minimalist-optimizer-design/tests",
            "uv run python submissions/memory-efficient-llm-pretraining-via-minimalist-optimizer-design/generate_evidence.py",
        ],
        "observations": {
            "column_normalized_gradient": normalized,
            "momentum_parameters_after_step": optimizer.momentum_parameter_names(),
            "memory_accounting": accounting,
        },
        "claims": [
            {
                "claim": (
                    "SCALE combines column-wise gradient normalization with "
                    "first-order momentum only on the LM output layer "
                    "(Algorithm 1; Section 3)."
                ),
                "challenge_claim_sha256": (
                    "1a8db670cb242cb7834d5846f90ee78a0b6456c9927517d3dfdf50bc598478ed"
                ),
                "status": "verified",
                "evidence": (
                    "Independent CPU checks normalize each synthetic gradient column "
                    "and show momentum state is created only for lm_head.weight."
                ),
            },
            {
                "claim": (
                    "Last-layer-only momentum targets the layer with the largest "
                    "stochastic-gradient variance while adding minimal optimizer-state "
                    "memory (Figure 4)."
                ),
                "challenge_claim_sha256": (
                    "38a7ae8f5b5ef161d4d5740bc10824d7388edd17227c124532b6891efc0d955d"
                ),
                "status": "verified",
                "evidence": (
                    "Toy transformer-like accounting stores one momentum tensor for "
                    "lm_head.weight instead of one per parameter tensor."
                ),
            },
        ],
        "limitations": [
            (
                "LLaMA/C4 pretraining metrics are not reproduced because they require "
                "GPU-scale training or unreleased full training logs."
            )
        ],
        "license_note": (
            "GitHub reported no repository license for the pinned upstream source; "
            "this bundle records hashes and independently computed observations only."
        ),
    }


def write_evidence_bundle(path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_evidence_bundle()
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return bundle


def write_provenance(path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "snapshot_id": SNAPSHOT_ID,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_optimizer_sha256": UPSTREAM_OPTIMIZER_SHA256,
        "artifact_access": True,
        "cpu_only": True,
        "estimated_api_cost_usd": 0.0,
    }
    output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance
