import json
import math
import subprocess
import sys
from pathlib import Path


def _matrix_column_norms(matrix):
    return [
        math.sqrt(sum(row[column] ** 2 for row in matrix))
        for column in range(len(matrix[0]))
    ]


def _matrix_row_norms(matrix):
    return [math.sqrt(sum(value**2 for value in row)) for row in matrix]


def test_column_normalization_uses_columns_not_rows():
    import scale_repro.scale as scale

    assert hasattr(scale, "column_normalize")
    assert hasattr(scale, "row_normalize")

    gradient = [[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]

    column_normalized = scale.column_normalize(gradient)
    row_normalized = scale.row_normalize(gradient)

    assert all(
        math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12)
        for norm in _matrix_column_norms(column_normalized)
    )
    assert not all(
        math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12)
        for norm in _matrix_row_norms(column_normalized)
    )
    assert column_normalized != row_normalized


def test_scale_step_tracks_momentum_only_for_lm_output_parameter():
    import scale_repro.scale as scale

    assert hasattr(scale, "ScaleToyOptimizer")

    optimizer = scale.ScaleToyOptimizer(
        parameter_shapes={
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

    assert optimizer.momentum_parameter_names() == ["lm_head.weight"]
    assert "lm_head.weight" in optimizer.state
    assert "momentum" in optimizer.state["lm_head.weight"]
    assert "momentum" not in optimizer.state["model.embed_tokens.weight"]
    assert "momentum" not in optimizer.state["model.layers.0.mlp.weight"]


def test_last_layer_momentum_state_uses_less_memory_than_full_model_momentum():
    import scale_repro.scale as scale

    assert hasattr(scale, "memory_accounting")

    inventory = {
        "model.embed_tokens.weight": (32000, 512),
        "model.layers.0.mlp.weight": (512, 2048),
        "model.layers.0.attn.weight": (512, 512),
        "lm_head.weight": (512, 32000),
    }

    accounting = scale.memory_accounting(inventory, lm_output_parameter="lm_head.weight")

    assert accounting["last_layer_momentum_bytes"] < accounting["full_momentum_bytes"]
    assert accounting["last_layer_momentum_bytes"] == accounting["parameter_bytes"]["lm_head.weight"]
    assert 0.0 < accounting["last_layer_fraction_of_full_momentum"] < 1.0


def test_evidence_bundle_is_bound_to_scale_claims_and_upstream(tmp_path):
    import scale_repro.evidence as evidence

    assert hasattr(evidence, "build_evidence_bundle")
    assert hasattr(evidence, "write_evidence_bundle")

    bundle = evidence.build_evidence_bundle()
    output = tmp_path / "results.json"

    assert evidence.write_evidence_bundle(output) == bundle
    assert json.loads(Path(output).read_text()) == bundle
    assert bundle["attempt_id"] == "3cee6045-d032-484f-b22e-6f602d733bad"
    assert bundle["paper_id"] == "prvGhNz39e"
    assert bundle["estimated_api_cost_usd"] == 0.0
    assert bundle["upstream_revision"] == (
        "arxiv:2506.16659v3+"
        "github:OptimAI-Lab/Minimalist_LLM_Pretraining@"
        "94712d907f6cc94528b04dffb632215b5ed20a0d"
    )
    assert [claim["challenge_claim_sha256"] for claim in bundle["claims"]] == [
        "1a8db670cb242cb7834d5846f90ee78a0b6456c9927517d3dfdf50bc598478ed",
        "38a7ae8f5b5ef161d4d5740bc10824d7388edd17227c124532b6891efc0d955d",
    ]
    assert {claim["status"] for claim in bundle["claims"]} <= {
        "verified",
        "toy",
        "falsified",
        "inconclusive",
        "unavailable",
    }
    assert all(
        "LLaMA/C4 pretraining metrics are not reproduced" in limitation
        for limitation in bundle["limitations"]
    )


def test_generate_evidence_script_writes_expected_files():
    script = Path(__file__).resolve().parent.parent / "generate_evidence.py"

    subprocess.run([sys.executable, str(script)], check=True)

    root = script.parent
    assert (root / "evidence" / "results.json").is_file()
    assert (root / "evidence" / "provenance.json").is_file()
