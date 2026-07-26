from pathlib import Path
from recurrent_sampler_repro.evidence import run_pipeline


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_claim_bindings_and_manifest():
    project_root = get_project_root()
    manifest = run_pipeline(project_root)["manifest"]

    assert manifest["attempt_id"] == "534db42c-5b16-4f00-9a7d-a47056fc9dd4"
    assert manifest["paper_id"] == "h7WBYYJF1Q"
    assert manifest["python_requirement"] == ">=3.10"

    cmds = manifest["commands"]
    assert cmds["evidence_generation"] == "uv run --locked --project submissions/efficient-parallel-samplers-for-recurrent-depth-models python -m recurrent_sampler_repro.evidence --project-root submissions/efficient-parallel-samplers-for-recurrent-depth-models"
    assert cmds["test_suite"] == "uv run --locked --project submissions/efficient-parallel-samplers-for-recurrent-depth-models python -m pytest submissions/efficient-parallel-samplers-for-recurrent-depth-models/tests"

    inputs = manifest["inputs"]
    assert "arxiv_submission.tex" in inputs
    assert "raven_modeling_minimal.py" in inputs
    assert "LICENSE" in inputs
    assert "ATTRIBUTION.md" in inputs
    assert inputs["ATTRIBUTION.md"]["sha256"] == "79775b50c72988b90eae75ef87e9d4df9dbd0bfceefaed60b398656a88d8a735"

    outputs = manifest["outputs"]
    assert "evidence/claim-1-wavefront.json" in outputs
    assert outputs["evidence/claim-1-wavefront.json"]["sha256"] is not None
    assert "evidence/manifest.json" in outputs
    assert outputs["evidence/manifest.json"]["sha256"] is None
    assert outputs["evidence/manifest.json"]["unhashed_reason"] == "Self-referential manifest copy"
