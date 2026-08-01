import json
from pathlib import Path

from neural_thickets_repro.evidence import (
    CLAIMS,
    UPSTREAM_PINS,
    audit_artifacts,
    build_evidence_bundle,
    simulate_neural_thicket,
    write_evidence,
)


def test_readme_declares_required_space_tags():
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text(encoding="utf-8")

    assert readme.startswith("---\n")
    metadata = readme.split("---\n", 2)[1]
    assert "icml2026-repro" in metadata
    assert "paper-92oF5bU4cU" in metadata


def test_artifact_audit_detects_randopt_selection_and_ensembling():
    audit = audit_artifacts(
        {
            "README.md": "Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct PPO GRPO ES Project Page",
            "randopt.py": "random perturbation top_k Counter majority vote population_size sigma",
            "core/engines.py": "launch_engines cleanup_engines vllm model_path",
            "data_handlers/__init__.py": "gsm8k math500 countdown uspto rocstories gqa",
            "baselines/README.md": "PPO GRPO ES verl Qwen Llama OLMo",
            "simple_1D_signals_expts/toy.py": "needle haystack density diversity",
        }
    )

    assert audit["randopt_algorithm"]["status"] == "present"
    assert audit["majority_vote"]["status"] == "present"
    assert audit["model_scale_family"]["status"] == "present"
    assert audit["baseline_protocols"]["status"] == "present"
    assert audit["toy_1d_experiment"]["status"] == "present"


def test_toy_simulation_reproduces_qualitative_density_and_diversity_mechanisms():
    result = simulate_neural_thicket(seed=7, population_size=96)

    assert result["large_model_density"] > result["small_model_density"]
    assert result["ensemble_accuracy"] >= result["best_single_accuracy"]
    assert result["population_curve"]["96"] >= result["population_curve"]["24"]
    assert len(result["specialty_counts"]) >= 3


def test_bundle_records_claims_and_marks_scale_metrics_unavailable_without_raw_results():
    bundle = build_evidence_bundle(
        {
            "README.md": "Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct",
            "randopt.py": "random perturbation top_k Counter majority vote population_size sigma",
            "core/engines.py": "launch_engines cleanup_engines",
            "data_handlers/__init__.py": "gsm8k math500 countdown uspto rocstories gqa",
            "baselines/README.md": "PPO GRPO ES",
            "simple_1D_signals_expts/toy.py": "density diversity",
        },
        raw_result_artifacts={},
    )

    assert bundle["attempt_id"] == "228a446e-f3c6-4ee1-8d80-28b6d1226520"
    assert bundle["paper_id"] == "92oF5bU4cU"
    assert bundle["snapshot_id"] == "cd566b1fc072468cea13824a2382d9be6916bd5ffb684b5affcbfa814f753528"
    assert bundle["upstream_pins"] == UPSTREAM_PINS
    assert [result["claim_sha256"] for result in bundle["claim_results"]] == [
        claim["challenge_claim_sha256"] for claim in CLAIMS
    ]
    assert bundle["claim_results"][0]["status"] == "toy"
    assert bundle["claim_results"][1]["status"] == "unavailable"
    assert bundle["claim_results"][3]["status"] == "unavailable"
    json.dumps(bundle)


def test_write_evidence_outputs_pre_commit_clean_json(tmp_path):
    output_path = tmp_path / "evidence" / "bundle.json"

    write_evidence(output_path)

    assert output_path.read_bytes().endswith(b"\n")
