from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "generate_evidence.py"


def load_generator():
    assert GENERATOR_PATH.exists(), "generate_evidence.py must provide the evidence generator"
    spec = importlib.util.spec_from_file_location("rgr_generate_evidence", GENERATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_records_arxiv_v2_and_conservative_statuses(tmp_path: Path):
    """Catches missing arXiv provenance or unsupported training claims."""
    module = load_generator()
    bundle = module.build_evidence_bundle(output_path=tmp_path / "bundle.json")

    assert json.loads((tmp_path / "bundle.json").read_text()) == bundle
    assert bundle["paper_id"] == "AfqsNFzJcs"
    assert bundle["attempt_id"] == "9db24b0e-865b-43ab-b5a9-204b7f8a4843"
    assert bundle["upstream"]["arxiv_id"] == "2511.12344v2"
    assert len(bundle["upstream"]["source_sha256"]) == 64
    assert bundle["estimated_paid_api_cost_usd"] == 0.0

    statuses = {claim["claim_sha256"]: claim["status"] for claim in bundle["claims"]}
    assert statuses["5c0d4622dc06cf2d2dc4ccc24627cfd9e4a2776441d9039d8546733a71f25566"] == "toy"
    assert statuses["c8f80b119663d66a805f4e39fb1f378f72edc3e3a0e825c1f0baa574c899344f"] == "toy"
    assert statuses["3ac159e9611c46f76f06a1a093e676d0d26bfa21efd9c9f3c5a099544e0f4a1d"] in {
        "inconclusive",
        "unavailable",
    }


def test_source_audit_finds_algorithm_components_and_benchmark_headers(tmp_path: Path):
    """Catches evidence that never inspects the TeX source."""
    module = load_generator()
    bundle = module.build_evidence_bundle(output_path=tmp_path / "bundle.json")

    source = bundle["observations"]["source_audit"]
    assert source["algorithm_components_present"] is True
    assert source["rubric_reward_equations"] >= 2
    assert source["benchmark_header_count"] == 12
    assert source["domain_headers"] == ["Math", "Physics", "Chemistry", "General"]


def test_toy_exploration_assessment_branches_on_failed_rubrics(tmp_path: Path):
    """Catches inverted exploration-assessment branching."""
    module = load_generator()
    bundle = module.build_evidence_bundle(output_path=tmp_path / "bundle.json")
    toy = bundle["observations"]["toy_exploration_assessment"]

    assert toy["all_criteria_satisfied"]["uses_off_policy_refinement"] is False
    assert toy["one_failed_criterion"]["uses_off_policy_refinement"] is True
    assert toy["one_failed_criterion"]["failed_criteria"] == ["process_2"]
