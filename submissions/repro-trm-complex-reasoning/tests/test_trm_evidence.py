import importlib.util
import json
from pathlib import Path


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]


def test_artifact_observations_are_revision_pinned():
    from trm_complex_reasoning_repro.evidence import collect_observations

    observations = collect_observations()

    assert observations["paper_source"]["arxiv_id"] == "2602.08498"
    assert observations["paper_source"]["source_sha256"] == (
        "6c01d28b97ecbc850b2813c3e0af85ce2b2ec57e6fc9173ab6a0e5ac3dfb3f7f"
    )
    assert observations["github"]["revision"] == (
        "82ac3778aaba9cf63b237b3db434dc2ba813ef29"
    )
    assert observations["github"]["license"] == "MIT"
    assert observations["github"]["me2_dimensions"] == [
        "Macro-Efficiency",
        "Macro-Effectiveness",
        "Micro-Efficiency",
        "Micro-Effectiveness",
    ]
    assert observations["github"]["dag_steps"] == [
        "partition",
        "build_dag",
        "merge_view",
    ]


def test_released_dag_package_builds_branch_and_merge_graph():
    from trm_complex_reasoning_repro.evidence import run_released_dag_smoke

    dag = run_released_dag_smoke()

    assert dag["raw_actions"] == {
        "0": "root",
        "1": "continue",
        "2": "backtrack",
        "3": "merge",
    }
    assert dag["raw_parents"]["3"] == [1, 2]
    assert dag["merged_parent_for_merge"] == [1, 2]
    assert dag["usage_calls"] == 3


def test_preference_dataset_and_reward_model_artifacts_are_observed():
    from trm_complex_reasoning_repro.evidence import collect_observations

    observations = collect_observations()

    assert observations["trm_preference"]["revision"] == (
        "0d0752035ea0e8f7d5c28e1e7a7d8f27e2e45d61"
    )
    assert observations["trm_preference"]["files"]["TRM-preference-test.json"] == 28973005
    assert observations["trm_preference"]["files"]["TRM-preference-train.json"] == 2046546117
    assert observations["trm_preference"]["test_examples"] == 1500
    assert observations["trm_preference"]["sample_keys"] == [
        "chosen",
        "prompt",
        "rejected",
    ]
    assert observations["trm_model"]["revision"] == (
        "b84f02bf6b4227675284538a4deb82822371ebbd"
    )
    assert observations["trm_model"]["architecture"] == "LlamaForSequenceClassification"
    assert observations["trm_model"]["pipeline_tag"] == "text-classification"
    assert observations["trm_model"]["card_tags"] == [
        "generated_from_trainer",
        "trl",
        "reward-trainer",
    ]
    assert observations["trm_model"]["eval_accuracy"] == 0.8835227272727273
    assert observations["training_script"]["train_file_arg"] == "TRM-preference-train.json"
    assert observations["training_script"]["validation_file_arg"] == "TRM-preference-test.json"
    assert observations["training_script"]["metric_for_best_model"] == "accuracy"
    assert observations["training_script"]["train_py_present"] is False


def test_evidence_bundle_binds_claims_and_marks_training_code_partial():
    from trm_complex_reasoning_repro.evidence import build_evidence_bundle

    bundle = build_evidence_bundle()

    assert bundle["paper_id"] == "IMFgiWw4jd"
    assert bundle["attempt_id"] == "b4432d6d-0e2d-413d-a4dd-28720b0d335b"
    assert [claim["challenge_claim_sha256"] for claim in bundle["claims"]] == [
        "3e23512497a9642162ac69ed2bd7cbcef00962324c530b34e93bc60475893862",
        "6ccf7ef3305d4a60c6456012a0ed4fdd4d59ad28cc2c7dca5f4c0583a848a370",
        "db021919e89a8a7b8485b63b85c4e2d00a1ff7477d23fbfaf8310b1a09832d92",
    ]
    assert [claim["status"] for claim in bundle["claims"]] == [
        "verified",
        "verified",
        "toy",
    ]
    assert "train.py is not present" in bundle["claims"][2]["summary"]


def test_generate_evidence_writes_deterministic_bundle():
    from trm_complex_reasoning_repro.evidence import write_evidence_bundle

    output = SUBMISSION_ROOT / "evidence" / "bundle.json"
    first = write_evidence_bundle(output)
    second = write_evidence_bundle(output)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert first == second == persisted
    assert persisted["generated_at"] == "2026-07-29T04:05:00+00:00"


def test_space_assets_exist_and_import():
    pages = sorted((SUBMISSION_ROOT / "pages").glob("*.md"))
    total_characters = sum(len(path.read_text(encoding="utf-8").strip()) for path in pages)

    assert total_characters >= 200

    spec = importlib.util.spec_from_file_location(
        "trm_complex_reasoning_space_app", SUBMISSION_ROOT / "app.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "demo")


def test_space_readme_contains_required_metadata():
    readme = (SUBMISSION_ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.startswith("---\n")
    assert "sdk: gradio" in readme
    assert "app_file: app.py" in readme
    assert "  - icml2026-repro\n" in readme
    assert "  - paper-IMFgiWw4jd\n" in readme
