import importlib.util
import json
from pathlib import Path


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]


def test_artifact_observations_are_pinned_and_counted():
    from terminaltraj_repro.evidence import collect_observations

    observations = collect_observations()

    assert observations["github"]["revision"] == (
        "01305cbf0425b08b41cf8cfc3e30abb0f4953c27"
    )
    assert observations["github"]["repo_license_records"] == 2481
    assert observations["github"]["has_executable_pipeline_code"] is False
    assert observations["hf_dataset"]["revision"] == (
        "5c1823f4a8b9ca0cf02c27d2db52c5b35b53a308"
    )
    assert observations["hf_dataset"]["train_examples"] == 20000
    assert observations["hf_instances"]["files"] == [
        ".gitattributes",
        "5k_instances.tar.gz",
    ]
    assert observations["paper_counts"]["trajectories"] == 50733
    assert observations["paper_counts"]["docker_images_exact"] == 32325


def test_evidence_bundle_binds_selected_claims_and_statuses():
    from terminaltraj_repro.evidence import build_evidence_bundle

    bundle = build_evidence_bundle()

    assert bundle["paper_id"] == "PeFSCRulgy"
    assert bundle["attempt_id"] == "fe586a0f-c0ff-4290-882b-b7fadb2ec2f4"
    assert [claim["challenge_claim_sha256"] for claim in bundle["claims"]] == [
        "39a708c775ac4fbff63c3c664dc7739dfe4b55247d864174215b269db249921f",
        "4908ae61e70d4225466dc1443bdb44a48ef4a6701f5b0f77a39a9fd750203268",
    ]
    assert [claim["status"] for claim in bundle["claims"]] == [
        "toy",
        "unavailable",
    ]
    assert "20,000" in bundle["claims"][1]["summary"]


def test_generate_evidence_writes_deterministic_bundle():
    from terminaltraj_repro.evidence import write_evidence_bundle

    output = SUBMISSION_ROOT / "evidence" / "bundle.json"
    first = write_evidence_bundle(output)
    second = write_evidence_bundle(output)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert first == second == persisted
    assert persisted["generated_at"] == "2026-07-29T03:10:00+00:00"


def test_space_assets_exist_and_import():
    pages = sorted((SUBMISSION_ROOT / "pages").glob("*.md"))
    total_characters = sum(len(path.read_text(encoding="utf-8").strip()) for path in pages)

    assert total_characters >= 200

    spec = importlib.util.spec_from_file_location(
        "terminaltraj_space_app", SUBMISSION_ROOT / "app.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "demo")
