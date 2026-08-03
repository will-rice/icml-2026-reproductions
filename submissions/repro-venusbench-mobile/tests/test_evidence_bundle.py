from pathlib import Path

import pytest

from venusbench_mobile_repro.evidence import (
    EXPECTED_UPSTREAM_COMMIT,
    build_evidence_bundle,
    load_evidence_bundle,
    render_summary_markdown,
)


def test_build_evidence_bundle_records_released_artifact_counts():
    bundle = build_evidence_bundle(
        source_root=Path("/tmp/icml-venusbench-inspect-fz4OWv"),
        command_log=["pytest tests"],
    )

    artifacts = bundle["artifact_observations"]
    assert bundle["paper_id"] == "coHiGZOFtS"
    assert bundle["upstream"]["commit"] == EXPECTED_UPSTREAM_COMMIT
    assert artifacts["task_instance_goal_count"] == 189
    assert artifacts["metadata_task_count"] == 189
    assert artifacts["androidworld_baseline_task_count"] == 116
    assert artifacts["readme_claimed_primary_tasks"] == 149
    assert artifacts["readme_claimed_stability_variants"] == 80
    assert artifacts["readme_claimed_apps"] == 27


def test_build_evidence_bundle_verifies_pudam_and_hybrid_verifier_paths():
    bundle = build_evidence_bundle(source_root=Path("/tmp/icml-venusbench-inspect-fz4OWv"))

    artifacts = bundle["artifact_observations"]
    assert artifacts["pudam_keys"] == ["p", "u", "d", "a", "m"]
    assert artifacts["pudam_dimensions"] == [
        "Perception",
        "Understanding",
        "Decision",
        "Action",
        "Memory",
    ]
    assert artifacts["evaluation_method_counts"]["p"] > 0
    assert artifacts["evaluation_method_counts"]["m"] > 0
    assert artifacts["hybrid_verification_files"] == [
        "README.md",
        "android_world/policy/verification.py",
        "android_world/suite_utils.py",
    ]


def test_build_evidence_bundle_records_stability_modes_and_discrepancy():
    bundle = build_evidence_bundle(source_root=Path("/tmp/icml-venusbench-inspect-fz4OWv"))

    artifacts = bundle["artifact_observations"]
    assert artifacts["stability_base_subset_count"] == 20
    assert artifacts["stability_instruction_variation_count"] == 40
    assert artifacts["stability_total_execution_modes"] == 5
    assert artifacts["stability_modes_from_readme"] == [
        "Original",
        "Question Variation",
        "Chinese",
        "Mobile Dark mode",
        "Pad mode",
    ]
    assert artifacts["min_max_setting_variant_evidence"] == "not_found_in_released_scripts"


def test_evidence_bundle_contains_four_target_claims_with_statuses():
    bundle = load_evidence_bundle()

    assert len(bundle["target_claims"]) == 4
    statuses = {claim["status"] for claim in bundle["target_claims"]}
    assert statuses <= {"verified", "partial"}
    assert any(claim["status"] == "partial" for claim in bundle["target_claims"])
    assert all(claim["challenge_claim_sha256"] for claim in bundle["target_claims"])


def test_build_evidence_bundle_rejects_wrong_upstream_commit(tmp_path):
    with pytest.raises(ValueError, match="upstream commit"):
        build_evidence_bundle(source_root=tmp_path)


def test_render_summary_markdown_surfaces_partial_stability_claim():
    markdown = render_summary_markdown(load_evidence_bundle())

    assert "# VenusBench-Mobile Reproduction Evidence" in markdown
    assert "claim_4_stability_protocol: partial" in markdown
    assert "not_found_in_released_scripts" in markdown
