import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from generate_evidence import (
    CLAIM_BINDINGS,
    PROJECT_PAGE_SHA256,
    audit_augmentations,
    build_evidence_bundle,
    main,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_audit_counts_severity_binary_categories_and_corrupted_settings():
    audit = audit_augmentations(
        aug_source=FIXTURES / "aug.py",
        readme_source=FIXTURES / "README.md",
    )

    assert audit["severity_based_count"] == 42
    assert audit["binary_count"] == 7
    assert audit["total_augmentation_count"] == 49
    assert audit["severity_category_count_excluding_binary"] == 9
    assert audit["corrupted_settings_per_model_dataset"] == 133
    assert audit["severity_levels_used"] == [1, 3, 5]
    assert audit["categories"]["Blur"] == [
        "gaussian_blur",
        "motion_blur",
        "defocus_blur",
        "glass_blur",
        "zoom_blur",
    ]


def test_bundle_binds_exact_claims_and_does_not_overclaim_glass_blur():
    bundle = build_evidence_bundle(
        repo_dir=FIXTURES,
        project_page=FIXTURES / "project.html",
        project_app=FIXTURES / "app.js",
    )

    assert bundle["paper_id"] == "HwXyyvK7ZJ"
    assert bundle["upstream"]["github_commit"] == (
        "8bc793d1649e574e000f91c59cb6ce7432c95073"
    )
    assert bundle["upstream"]["project_page_sha256"] == PROJECT_PAGE_SHA256
    assert [claim["challenge_claim_sha256"] for claim in bundle["claims"]] == [
        binding["challenge_claim_sha256"] for binding in CLAIM_BINDINGS
    ]

    statuses = {claim["claim_index"]: claim["status"] for claim in bundle["claims"]}
    assert statuses[1] == "verified"
    assert statuses[2] == "verified"
    assert statuses[3] in {"toy", "inconclusive", "unavailable"}
    assert statuses[3] != "verified"

    glass_claim = bundle["claims"][2]
    assert "paper-reported value was not used as reproduced evidence" in glass_claim["evidence"]
    assert glass_claim["observations"]["exact_8_1_primary_value_found"] is False


def test_generated_bundle_is_json_serializable(tmp_path):
    bundle = build_evidence_bundle(
        repo_dir=FIXTURES,
        project_page=FIXTURES / "project.html",
        project_app=FIXTURES / "app.js",
    )
    out = tmp_path / "bundle.json"
    out.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["evidence_schema"] == "icml-repro-v1"
    assert len(loaded["claims"]) == 3


def test_build_evidence_bundle_is_deterministic():
    first = build_evidence_bundle(
        repo_dir=FIXTURES,
        project_page=FIXTURES / "project.html",
        project_app=FIXTURES / "app.js",
    )
    second = build_evidence_bundle(
        repo_dir=FIXTURES,
        project_page=FIXTURES / "project.html",
        project_app=FIXTURES / "app.js",
    )

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_cli_writes_json_with_final_newline(tmp_path):
    out = tmp_path / "bundle.json"

    assert main(
        [
            "--repo-dir",
            str(FIXTURES),
            "--project-page",
            str(FIXTURES / "project.html"),
            "--project-app",
            str(FIXTURES / "app.js"),
            "--output",
            str(out),
        ]
    ) == 0

    assert out.read_bytes().endswith(b"\n")


def test_space_source_includes_scoring_pages():
    pages = PROJECT / "pages"
    markdown = sorted(path for path in pages.glob("*.md") if path.is_file())
    substantive_characters = sum(
        len(path.read_text(encoding="utf-8").strip()) for path in markdown
    )

    assert substantive_characters >= 200


def test_readme_uses_huggingface_emoji_metadata():
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    metadata = readme.split("---", 2)[1]

    assert "emoji: test_tube" not in metadata
    assert "emoji: " in metadata
