from pathlib import Path

from skill_neologisms_repro.evidence import build_evidence_bundle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_ROOT = Path("/tmp/icml-skill-neologisms-upstream-path").read_text(
    encoding="utf-8"
).strip()


def test_generated_bundle_records_current_attempt_identity():
    bundle = build_evidence_bundle(source_root=Path(UPSTREAM_ROOT))

    assert bundle["attempt_id"] == "6ee240d6-4363-419b-a6e8-d05aae509de4"
    assert bundle["owner"] == "codex-paper-owner-03"
    assert (
        bundle["snapshot_id"]
        == "31492eaeed30c533df53d18a5b536207a39cda717423ad7ff0b1ad2b017a82bc"
    )


def test_space_readme_declares_challenge_tags():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.startswith("---\n")
    assert "sdk: gradio\n" in readme
    assert "  - icml2026-repro\n" in readme
    assert "  - paper-5VgZUEpK6W\n" in readme
