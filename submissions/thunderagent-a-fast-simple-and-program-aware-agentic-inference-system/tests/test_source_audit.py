from pathlib import Path

from thunderagent_repro.source_audit import audit_source_tree, build_evidence_bundle


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "ThunderAgent"


def test_audit_detects_program_abstraction_scheduler_and_release_paths():
    audit = audit_source_tree(FIXTURE_ROOT)

    assert audit["upstream_revision"].endswith("@7ddc8610270e56d3b109eed8796b3a4360fc67c9")
    assert audit["claims"]["program_abstraction"]["status"] == "verified"
    assert audit["claims"]["program_abstraction"]["observations"]["program_fields"] >= {
        "program_id",
        "backend_url",
        "status",
        "state",
        "context_len",
        "total_tokens",
        "waiting_event",
        "acting_since",
    }
    assert audit["claims"]["program_scheduler"]["status"] == "verified"
    assert audit["claims"]["program_scheduler"]["observations"]["scheduler_methods"] >= {
        "_scheduler_loop",
        "_scheduled_check",
        "_pause_until_safe",
        "_greedy_resume",
        "_resume_program",
    }
    assert audit["claims"]["resource_lifecycle"]["status"] == "toy"
    assert "release_program" in audit["claims"]["resource_lifecycle"]["observations"]["release_paths"]
    assert audit["claims"]["resource_lifecycle"]["observations"]["missing_source_features"] == [
        "No explicit disk resource manager was found in the pinned source.",
        "No explicit port resource manager was found in the pinned source.",
    ]


def test_evidence_bundle_binds_live_claim_hashes_and_source_hashes():
    bundle = build_evidence_bundle(FIXTURE_ROOT)

    assert bundle["attempt_id"] == "72481d5a-6899-4880-94ac-47e5525c0778"
    assert bundle["paper_id"] == "kR4iOTaAOJ"
    assert bundle["snapshot_id"] == "8a76dd68f25c012b70bdd330d35d5b8b73785cf9665b0e886a4d4bd868c88081"
    assert {claim["challenge_claim_sha256"] for claim in bundle["claims"]} == {
        "f3c921a44400a59b56213973efcf334f326cf8f9f3f1ede152eba85119c08faf",
        "030df6bacd99e5a892294e6960768d02e9e6f7b561ec4d24cf44a5d0606ee9f1",
        "733855963a596a93d707e7d5b97f94849c572b0ab5bdcf24c4c215d91c961511",
    }
    assert "ThunderAgent/program/state.py" in bundle["source_hashes"]
    assert "ThunderAgent/scheduler/router.py" in bundle["source_hashes"]
    assert all(len(value) == 64 for value in bundle["source_hashes"].values())
