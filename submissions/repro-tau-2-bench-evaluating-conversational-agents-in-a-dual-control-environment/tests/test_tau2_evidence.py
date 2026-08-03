from pathlib import Path

from tau2_bench_repro.evidence import build_evidence, resolve_upstream_root


ROOT = Path(__file__).resolve().parents[1]


def test_build_evidence_verifies_selected_claims_from_pinned_tau2_source() -> None:
    evidence = build_evidence(resolve_upstream_root(ROOT))

    assert evidence["paper_id"] == "OC2z7iSQKa"
    assert evidence["upstream"]["commit"] == "1d244f5dca42944b67a379b44bfeb9f5748f189d"

    statuses = {claim["claim_id"]: claim["status"] for claim in evidence["claims"]}
    assert statuses == {
        "dual_control_shared_state": "verified",
        "telecom_artifact_counts": "verified",
        "compositional_task_generator": "verified",
    }


def test_telecom_counts_match_challenge_claim() -> None:
    evidence = build_evidence(resolve_upstream_root(ROOT))
    count_claim = next(
        claim
        for claim in evidence["claims"]
        if claim["claim_id"] == "telecom_artifact_counts"
    )

    assert count_claim["observations"] == {
        "agent_tool_count": 13,
        "user_tool_count": 30,
        "base_task_count": 114,
        "full_task_count": 2285,
    }


def test_evidence_records_hashes_for_source_files_used() -> None:
    evidence = build_evidence(resolve_upstream_root(ROOT))
    source_hashes = evidence["upstream"]["source_hashes"]

    assert set(source_hashes) >= {
        "src/tau2/domains/telecom/environment.py",
        "src/tau2/domains/telecom/tools.py",
        "src/tau2/domains/telecom/user_tools.py",
        "src/tau2/domains/telecom/tasks/create_tasks.py",
        "data/tau2/domains/telecom/split_tasks.json",
        "data/tau2/domains/telecom/tasks_full.json",
    }
    assert all(len(value) == 64 for value in source_hashes.values())
