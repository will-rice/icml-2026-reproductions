from pathlib import Path

from agent_primitives_repro.evidence import (
    ATTEMPT_ID,
    CLAIMS,
    arxiv_audit,
    build_evidence_bundle,
    kv_cache_shape_valid,
    organizer_select,
    plan_and_execute,
    select_vote,
    simulate_review,
    write_evidence,
)


def test_claim_bindings_and_statuses_match_approved_strategy():
    bundle = build_evidence_bundle(
        arxiv_text="Review Voting Selection Planning Execution KV-cache Organizer accuracy latency",
        arxiv_files=["main.tex", "appendix_e.tex", "figures/primitive_pool.pdf"],
    )

    assert [claim["challenge_claim_sha256"] for claim in bundle["claims"]] == [
        "8115a999582028cd40604b3d6dd9ee69546547d4babd00b02e8c1abb1b719ff1",
        "5b9aaefaf80d29d999f70165b70dde7f2e9cb90f080dffee3c41a8f4d17228b5",
        "3ff1e2007a7b9f313467db56636bd1e60598d8dfad968e2cec21aeff157168bc",
        "7eadb6ed2e71e12ea036c70a7f6c4e4ef52c30861df2d99f613d30d1b0012129",
        "67bd705462e640c09cc50efc1020c8eb02746643ecc5cfa913ef7f6d80da91a1",
    ]
    assert [claim["status"] for claim in bundle["claims"]] == [
        "toy",
        "toy",
        "toy",
        "inconclusive",
        "inconclusive",
    ]
    assert len(CLAIMS) == 5


def test_primitive_simulations_are_deterministic_and_observable():
    review = simulate_review("draft answer", ["cite source", "tighten math"])
    vote = select_vote(
        ["direct solver", "tool solver", "critic solver"],
        {"critic solver": 0.8, "direct solver": 0.4, "tool solver": 0.8},
    )
    plan = plan_and_execute("answer a query", ["retrieve", "reason", "verify"])

    assert review == {
        "initial": "draft answer",
        "critiques": ["cite source", "tighten math"],
        "revision": "draft answer | review-1: cite source | review-2: tighten math",
        "rounds": 2,
    }
    assert vote == {"selected": "critic solver", "score": 0.8, "tie_break": "lexicographic"}
    assert plan["trace"] == ["plan:retrieve", "execute:reason", "verify:verify"]
    assert plan["completed"] is True


def test_kv_cache_shape_validation_rejects_invalid_dimensions():
    assert kv_cache_shape_valid(layers=2, tokens=16, heads=4, dim=64) == {
        "valid": True,
        "shape": [2, 2, 16, 4, 64],
        "elements": 16384,
    }
    assert kv_cache_shape_valid(layers=0, tokens=16, heads=4, dim=64)["valid"] is False


def test_organizer_selects_by_overlap_then_name():
    pool = [
        {"name": "review", "tags": ["critique", "revise"]},
        {"name": "planning", "tags": ["decompose", "execute"]},
        {"name": "voting", "tags": ["rank", "select"]},
    ]

    assert organizer_select("decompose and execute the solution", pool) == {
        "selected": "planning",
        "overlap": 2,
        "available": ["planning", "review", "voting"],
    }


def test_arxiv_audit_records_terms_and_asset_names():
    audit = arxiv_audit(
        "Review Voting and Selection Planning and Execution KV-cache Organizer Appendix E",
        ["main.tex", "sections/method.tex", "appendix_e.tex"],
    )

    assert audit["terms_found"] == {
        "review": True,
        "voting_selection": True,
        "planning_execution": True,
        "kv_cache": True,
        "organizer": True,
        "appendix_e": True,
    }
    assert audit["source_files"] == ["appendix_e.tex", "main.tex", "sections/method.tex"]


def test_write_evidence_creates_bundle_and_report(tmp_path: Path):
    bundle_path, report_path = write_evidence(
        tmp_path,
        arxiv_text="Review Voting Selection Planning Execution KV-cache Organizer",
        arxiv_files=["main.tex"],
    )

    assert bundle_path.read_text().count(ATTEMPT_ID) == 1
    assert report_path.read_text().startswith("# Agent Primitives Reproduction Evidence")
