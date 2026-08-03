"""Tests for the worker proposal contract."""

from __future__ import annotations

from pathlib import Path

from rbench_repro.cli import build_worker_proposal
from rbench_repro.model import sha256_bytes


def test_worker_proposal_requests_controller_validation():
    proposal = build_worker_proposal(b"evidence\n", "a" * 40, "b" * 40)
    assert proposal["requested_action"] == "controller_validation"
    assert proposal["external_mutations"] == []
    assert proposal["source_commit"] == "a" * 40
    assert proposal["source_tree"] == "b" * 40


def test_worker_proposal_is_scoped_and_non_mutating():
    project_root = Path(__file__).resolve().parents[1]
    evidence_path = project_root / "evidence" / "results.json"
    if not evidence_path.is_file():
        # Bundle not yet committed; test with synthetic data
        evidence_bytes = b"evidence\n"
    else:
        evidence_bytes = evidence_path.read_bytes()
    proposal = build_worker_proposal(
        evidence_bytes,
        "a" * 40,
        "b" * 40,
    )
    assert proposal["paper_id"] == "p5QSlnwume"
    assert proposal["attempt_id"] == "8c21f2dc-a357-422e-9c1b-79a4d417e3dc"
    assert proposal["requested_action"] == "controller_validation"
    assert proposal["external_mutations"] == []
    assert len(proposal["source_commit"]) == 40
    assert len(proposal["source_tree"]) == 40
    assert proposal["evidence_sha256"] == sha256_bytes(evidence_bytes)
