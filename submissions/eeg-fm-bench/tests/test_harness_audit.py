from __future__ import annotations

import json
from pathlib import Path

from eeg_fm_bench_repro.harness_audit import run_harness_audit
from eeg_fm_bench_repro.upstream import ensure_repo_snapshot


def test_harness_proves_flags_paths_and_three_cpu_steps(tmp_path: Path) -> None:
    """Catches synthetic smoke runs that do not establish upstream harness support."""
    snapshot = ensure_repo_snapshot(tmp_path / "cache")

    first = run_harness_audit(snapshot)
    second = run_harness_audit(snapshot)

    assert first == second
    assert first["claim_id"] == "three-strategy-evaluation-harness"
    assert first["kind"] == "hybrid_structural_smoke_audit"
    assert first["status"] == "partial"
    assert first["released_execution"] is False
    assert first["smoke_model"] == "audit_local_tiny_harness"
    assert first["upstream_contract"] == {
        "freeze_encoder_declared": True,
        "freeze_encoder_honored": True,
        "multitask_declared": True,
        "multitask_mixed_loader": True,
        "multitask_training_branch": True,
    }
    assert list(first["strategies"]) == [
        "frozen-backbone-single-task",
        "full-parameter-single-task",
        "full-parameter-multi-task",
    ]
    assert first["strategies"]["frozen-backbone-single-task"][
        "encoder_changed"
    ] is False
    assert first["strategies"]["full-parameter-single-task"]["encoder_changed"] is True
    multi = first["strategies"]["full-parameter-multi-task"]
    assert multi["encoder_changed"] is True
    assert multi["dataset_heads_exercised"] == ["dataset_a", "dataset_b"]
    assert all(item["finite_loss"] for item in first["strategies"].values())
    assert len(first["source_sha256"]) >= 3
    json.dumps(first)
