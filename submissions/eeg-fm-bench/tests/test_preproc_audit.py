from __future__ import annotations

import json
from pathlib import Path

from eeg_fm_bench_repro.preproc_audit import run_preproc_audit
from eeg_fm_bench_repro.upstream import ensure_repo_snapshot


def test_preprocessing_executes_pinned_primitives_deterministically(tmp_path: Path) -> None:
    """Catches replacing released preprocessing methods with audit-local math."""
    snapshot = ensure_repo_snapshot(tmp_path / "cache")

    first = run_preproc_audit(snapshot)
    second = run_preproc_audit(snapshot)

    assert first == second
    assert first["claim_id"] == "standardized-preprocessing-reproducibility"
    assert first["kind"] == "numerical_audit"
    assert first["status"] == "verified"
    assert first["deterministic"] is True
    assert first["source_execution"]["methods"] == [
        "EEGDatasetBuilder._generate_window_sample",
        "EEGDatasetBuilder._resample_and_filter",
        "AdftdBuilder.standardize_chs_names",
        "WorkloadBuilder.standardize_chs_names",
    ]
    assert len(first["source_execution"]["sha256"]) == 4
    assert set(first["datasets"]) == {"adftd", "workload"}
    assert all(item["repeat_identical"] for item in first["datasets"].values())
    assert all(item["resampled_sfreq"] == 256.0 for item in first["datasets"].values())
    assert all(item["window_shape"][0] >= 1 for item in first["datasets"].values())
    json.dumps(first)
