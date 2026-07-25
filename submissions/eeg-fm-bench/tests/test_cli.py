from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from eeg_fm_bench_repro.upstream import ensure_paper_pdf, ensure_repo_snapshot
from eeg_fm_bench_repro.cli import _claim

CLAIMS = [
    "fourteen-dataset-ten-paradigm-curation",
    "standardized-preprocessing-reproducibility",
    "three-strategy-evaluation-harness",
]


def _run(cache: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "eeg_fm_bench_repro.cli",
            "--cache-dir",
            str(cache),
            "--output-dir",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_emits_byte_identical_schema_complete_bundle(tmp_path: Path) -> None:
    """Catches nondeterministic bundles or missing claim/provenance boundaries."""
    cache = tmp_path / "cache"
    ensure_repo_snapshot(cache)
    ensure_paper_pdf(cache)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = _run(cache, first_output)
    second = _run(cache, second_output)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_json = (first_output / "results.json").read_bytes()
    first_csv = (first_output / "measurements.csv").read_bytes()
    assert first_json == (second_output / "results.json").read_bytes()
    assert first_csv == (second_output / "measurements.csv").read_bytes()

    bundle = json.loads(first_json)
    assert bundle["schema_version"] == 1
    assert bundle["scope"] == "released_artifact_audit_not_leaderboard_reproduction"
    assert [claim["claim_id"] for claim in bundle["claims"]] == CLAIMS
    assert all(
        {"claim_id", "kind", "status", "observations", "thresholds"}
        <= claim.keys()
        for claim in bundle["claims"]
    )
    assert bundle["unavailable_claims"]
    assert all(item["status"] == "unavailable" for item in bundle["unavailable_claims"])
    assert bundle["provenance"]["inputs"]["repository"]["revision"].startswith("325398d")
    assert set(bundle["environment"]) == {"numpy", "python", "torch"}

    rows = list(csv.DictReader(first_csv.decode().splitlines()))
    assert [row["claim_id"] for row in rows] == CLAIMS
    assert all(row["status"] in {"verified", "partial", "inconclusive"} for row in rows)


def test_cli_rejects_unknown_arguments() -> None:
    """Catches silently ignored CLI arguments that make reruns ambiguous."""
    result = subprocess.run(
        [sys.executable, "-m", "eeg_fm_bench_repro.cli", "--not-a-real-option"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr


def test_claim_status_checks_observations_not_self_reported_status() -> None:
    """Catches a stale verified label surviving failed preprocessing thresholds."""
    record = {
        "claim_id": "standardized-preprocessing-reproducibility",
        "kind": "numerical_audit",
        "status": "verified",
        "deterministic": False,
        "primitive_backend": "audit-local",
        "datasets": {
            "one": {
                "repeat_identical": False,
                "resampled_sfreq": 128.0,
                "all_values_finite": True,
                "window_shape": [0, 19, 1280],
                "channel_count": 19,
                "window_seconds": 10,
            }
        },
    }
    claim = _claim(
        record,
        {
            "datasets_minimum": 2,
            "repeat_identical": True,
            "target_sfreq": 256.0,
            "all_values_finite": True,
            "released_primitives": True,
            "minimum_windows": 1,
            "consistent_window_structure": True,
        },
    )

    assert claim["status"] == "inconclusive"
