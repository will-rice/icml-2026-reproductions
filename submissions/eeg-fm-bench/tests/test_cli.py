from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from eeg_fm_bench_repro import cli, upstream
from eeg_fm_bench_repro.cli import _claim
from eeg_fm_bench_repro.upstream import ensure_paper_pdf, ensure_repo_snapshot

CLAIMS = [
    "fourteen-dataset-ten-paradigm-curation",
    "standardized-preprocessing-reproducibility",
    "three-strategy-evaluation-harness",
]


def _write_provenance(path: Path, repo_sha256: str, paper_sha256: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "inputs": {
                    "repository": {"sha256": repo_sha256},
                    "paper": {"sha256": paper_sha256},
                },
            }
        ),
        encoding="utf-8",
    )


def _repository_archive() -> bytes:
    source = b"trusted = True\n"
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        root = tarfile.TarInfo(upstream.REPO_SNAPSHOT_DIRECTORY)
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        source_file = tarfile.TarInfo(
            f"{upstream.REPO_SNAPSHOT_DIRECTORY}/source.py"
        )
        source_file.size = len(source)
        archive.addfile(source_file, io.BytesIO(source))
    return output.getvalue()


def _run(cache: Path, output: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["EEG_FM_BENCH_REGISTRY_ROOT"] = str(
        output.parent / "cache-registry"
    )
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
        env=environment,
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


def test_explicit_cli_cache_is_reused_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a later process downloading bytes already verified by the CLI."""
    archive = _repository_archive()
    repo_sha256 = hashlib.sha256(archive).hexdigest()
    paper = b"%PDF-1.7 pinned paper"
    provenance = tmp_path / "provenance.json"
    _write_provenance(provenance, repo_sha256, hashlib.sha256(paper).hexdigest())
    monkeypatch.setattr(upstream, "PROVENANCE_PATH", provenance)

    seeded_cache = tmp_path / "seeded-cache"
    seeded_cache.mkdir()
    (seeded_cache / upstream.REPO_ARCHIVE_FILENAME).write_bytes(archive)
    (seeded_cache / "2508.17742v3.pdf").write_bytes(paper)

    registry = tmp_path / "cache-registry"
    monkeypatch.setattr(upstream, "CACHE_REGISTRY_PATH", registry, raising=False)
    monkeypatch.setattr(
        upstream, "DEFAULT_CACHE_DIR", tmp_path / "project-cache"
    )
    monkeypatch.delenv("EEG_FM_BENCH_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        cli,
        "build_bundle",
        lambda _cache: {
            "schema_version": 1,
            "scope": "test",
            "claims": [],
            "unavailable_claims": [],
            "provenance": {},
            "environment": {},
        },
    )

    cli.write_bundle(seeded_cache, tmp_path / "evidence")
    monkeypatch.setattr(
        upstream,
        "_fetch",
        lambda _url: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    fresh_cache = tmp_path / "fresh-cache"
    assert ensure_repo_snapshot(fresh_cache) == (
        fresh_cache / upstream.REPO_SNAPSHOT_DIRECTORY
    )
    assert ensure_paper_pdf(fresh_cache).read_bytes() == paper
    assert registry.is_file()
    assert not upstream.DEFAULT_CACHE_DIR.exists()


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


def test_default_cache_location_is_the_documented_project_local_cache() -> None:
    """Catches changing cwd silently changing the default cache destination."""
    arguments = cli._parser().parse_args([])

    assert arguments.cache_dir == upstream.DEFAULT_CACHE_DIR


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
