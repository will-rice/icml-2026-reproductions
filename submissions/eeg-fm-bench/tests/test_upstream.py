import hashlib
import json
from pathlib import Path

import pytest

from eeg_fm_bench_repro import upstream


def test_evidence_python_version_is_project_pinned() -> None:
    project_root = upstream.PROVENANCE_PATH.parent.parent

    assert (project_root / ".python-version").read_text(encoding="utf-8") == (
        "3.12.11\n"
    )


def test_paper_license_matches_pinned_arxiv_v3_metadata() -> None:
    provenance = json.loads(upstream.PROVENANCE_PATH.read_text(encoding="utf-8"))
    license_record = provenance["inputs"]["paper"]["license"]

    assert license_record == {
        "spdx": "CC-BY-4.0",
        "source": "https://arxiv.org/abs/2508.17742v3 license link",
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "note": (
            "arXiv v3 identifies the paper as Creative Commons "
            "Attribution 4.0 International."
        ),
    }


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


def test_repo_sha256_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = b"not-the-pinned-archive"
    provenance = tmp_path / "provenance.json"
    _write_provenance(provenance, "0" * 64, hashlib.sha256(b"paper").hexdigest())
    monkeypatch.setattr(upstream, "PROVENANCE_PATH", provenance)
    monkeypatch.setattr(upstream, "_fetch", lambda _url: archive)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        upstream.ensure_repo_snapshot(tmp_path / "cache")


def test_paper_cache_hit_avoids_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = b"%PDF-1.7 cached"
    provenance = tmp_path / "provenance.json"
    _write_provenance(
        provenance,
        hashlib.sha256(b"repo").hexdigest(),
        hashlib.sha256(paper).hexdigest(),
    )
    monkeypatch.setattr(upstream, "PROVENANCE_PATH", provenance)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_pdf = cache_dir / "2508.17742v3.pdf"
    cached_pdf.write_bytes(paper)

    def fail_fetch(_url: str) -> bytes:
        raise AssertionError("cache hit attempted a download")

    monkeypatch.setattr(upstream, "_fetch", fail_fetch)

    assert upstream.ensure_paper_pdf(cache_dir) == cached_pdf


def test_cached_repo_snapshot_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = tmp_path / "provenance.json"
    _write_provenance(
        provenance,
        hashlib.sha256(b"repo").hexdigest(),
        hashlib.sha256(b"paper").hexdigest(),
    )
    monkeypatch.setattr(upstream, "PROVENANCE_PATH", provenance)

    snapshot = tmp_path / "cache" / upstream.REPO_SNAPSHOT_DIRECTORY
    snapshot.mkdir(parents=True)
    marker = snapshot / ".snapshot-sha256"
    marker.write_text("a" * 64 + "\n", encoding="utf-8")
    data_file = snapshot / "kept.txt"
    data_file.write_text("cached", encoding="utf-8")
    (snapshot / ".tree-sha256").write_text(
        upstream._tree_hash(snapshot) + "\n", encoding="utf-8"
    )
    provenance_data = json.loads(provenance.read_text(encoding="utf-8"))
    provenance_data["inputs"]["repository"]["sha256"] = "a" * 64
    provenance.write_text(json.dumps(provenance_data), encoding="utf-8")

    monkeypatch.setattr(
        upstream,
        "_fetch",
        lambda _url: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    assert upstream.ensure_repo_snapshot(tmp_path / "cache") == snapshot
    assert data_file.read_text(encoding="utf-8") == "cached"


def test_cached_repo_snapshot_rejects_modified_extracted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches attributing a tampered extracted tree to the pinned archive."""
    provenance = tmp_path / "provenance.json"
    _write_provenance(provenance, "a" * 64, hashlib.sha256(b"paper").hexdigest())
    monkeypatch.setattr(upstream, "PROVENANCE_PATH", provenance)
    snapshot = tmp_path / "cache" / upstream.REPO_SNAPSHOT_DIRECTORY
    snapshot.mkdir(parents=True)
    (snapshot / ".snapshot-sha256").write_text("a" * 64 + "\n", encoding="utf-8")
    data_file = snapshot / "source.py"
    data_file.write_text("trusted = True\n", encoding="utf-8")
    (snapshot / ".tree-sha256").write_text(
        upstream._tree_hash(snapshot) + "\n", encoding="utf-8"
    )
    data_file.write_text("trusted = False\n", encoding="utf-8")
    monkeypatch.setattr(
        upstream,
        "_fetch",
        lambda _url: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    with pytest.raises(ValueError, match="tree hash mismatch"):
        upstream.ensure_repo_snapshot(tmp_path / "cache")
