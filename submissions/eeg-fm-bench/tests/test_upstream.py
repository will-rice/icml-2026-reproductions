import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
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


def _repository_archive(source: bytes = b"trusted = True\n") -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        root = tarfile.TarInfo(upstream.REPO_SNAPSHOT_DIRECTORY)
        root.type = tarfile.DIRTYPE
        root.mode = 0o755
        archive.addfile(root)
        source_file = tarfile.TarInfo(
            f"{upstream.REPO_SNAPSHOT_DIRECTORY}/source.py"
        )
        source_file.size = len(source)
        source_file.mode = 0o644
        archive.addfile(source_file, io.BytesIO(source))
    return output.getvalue()


def test_repo_sha256_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = b"not-the-pinned-archive"
    provenance = tmp_path / "provenance.json"
    _write_provenance(provenance, "0" * 64, hashlib.sha256(b"paper").hexdigest())
    monkeypatch.setattr(upstream, "PROVENANCE_PATH", provenance)
    monkeypatch.setattr(upstream, "_fetch", lambda _url: archive)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        upstream.ensure_repo_snapshot(tmp_path / "cache")


def test_downloaded_repository_archive_is_persisted_and_reverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches discarding the only provenance-authenticated repository bytes."""
    archive = _repository_archive()
    provenance = tmp_path / "provenance.json"
    _write_provenance(
        provenance,
        hashlib.sha256(archive).hexdigest(),
        hashlib.sha256(b"paper").hexdigest(),
    )
    monkeypatch.setattr(upstream, "PROVENANCE_PATH", provenance)
    monkeypatch.setattr(upstream, "_fetch", lambda _url: archive)
    monkeypatch.setattr(
        upstream, "CACHE_REGISTRY_PATH", tmp_path / "missing-registry"
    )
    monkeypatch.setattr(upstream, "DEFAULT_CACHE_DIR", tmp_path / "missing-default")
    cache = tmp_path / "cache"

    snapshot = upstream.ensure_repo_snapshot(cache)

    cached_archive = cache / f"{upstream.REPO_SNAPSHOT_DIRECTORY}.tar.gz"
    assert cached_archive.read_bytes() == archive
    assert (snapshot / "source.py").read_bytes() == b"trusted = True\n"

    cached_archive.write_bytes(b"forged archive")
    (snapshot / "source.py").write_bytes(b"forged = True\n")
    (snapshot / ".snapshot-sha256").write_text(
        hashlib.sha256(archive).hexdigest() + "\n", encoding="utf-8"
    )
    (snapshot / ".tree-sha256").write_text(
        upstream._tree_hash(snapshot) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        upstream,
        "_fetch",
        lambda _url: (_ for _ in ()).throw(AssertionError("unexpected network")),
    )

    with pytest.raises(ValueError, match="repository sha256 mismatch"):
        upstream.ensure_repo_snapshot(cache)


def test_verified_archive_reextracts_over_forged_tree_and_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches allowing a mutable tree and its markers to authenticate themselves."""
    archive = _repository_archive()
    provenance = tmp_path / "provenance.json"
    expected = hashlib.sha256(archive).hexdigest()
    _write_provenance(
        provenance,
        expected,
        hashlib.sha256(b"paper").hexdigest(),
    )
    monkeypatch.setattr(upstream, "PROVENANCE_PATH", provenance)
    monkeypatch.setattr(
        upstream,
        "_fetch",
        lambda _url: (_ for _ in ()).throw(AssertionError("unexpected network")),
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{upstream.REPO_SNAPSHOT_DIRECTORY}.tar.gz").write_bytes(archive)
    snapshot = upstream.ensure_repo_snapshot(cache)

    (snapshot / "source.py").write_bytes(b"forged = True\n")
    (snapshot / ".snapshot-sha256").write_text(expected + "\n", encoding="utf-8")
    (snapshot / ".tree-sha256").write_text(
        upstream._tree_hash(snapshot) + "\n", encoding="utf-8"
    )

    reused = upstream.ensure_repo_snapshot(cache)

    assert reused == snapshot
    assert (reused / "source.py").read_bytes() == b"trusted = True\n"


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


def test_registry_publish_does_not_follow_predictable_temp_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a precreated active-cache.tmp symlink redirecting registry writes."""
    cache = tmp_path / "cache"
    cache.mkdir()
    registry = tmp_path / "registry" / "active-cache"
    registry.parent.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("do not overwrite\n", encoding="utf-8")
    registry.with_suffix(".tmp").symlink_to(victim)
    monkeypatch.setattr(upstream, "CACHE_REGISTRY_PATH", registry)

    upstream.register_cache_dir(cache)

    assert victim.read_text(encoding="utf-8") == "do not overwrite\n"
    assert registry.read_text(encoding="utf-8") == f"{cache.resolve()}\n"


def test_registry_publish_rejects_symlinked_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an untrusted registry parent redirecting writes outside its root."""
    cache = tmp_path / "cache"
    cache.mkdir()
    victim_directory = tmp_path / "victim"
    victim_directory.mkdir()
    symlinked_parent = tmp_path / "registry"
    symlinked_parent.symlink_to(victim_directory, target_is_directory=True)
    monkeypatch.setattr(
        upstream, "CACHE_REGISTRY_PATH", symlinked_parent / "active-cache"
    )

    with pytest.raises(ValueError, match="registry parent"):
        upstream.register_cache_dir(cache)

    assert not (victim_directory / "active-cache").exists()


def test_registry_publish_rejects_world_writable_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches publishing a trusted-looking pointer in an attacker-writable directory."""
    cache = tmp_path / "cache"
    cache.mkdir()
    registry_parent = tmp_path / "registry"
    registry_parent.mkdir()
    registry_parent.chmod(0o777)
    monkeypatch.setattr(
        upstream, "CACHE_REGISTRY_PATH", registry_parent / "active-cache"
    )

    with pytest.raises(ValueError, match="registry parent permissions"):
        upstream.register_cache_dir(cache)

    assert not (registry_parent / "active-cache").exists()


def test_registry_reader_ignores_symlinked_pointer_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches trusting registry or cache symlinks as authoritative paths."""
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    pointer = tmp_path / "pointer"
    pointer.write_text(f"{cache}\n", encoding="utf-8")
    registry = tmp_path / "active-cache"
    registry.symlink_to(pointer)
    monkeypatch.setattr(upstream, "CACHE_REGISTRY_PATH", registry)
    monkeypatch.setattr(upstream, "DEFAULT_CACHE_DIR", fallback)
    monkeypatch.delenv("EEG_FM_BENCH_CACHE_DIR", raising=False)

    assert upstream._reusable_cache_dir() == fallback

    registry.unlink()
    symlinked_cache = tmp_path / "symlinked-cache"
    symlinked_cache.symlink_to(cache, target_is_directory=True)
    registry.write_text(f"{symlinked_cache}\n", encoding="utf-8")

    assert upstream._reusable_cache_dir() == fallback


def test_cross_process_reuse_verifies_archive_offline_across_tmpdirs(
    tmp_path: Path,
) -> None:
    """Catches pointer-only tests that never perform real verified cache reuse."""
    archive = _repository_archive()
    paper = b"%PDF-1.7 synthetic pinned paper"
    provenance = tmp_path / "provenance.json"
    _write_provenance(
        provenance,
        hashlib.sha256(archive).hexdigest(),
        hashlib.sha256(paper).hexdigest(),
    )
    source_cache = tmp_path / "source-cache"
    source_cache.mkdir()
    (source_cache / f"{upstream.REPO_SNAPSHOT_DIRECTORY}.tar.gz").write_bytes(
        archive
    )
    (source_cache / "2508.17742v3.pdf").write_bytes(paper)
    first_tmpdir = tmp_path / "first-tmp"
    second_tmpdir = tmp_path / "second-tmp"
    first_tmpdir.mkdir()
    second_tmpdir.mkdir()
    registry_root = tmp_path / "registry-root"
    project_src = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment.update(
        {
            "EEG_FM_BENCH_REGISTRY_ROOT": str(registry_root),
            "PYTHONPATH": str(project_src),
        }
    )
    script = (
        "from pathlib import Path; import sys; "
        "from eeg_fm_bench_repro import upstream; "
        "upstream.PROVENANCE_PATH = Path(sys.argv[2]); "
        "upstream._fetch = lambda _url: (_ for _ in ()).throw("
        "AssertionError('network forbidden')); "
        "snapshot = upstream.ensure_repo_snapshot(Path(sys.argv[1])); "
        "upstream.ensure_paper_pdf(Path(sys.argv[1])); "
        "upstream.register_cache_dir(Path(sys.argv[1])); "
        "print((snapshot / 'source.py').read_text().strip())"
    )
    first = subprocess.run(
        [sys.executable, "-c", script, str(source_cache), str(provenance)],
        capture_output=True,
        env=environment | {"TMPDIR": str(first_tmpdir)},
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    assert first.stdout.strip() == "trusted = True"
    shutil.rmtree(first_tmpdir)

    source_snapshot = source_cache / upstream.REPO_SNAPSHOT_DIRECTORY
    (source_snapshot / "source.py").write_bytes(b"forged = True\n")
    (source_snapshot / ".snapshot-sha256").write_text(
        hashlib.sha256(archive).hexdigest() + "\n", encoding="utf-8"
    )
    (source_snapshot / ".tree-sha256").write_text(
        upstream._tree_hash(source_snapshot) + "\n", encoding="utf-8"
    )

    reused_cache = tmp_path / "reused-cache"
    second = subprocess.run(
        [sys.executable, "-c", script, str(reused_cache), str(provenance)],
        capture_output=True,
        env=environment | {"TMPDIR": str(second_tmpdir)},
        text=True,
        check=False,
    )

    assert second.returncode == 0, second.stderr
    assert second.stdout.strip() == "trusted = True"
    assert (
        reused_cache / f"{upstream.REPO_SNAPSHOT_DIRECTORY}.tar.gz"
    ).read_bytes() == archive
    assert (reused_cache / "2508.17742v3.pdf").read_bytes() == paper
