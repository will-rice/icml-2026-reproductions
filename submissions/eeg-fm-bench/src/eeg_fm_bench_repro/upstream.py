"""Acquire and verify the immutable inputs used by the audits."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO_REVISION = "325398d7d057ecc1216fb3510d70c16eb60337cc"
REPO_URL = (
    "https://codeload.github.com/xw1216/EEG-FM-Bench/tar.gz/"
    f"{REPO_REVISION}"
)
PAPER_URL = "https://arxiv.org/pdf/2508.17742v3"
REPO_SNAPSHOT_DIRECTORY = f"EEG-FM-Bench-{REPO_REVISION}"
PROVENANCE_PATH = Path(__file__).resolve().parents[2] / "evidence" / "provenance.json"


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "eeg-fm-bench-repro/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _provenance_hash(input_name: str) -> str:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    value = provenance["inputs"][input_name]["sha256"]
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"invalid sha256 for {input_name} in {PROVENANCE_PATH}")
    return value


def _verify(data: bytes, expected: str, input_name: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(
            f"{input_name} sha256 mismatch: expected {expected}, got {actual}"
        )


def ensure_paper_pdf(cache_dir: Path) -> Path:
    """Return a verified cached copy of the pinned paper PDF."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "2508.17742v3.pdf"
    expected = _provenance_hash("paper")
    if destination.is_file():
        _verify(destination.read_bytes(), expected, "paper")
        return destination

    data = _fetch(PAPER_URL)
    _verify(data, expected, "paper")
    temporary = destination.with_suffix(".pdf.tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)
    return destination


def ensure_repo_snapshot(cache_dir: Path) -> Path:
    """Return the extracted, sha256-verified pinned repository snapshot."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / REPO_SNAPSHOT_DIRECTORY
    expected = _provenance_hash("repository")
    marker = destination / ".snapshot-sha256"
    if destination.is_dir() and marker.is_file():
        if marker.read_text(encoding="utf-8").strip() == expected:
            return destination
        raise ValueError("repository snapshot sha256 mismatch in cache marker")

    data = _fetch(REPO_URL)
    _verify(data, expected, "repository")

    with tempfile.TemporaryDirectory(prefix=".repo-extract-", dir=cache_dir) as tmp:
        temporary_root = Path(tmp)
        archive = temporary_root / "snapshot.tar.gz"
        archive.write_bytes(data)
        extract_dir = temporary_root / "extract"
        extract_dir.mkdir()
        with tarfile.open(archive, mode="r:gz") as tar:
            tar.extractall(extract_dir, filter="data")
        roots = [path for path in extract_dir.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise ValueError("repository archive must contain exactly one root directory")
        (roots[0] / ".snapshot-sha256").write_text(expected + "\n", encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(roots[0]), destination)

    return destination
