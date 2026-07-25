"""Acquire and verify the immutable inputs used by the audits."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import shutil
import tarfile
import tempfile
import urllib.request
from io import BytesIO
from pathlib import Path

REPO_REVISION = "325398d7d057ecc1216fb3510d70c16eb60337cc"
REPO_URL = (
    "https://codeload.github.com/xw1216/EEG-FM-Bench/tar.gz/"
    f"{REPO_REVISION}"
)
PAPER_URL = "https://arxiv.org/pdf/2508.17742v3"
REPO_SNAPSHOT_DIRECTORY = f"EEG-FM-Bench-{REPO_REVISION}"
REPO_ARCHIVE_FILENAME = f"{REPO_SNAPSHOT_DIRECTORY}.tar.gz"
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


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".tree-sha256":
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "upstream"
CACHE_REGISTRY_PATH = (
    Path(os.environ.get("EEG_FM_BENCH_REGISTRY_ROOT", "/tmp"))
    / f"eeg-fm-bench-repro-{getattr(os, 'getuid', lambda: 'user')()}"
    / "active-cache"
)


def _assert_no_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} may not contain symlinks: {current}")


def _resolved_plain_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    _assert_no_symlink_components(path, label)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist: {path}") from error
    if not stat.S_ISDIR(mode):
        raise ValueError(f"{label} must be a directory")
    return path.resolve(strict=True)


def _validate_registry_parent(path: Path) -> None:
    _assert_no_symlink_components(path, "registry parent")
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("registry parent must be a directory")
    getuid = getattr(os, "getuid", None)
    if getuid is not None and metadata.st_uid != getuid():
        raise ValueError("registry parent must be owned by the current user")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise ValueError("unsafe registry parent permissions")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def register_cache_dir(cache_dir: Path) -> None:
    """Remember a successfully verified CLI cache for later processes."""

    resolved = _resolved_plain_directory(cache_dir.absolute(), "cache directory")
    registry = CACHE_REGISTRY_PATH
    if not registry.is_absolute():
        raise ValueError("cache registry path must be absolute")
    _assert_no_symlink_components(registry.parent, "registry parent")
    registry.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    _validate_registry_parent(registry.parent)
    parent_descriptor = os.open(
        registry.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary_name: str | None = None
    try:
        for _ in range(100):
            candidate = f".{registry.name}.{os.urandom(12).hex()}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if temporary_name is None:
            raise OSError("could not allocate unique cache registry temporary file")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((str(resolved) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_name,
            registry.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
        os.fsync(parent_descriptor)
    finally:
        if temporary_name is not None:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)


def _registered_cache_dir() -> Path | None:
    registry = CACHE_REGISTRY_PATH
    if not registry.is_absolute():
        return None
    try:
        _validate_registry_parent(registry.parent)
        _assert_no_symlink_components(registry, "cache registry path")
        descriptor = os.open(
            registry, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError):
        return None
    try:
        metadata = os.fstat(descriptor)
        getuid = getattr(os, "getuid", None)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > 4096
            or stat.S_IMODE(metadata.st_mode) & 0o022
            or (getuid is not None and metadata.st_uid != getuid())
        ):
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(4097)
    finally:
        os.close(descriptor)
    if len(raw) > 4096:
        return None
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not value.endswith("\n") or "\n" in value[:-1]:
        return None
    candidate = Path(value[:-1])
    try:
        return _resolved_plain_directory(candidate, "registered cache directory")
    except ValueError:
        return None


def _reusable_cache_dir() -> Path:
    configured = os.environ.get("EEG_FM_BENCH_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    registered = _registered_cache_dir()
    if registered is not None:
        return registered
    return DEFAULT_CACHE_DIR


def _verified_file_bytes(path: Path, expected: str, input_name: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{input_name} cache path may not be a symlink")
    data = path.read_bytes()
    _verify(data, expected, input_name)
    return data


def ensure_paper_pdf(cache_dir: Path) -> Path:
    """Return a verified cached copy of the pinned paper PDF."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "2508.17742v3.pdf"
    expected = _provenance_hash("paper")
    if not destination.is_file():
        reusable_cache = _reusable_cache_dir()
        reusable_file = reusable_cache / "2508.17742v3.pdf"
        if (
            reusable_file.is_file()
            and cache_dir.resolve() != reusable_cache.resolve()
        ):
            reusable_data = _verified_file_bytes(reusable_file, expected, "paper")
            _atomic_write_bytes(destination, reusable_data)

    if destination.is_file():
        _verified_file_bytes(destination, expected, "paper")
        return destination

    data = _fetch(PAPER_URL)
    _verify(data, expected, "paper")
    _atomic_write_bytes(destination, data)
    return destination


def _repository_archive_bytes(cache_dir: Path, expected: str) -> bytes:
    archive_path = cache_dir / REPO_ARCHIVE_FILENAME
    if archive_path.is_file():
        return _verified_file_bytes(archive_path, expected, "repository")

    reusable_cache = _reusable_cache_dir()
    reusable_archive = reusable_cache / REPO_ARCHIVE_FILENAME
    if (
        reusable_archive.is_file()
        and cache_dir.resolve() != reusable_cache.resolve()
    ):
        data = _verified_file_bytes(reusable_archive, expected, "repository")
    else:
        data = _fetch(REPO_URL)
        _verify(data, expected, "repository")
    _atomic_write_bytes(archive_path, data)
    return data


def _replace_snapshot_from_archive(
    cache_dir: Path, destination: Path, archive_data: bytes, expected: str
) -> None:
    with tempfile.TemporaryDirectory(prefix=".repo-extract-", dir=cache_dir) as tmp:
        temporary_root = Path(tmp)
        extract_dir = temporary_root / "extract"
        extract_dir.mkdir()
        with tarfile.open(fileobj=BytesIO(archive_data), mode="r:gz") as archive:
            archive.extractall(extract_dir, filter="data")
        roots = [path for path in extract_dir.iterdir() if path.is_dir()]
        if len(roots) != 1 or roots[0].name != REPO_SNAPSHOT_DIRECTORY:
            raise ValueError(
                "repository archive must contain exactly the pinned root directory"
            )
        extracted = roots[0]
        (extracted / ".snapshot-sha256").write_text(
            expected + "\n", encoding="utf-8"
        )
        (extracted / ".tree-sha256").write_text(
            _tree_hash(extracted) + "\n", encoding="utf-8"
        )
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise ValueError("repository snapshot path may not be a symlink")
            shutil.rmtree(destination)
        os.replace(extracted, destination)


def ensure_repo_snapshot(cache_dir: Path) -> Path:
    """Return the extracted, sha256-verified pinned repository snapshot."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / REPO_SNAPSHOT_DIRECTORY
    expected = _provenance_hash("repository")
    archive_data = _repository_archive_bytes(cache_dir, expected)
    _replace_snapshot_from_archive(cache_dir, destination, archive_data, expected)
    return destination
