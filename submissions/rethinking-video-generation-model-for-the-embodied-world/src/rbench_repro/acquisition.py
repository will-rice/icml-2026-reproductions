import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Literal, Protocol, cast
from urllib.parse import unquote

from huggingface_hub import HfApi, RepoFile, RepoFolder, hf_hub_url
from huggingface_hub.utils import build_hf_headers
import httpx

from rbench_repro.model import canonical_json, sha256_bytes


MAX_FILE_BYTES = 1_048_576
MAX_TOTAL_BYTES = 8_388_608
READ_TIMEOUT_SECONDS = 30
MAX_HUB_REDIRECTS = 5
HUB_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
FULL_SHA = re.compile(r"[0-9a-f]{40}")
FILE_SHA = re.compile(r"[0-9a-f]{64}")
FILE_RECORD_KEYS = frozenset({"bytes", "path", "sha256"})
TREE_ENTRY_KEYS = frozenset({"kind", "path", "size"})
APPROVED_HUB_REDIRECT_PATHS = {
    "cdn-lfs.hf.co": "/repos/",
    "cdn-lfs-us-1.hf.co": "/repos/",
    "cdn-lfs-eu-1.hf.co": "/repos/",
    "cas-bridge.xethub.hf.co": "/xet-bridge-us/",
}
SOURCE_MANIFEST_KEYS = frozenset(
    {
        "acquired_at",
        "canonical_url",
        "command",
        "files",
        "kind",
        "label",
        "license_id",
        "license_source",
        "redistributable",
        "repo_id",
        "requested_revision",
        "resolved_revision",
        "tree",
    }
)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    label: str
    kind: Literal["git", "dataset", "space"]
    repo_id: str
    canonical_url: str
    requested_revision: str
    allowlist: tuple[str, ...]
    license_id: str
    license_source: str
    redistributable: bool
    command: str


@dataclass(frozen=True, slots=True)
class TreeEntry:
    path: str
    kind: Literal["file", "directory", "symlink"]
    size: int

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "path": self.path, "size": self.size}

    @classmethod
    def from_dict(cls, value: object) -> "TreeEntry":
        if not isinstance(value, dict) or set(value) != TREE_ENTRY_KEYS:
            raise ValueError("invalid tree entry: fields")
        path = value["path"]
        kind = value["kind"]
        size = value["size"]
        if not isinstance(path, str):
            raise ValueError("invalid tree entry: path")
        try:
            safe_path(path)
        except ValueError:
            raise ValueError("invalid tree entry: path") from None
        if not isinstance(kind, str) or kind not in {"file", "directory", "symlink"}:
            raise ValueError("invalid tree entry: kind")
        if type(size) is not int or size < 0:
            raise ValueError("invalid tree entry: size")
        return cls(path=path, kind=cast(Literal["file", "directory", "symlink"], kind), size=size)


@dataclass(frozen=True, slots=True)
class FileRecord:
    path: str
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {"bytes": self.bytes, "path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> "FileRecord":
        if not isinstance(value, dict) or set(value) != FILE_RECORD_KEYS:
            raise ValueError("invalid file record: fields")
        path = value["path"]
        byte_count = value["bytes"]
        digest = value["sha256"]
        if not isinstance(path, str):
            raise ValueError("invalid file record: path")
        try:
            safe_path(path)
        except ValueError:
            raise ValueError("invalid file record: path") from None
        if type(byte_count) is not int or byte_count < 0 or byte_count > MAX_FILE_BYTES:
            raise ValueError("invalid file record: bytes")
        if not isinstance(digest, str) or not FILE_SHA.fullmatch(digest):
            raise ValueError("invalid file record: sha256")
        return cls(path=path, bytes=byte_count, sha256=digest)


@dataclass(frozen=True, slots=True)
class SourceManifest:
    label: str
    kind: Literal["git", "dataset", "space"]
    repo_id: str
    canonical_url: str
    requested_revision: str
    resolved_revision: str
    acquired_at: str
    license_id: str
    license_source: str
    redistributable: bool
    command: str
    files: tuple[FileRecord, ...]
    tree: tuple[TreeEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "acquired_at": self.acquired_at,
            "canonical_url": self.canonical_url,
            "command": self.command,
            "files": [record.to_dict() for record in self.files],
            "kind": self.kind,
            "label": self.label,
            "license_id": self.license_id,
            "license_source": self.license_source,
            "redistributable": self.redistributable,
            "repo_id": self.repo_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "tree": [entry.to_dict() for entry in self.tree],
        }

    @classmethod
    def from_dict(cls, value: object) -> "SourceManifest":
        if not isinstance(value, dict) or set(value) != SOURCE_MANIFEST_KEYS:
            raise ValueError("invalid source manifest: fields")
        files = value["files"]
        if not isinstance(files, list) or not files:
            raise ValueError("invalid source manifest: files")
        tree = value["tree"]
        if not isinstance(tree, list) or not tree:
            raise ValueError("invalid source manifest: tree")
        kind = value["kind"]
        if not isinstance(kind, str) or kind not in {"git", "dataset", "space"}:
            raise ValueError("invalid source manifest: kind")
        string_fields = (
            "label",
            "repo_id",
            "canonical_url",
            "acquired_at",
            "license_id",
            "license_source",
            "command",
        )
        if any(not isinstance(value[field], str) or not value[field] for field in string_fields):
            raise ValueError("invalid source manifest: string field")
        requested_revision = value["requested_revision"]
        resolved_revision = value["resolved_revision"]
        if (
            not isinstance(requested_revision, str)
            or not FULL_SHA.fullmatch(requested_revision)
            or not isinstance(resolved_revision, str)
            or resolved_revision != requested_revision
        ):
            raise ValueError("invalid source manifest: revision")
        if type(value["redistributable"]) is not bool:
            raise ValueError("invalid source manifest: redistributable")
        records = tuple(FileRecord.from_dict(item) for item in files)
        if len({record.path for record in records}) != len(records):
            raise ValueError("duplicate file in source manifest")
        if sum(record.bytes for record in records) > MAX_TOTAL_BYTES:
            raise ValueError("invalid source manifest: total bytes")
        tree_entries = tuple(TreeEntry.from_dict(item) for item in tree)
        if len({entry.path for entry in tree_entries}) != len(tree_entries):
            raise ValueError("duplicate tree entry in source manifest")
        _validate_file_tree_binding(records, tree_entries)
        return cls(
            label=value["label"],
            kind=cast(Literal["git", "dataset", "space"], kind),
            repo_id=value["repo_id"],
            canonical_url=value["canonical_url"],
            requested_revision=requested_revision,
            resolved_revision=resolved_revision,
            acquired_at=value["acquired_at"],
            license_id=value["license_id"],
            license_source=value["license_source"],
            redistributable=value["redistributable"],
            command=value["command"],
            files=records,
            tree=tree_entries,
        )


def _validate_file_tree_binding(
    records: tuple[FileRecord, ...], tree: tuple[TreeEntry, ...]
) -> None:
    if any(entry.kind == "symlink" for entry in tree):
        raise ValueError("invalid source manifest: tree symlink")
    tree_by_path = {entry.path: entry for entry in tree}
    if any(
        (entry := tree_by_path.get(record.path)) is None
        or entry.kind != "file"
        or entry.size != record.bytes
        for record in records
    ):
        raise ValueError("file/tree mismatch in source manifest")


@dataclass(frozen=True, slots=True)
class AcquiredSource:
    manifest: SourceManifest
    root: Path


class SourceReader(Protocol):
    def resolve(self, spec: SourceSpec) -> str: ...

    def tree(self, spec: SourceSpec, revision: str) -> tuple[TreeEntry, ...]: ...

    def read(self, spec: SourceSpec, revision: str, path: str, timeout_seconds: int) -> bytes: ...


def safe_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"unsafe path: {value}")
    return path


class GitReader:
    def __init__(self, staging_root: Path):
        self.staging_root = staging_root
        self.staging_root.mkdir(parents=True)
        self.repositories: dict[str, Path] = {}

    def resolve(self, spec: SourceSpec) -> str:
        repository = self.staging_root / spec.label
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", f"{spec.canonical_url}.git", str(repository)],
            check=True,
            capture_output=True,
        )
        commands = (
            ["git", "-C", str(repository), "fetch", "--no-tags", "origin", spec.requested_revision],
            ["git", "-C", str(repository), "sparse-checkout", "init", "--cone"],
            ["git", "-C", str(repository), "sparse-checkout", "set", "eval", "scripts"],
            ["git", "-C", str(repository), "checkout", "--detach", spec.requested_revision],
        )
        for command in commands:
            subprocess.run(command, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.repositories[spec.label] = repository
        return result.stdout.strip()

    def tree(self, spec: SourceSpec, revision: str) -> tuple[TreeEntry, ...]:
        result = subprocess.run(
            ["git", "-C", str(self.repositories[spec.label]), "ls-tree", "-rlz", revision],
            check=True,
            capture_output=True,
        )
        entries = []
        for raw_entry in result.stdout.split(b"\0"):
            if not raw_entry:
                continue
            metadata, path_bytes = raw_entry.split(b"\t", 1)
            mode, object_kind, _object_id, size = metadata.decode().split()
            if object_kind != "blob":
                raise ValueError(f"invalid Git tree entry: {path_bytes.decode()}")
            kind = "symlink" if mode == "120000" else "file"
            entries.append(TreeEntry(path_bytes.decode(), kind, 0 if size == "-" else int(size)))
        return tuple(entries)

    def read(self, spec: SourceSpec, revision: str, path: str, timeout_seconds: int) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(self.repositories[spec.label]), "show", f"{revision}:{path}"],
            check=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        return result.stdout


class HubReader:
    def __init__(self, staging_root: Path):
        self.staging_root = staging_root
        self.api = HfApi()
        self.declared_sizes: dict[tuple[str, str, str, str], int] = {}

    def resolve(self, spec: SourceSpec) -> str:
        info = self.api.repo_info(
            repo_id=spec.repo_id,
            revision=spec.requested_revision,
            repo_type=spec.kind,
        )
        if info.sha is None:
            raise ValueError("missing resolved revision")
        return info.sha

    def tree(self, spec: SourceSpec, revision: str) -> tuple[TreeEntry, ...]:
        allowed = set(spec.allowlist)
        entries = []
        for item in self.api.list_repo_tree(
            repo_id=spec.repo_id,
            revision=revision,
            repo_type=spec.kind,
            recursive=True,
            expand=True,
        ):
            if isinstance(item, RepoFile):
                if type(item.size) is not int or item.size < 0:
                    raise ValueError(f"invalid Hub tree entry: {item.path}")
                entries.append(TreeEntry(item.path, "file", item.size))
            elif isinstance(item, RepoFolder):
                entries.append(TreeEntry(item.path, "directory", 0))
            else:
                raise ValueError(f"invalid Hub tree entry: {item.path}")
            if item.path in allowed and isinstance(item, RepoFile):
                self.declared_sizes[(spec.kind, spec.repo_id, revision, item.path)] = item.size
        return tuple(entries)

    def read(self, spec: SourceSpec, revision: str, path: str, timeout_seconds: int) -> bytes:
        url = hf_hub_url(
            repo_id=spec.repo_id,
            filename=path,
            revision=revision,
            repo_type=spec.kind,
        )
        allowed_huggingface_paths = validate_hub_resolve_url(
            url=url,
            spec=spec,
            revision=revision,
            path=path,
        )
        headers = build_hf_headers(library_name="rbench-repro", library_version="0.1.0")
        declared_size = self.declared_sizes.get((spec.kind, spec.repo_id, revision, path))
        if declared_size is None:
            raise ValueError(f"missing declared size: {path}")
        return asyncio.run(
            stream_hub_payload(
                url=url,
                headers=headers,
                declared_size=declared_size,
                timeout_seconds=timeout_seconds,
                path=path,
                allowed_huggingface_paths=allowed_huggingface_paths,
            )
        )


async def stream_hub_payload(
    url: str,
    headers: dict[str, str],
    declared_size: int,
    timeout_seconds: int,
    path: str,
    allowed_huggingface_paths: tuple[str, str],
) -> bytes:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    try:
        async with asyncio.timeout_at(deadline):
            return await stream_hub_payload_before_deadline(
                url=url,
                headers=headers,
                declared_size=declared_size,
                timeout_seconds=timeout_seconds,
                path=path,
                allowed_huggingface_paths=allowed_huggingface_paths,
            )
    except TimeoutError:
        raise TimeoutError(f"absolute transfer deadline exceeded: {path}") from None


async def stream_hub_payload_before_deadline(
    url: str,
    headers: dict[str, str],
    declared_size: int,
    timeout_seconds: int,
    path: str,
    allowed_huggingface_paths: tuple[str, str],
) -> bytes:
    payload = bytearray()
    async with httpx.AsyncClient(
        follow_redirects=False, timeout=httpx.Timeout(float(timeout_seconds))
    ) as client:
        current_url = httpx.URL(url)
        visited_urls = {str(current_url)}
        redirect_count = 0
        while True:
            request_headers = headers
            if current_url.host != "huggingface.co":
                request_headers = {
                    name: value for name, value in headers.items() if name.lower() != "authorization"
                }
            async with client.stream("GET", current_url, headers=request_headers) as response:
                if response.status_code in HUB_REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if location is None:
                        raise ValueError(f"missing redirect Location: {path}")
                    try:
                        next_url = current_url.join(location)
                    except (TypeError, ValueError):
                        raise ValueError(f"unsafe Hub URL: {location}") from None
                    validate_hub_redirect_url(
                        url=str(next_url),
                        allowed_huggingface_paths=allowed_huggingface_paths,
                    )
                    if str(next_url) in visited_urls:
                        raise ValueError(f"Hub redirect loop: {path}")
                    redirect_count += 1
                    if redirect_count > MAX_HUB_REDIRECTS:
                        raise ValueError(f"too many Hub redirects: {path}")
                    visited_urls.add(str(next_url))
                    current_url = next_url
                    continue

                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=65_536):
                    received_size = len(payload) + len(chunk)
                    if received_size > MAX_FILE_BYTES:
                        raise ValueError(f"byte limit exceeded: {path}")
                    if received_size > declared_size:
                        raise ValueError(f"declared size mismatch: {path}")
                    payload.extend(chunk)
                break
    if len(payload) != declared_size:
        raise ValueError(f"declared size mismatch: {path}")
    return bytes(payload)


def validate_hub_resolve_url(
    url: str,
    spec: SourceSpec,
    revision: str,
    path: str,
) -> tuple[str, str]:
    repo_type = {"dataset": "datasets", "space": "spaces"}.get(spec.kind)
    if repo_type is None:
        raise ValueError(f"unsafe Hub resolve URL: {url}")
    resolve_path = f"/{repo_type}/{spec.repo_id}/resolve/{revision}/{path}"
    cache_path = f"/api/resolve-cache/{repo_type}/{spec.repo_id}/{revision}/{path}"
    parsed = parse_safe_hub_url(url)
    if parsed.host != "huggingface.co" or parsed.path != resolve_path or parsed.query:
        raise ValueError(f"unsafe Hub resolve URL: {url}")
    return resolve_path, cache_path


def validate_hub_redirect_url(
    url: str,
    allowed_huggingface_paths: tuple[str, str],
) -> None:
    parsed = parse_safe_hub_url(url)
    if parsed.host == "huggingface.co":
        if parsed.path not in allowed_huggingface_paths:
            raise ValueError(f"unsafe Hub URL: {url}")
        return
    prefix = APPROVED_HUB_REDIRECT_PATHS.get(parsed.host)
    if prefix is None or not parsed.path.startswith(prefix):
        raise ValueError(f"unsafe Hub URL: {url}")


def parse_safe_hub_url(url: str) -> httpx.URL:
    try:
        parsed = httpx.URL(url)
    except (TypeError, ValueError):
        raise ValueError(f"unsafe Hub URL: {url}") from None
    decoded_path = parsed.path
    while "%" in decoded_path:
        next_path = unquote(decoded_path)
        if next_path == decoded_path:
            break
        decoded_path = next_path
    path = PurePosixPath(decoded_path)
    if (
        parsed.scheme != "https"
        or not parsed.host
        or parsed.port not in {None, 443}
        or parsed.userinfo
        or parsed.fragment
        or not parsed.path.startswith("/")
        or "\\" in decoded_path
        or ".." in path.parts
    ):
        raise ValueError(f"unsafe Hub URL: {url}")
    return parsed


def acquire_source(
    spec: SourceSpec,
    cache_root: Path,
    reader: SourceReader,
    acquired_at: str,
) -> AcquiredSource:
    _validate_spec(spec)
    try:
        resolved = reader.resolve(spec)
    except Exception as error:
        raise ValueError(f"acquisition failed: {spec.label}: {type(error).__name__}") from None
    if not FULL_SHA.fullmatch(resolved) or resolved != spec.requested_revision:
        raise ValueError(f"resolved revision mismatch: {spec.label}")
    try:
        entries = reader.tree(spec, resolved)
    except Exception as error:
        raise ValueError(f"acquisition failed: {spec.label}: {type(error).__name__}") from None
    tree, declared_sizes = _validate_tree(spec, entries)

    index_path = cache_root / ".indexes" / f"{_spec_key(spec)}.json"
    if index_path.is_file():
        manifest = _read_source_manifest(index_path)
        _verify_manifest_spec(manifest, spec, resolved)
        if manifest.tree != tree:
            raise ValueError(f"tree metadata mismatch: {spec.label}")
        root = cache_root / _content_key(manifest.files, manifest.tree)
        _verify_cache(root, manifest.files)
        return AcquiredSource(manifest=manifest, root=root)

    payloads: list[tuple[str, bytes]] = []
    records = []
    total = 0
    for path in spec.allowlist:
        try:
            payload = reader.read(spec, resolved, path, READ_TIMEOUT_SECONDS)
        except Exception as error:
            raise ValueError(f"acquisition failed: {spec.label}: {type(error).__name__}") from None
        declared_size = declared_sizes[path]
        total += len(payload)
        if len(payload) > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError(f"byte limit exceeded: {path}")
        if len(payload) != declared_size:
            raise ValueError(f"declared size mismatch: {path}")
        payloads.append((path, payload))
        records.append(FileRecord(path=path, bytes=len(payload), sha256=sha256_bytes(payload)))

    manifest = SourceManifest(
        label=spec.label,
        kind=spec.kind,
        repo_id=spec.repo_id,
        canonical_url=spec.canonical_url,
        requested_revision=spec.requested_revision,
        resolved_revision=resolved,
        acquired_at=acquired_at,
        license_id=spec.license_id,
        license_source=spec.license_source,
        redistributable=spec.redistributable,
        command=spec.command,
        files=tuple(records),
        tree=tree,
    )
    root = cache_root / _content_key(manifest.files, manifest.tree)
    _write_cache(root, payloads)
    _verify_cache(root, manifest.files)
    _atomic_write(index_path, canonical_json(manifest.to_dict()))
    return AcquiredSource(manifest=manifest, root=root)


def acquire_all(
    cache_root: Path,
    output_manifest: Path,
    acquired_at: str,
) -> tuple[AcquiredSource, ...]:
    with tempfile.TemporaryDirectory() as temporary:
        staging_root = Path(temporary)
        git_reader = GitReader(staging_root / "git")
        hub_reader = HubReader(staging_root / "hub")
        acquired = tuple(
            acquire_source(
                spec=spec,
                cache_root=cache_root,
                reader=git_reader if spec.kind == "git" else hub_reader,
                acquired_at=acquired_at,
            )
            for spec in SOURCE_SPECS
        )
    inputs_root = output_manifest.parent.parent / "inputs" / "rbench"
    dataset = next(source for source in acquired if source.manifest.label == "rbench-dataset")
    for record in dataset.manifest.files:
        destination = inputs_root / ("UPSTREAM_README.md" if record.path == "README.md" else record.path)
        _atomic_write(destination, (dataset.root / record.path).read_bytes())
    _atomic_write(
        output_manifest,
        canonical_json({"schema_version": 1, "sources": [source.manifest.to_dict() for source in acquired]}),
    )
    return acquired


def load_acquired(manifest_path: Path, cache_root: Path) -> dict[str, AcquiredSource]:
    try:
        value = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("invalid input manifest") from None
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "sources"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or not isinstance(value["sources"], list)
        or not value["sources"]
    ):
        raise ValueError("invalid input manifest")
    manifests = tuple(SourceManifest.from_dict(item) for item in value["sources"])
    if len({manifest.label for manifest in manifests}) != len(manifests):
        raise ValueError("duplicate source in input manifest")
    specs_by_label = {spec.label: spec for spec in SOURCE_SPECS}
    if {manifest.label for manifest in manifests} != set(specs_by_label):
        raise ValueError("source labels mismatch in input manifest")
    for manifest in manifests:
        spec = specs_by_label[manifest.label]
        _verify_manifest_spec(manifest, spec, spec.requested_revision)
    acquired = {}
    for manifest in manifests:
        root = cache_root / _content_key(manifest.files, manifest.tree)
        _verify_cache(root, manifest.files)
        acquired[manifest.label] = AcquiredSource(manifest=manifest, root=root)
    return acquired


def _validate_spec(spec: SourceSpec) -> None:
    if not FULL_SHA.fullmatch(spec.requested_revision):
        raise ValueError(f"invalid requested revision: {spec.label}")
    if len(set(spec.allowlist)) != len(spec.allowlist):
        raise ValueError(f"duplicate allowlist path: {spec.label}")
    for path in spec.allowlist:
        safe_path(path)


def _validate_tree(
    spec: SourceSpec, entries: tuple[TreeEntry, ...]
) -> tuple[tuple[TreeEntry, ...], dict[str, int]]:
    paths = set()
    files = {}
    total = 0
    for entry in entries:
        safe_path(entry.path)
        if entry.path in paths:
            raise ValueError(f"duplicate tree entry: {entry.path}")
        paths.add(entry.path)
        if type(entry.size) is not int or entry.size < 0:
            raise ValueError(f"invalid declared size: {entry.path}")
        if entry.kind not in {"file", "directory", "symlink"}:
            raise ValueError(f"invalid tree entry kind: {entry.path}")
        if entry.kind == "symlink":
            raise ValueError(f"symlink rejected: {entry.path}")
        if entry.kind == "directory":
            continue
        files[entry.path] = entry.size
    if not set(spec.allowlist) <= set(files):
        raise ValueError(f"allowlist mismatch: {spec.label}")
    declared_sizes = {path: files[path] for path in spec.allowlist}
    for path, size in declared_sizes.items():
        if size > MAX_FILE_BYTES:
            raise ValueError(f"byte limit exceeded: {path}")
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ValueError(f"byte limit exceeded: {path}")
    return tuple(sorted(entries, key=lambda entry: entry.path)), declared_sizes


def _spec_key(spec: SourceSpec) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "allowlist": list(spec.allowlist),
                "canonical_url": spec.canonical_url,
                "command": spec.command,
                "kind": spec.kind,
                "label": spec.label,
                "license_id": spec.license_id,
                "license_source": spec.license_source,
                "repo_id": spec.repo_id,
                "redistributable": spec.redistributable,
                "revision": spec.requested_revision,
            }
        )
    )


def _content_key(records: tuple[FileRecord, ...], tree: tuple[TreeEntry, ...]) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "files": [record.to_dict() for record in records],
                "tree": [entry.to_dict() for entry in tree],
            }
        )
    )


def _write_cache(root: Path, payloads: list[tuple[str, bytes]]) -> None:
    if root.exists():
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        for path, payload in payloads:
            destination = staging / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        try:
            os.replace(staging, root)
        except FileExistsError:
            pass
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _verify_cache(root: Path, records: tuple[FileRecord, ...]) -> None:
    if root.is_symlink():
        raise ValueError("cache symlink rejected")
    expected_files = {record.path for record in records}
    expected_directories = {
        str(parent)
        for record in records
        for parent in PurePosixPath(record.path).parents
        if str(parent) != "."
    }
    for cached_path in root.rglob("*"):
        relative = cached_path.relative_to(root).as_posix()
        if relative not in expected_files and relative not in expected_directories:
            raise ValueError(f"undeclared cache entry: {relative}")
        if cached_path.is_symlink():
            raise ValueError(f"cache symlink rejected: {relative}")
        if relative in expected_directories and not cached_path.is_dir():
            raise ValueError(f"cache hash mismatch: {relative}")
        if relative in expected_files and not cached_path.is_file():
            raise ValueError(f"cache hash mismatch: {relative}")
    for record in records:
        relative = safe_path(record.path)
        path = root
        for part in relative.parts:
            path /= part
            if path.is_symlink():
                raise ValueError(f"cache symlink rejected: {record.path}")
        if not path.is_file() or path.stat().st_size != record.bytes or sha256_bytes(path.read_bytes()) != record.sha256:
            raise ValueError(f"cache hash mismatch: {record.path}")


def _read_source_manifest(path: Path) -> SourceManifest:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("invalid source manifest: JSON") from None
    return SourceManifest.from_dict(value)


def _verify_manifest_spec(manifest: SourceManifest, spec: SourceSpec, resolved: str) -> None:
    if (
        manifest.label != spec.label
        or manifest.kind != spec.kind
        or manifest.repo_id != spec.repo_id
        or manifest.canonical_url != spec.canonical_url
        or manifest.requested_revision != spec.requested_revision
        or manifest.resolved_revision != resolved
        or manifest.license_id != spec.license_id
        or manifest.license_source != spec.license_source
        or manifest.redistributable is not spec.redistributable
        or manifest.command != spec.command
        or tuple(record.path for record in manifest.files) != spec.allowlist
    ):
        raise ValueError(f"cached source manifest mismatch: {spec.label}")


REVIDGEN_ALLOWLIST = (
    "README.md",
    "eval/4_embodiments/1_robot_subject_stability.py",
    "eval/4_embodiments/2_physical_plausibility.py",
    "eval/4_embodiments/3_task_adherence_consistency.py",
    "eval/4_embodiments/4_create_meta_info.py",
    "eval/4_embodiments/5_motion_amplitude.py",
    "eval/4_embodiments/6_motion_smoothness.py",
    "eval/4_embodiments/7_motion_total_score.py",
    "eval/4_embodiments/8_summarize_robot_results.py",
    "eval/4_embodiments/summarize_i2v_results.py",
    "eval/4_embodiments/summary_scores.py",
    "eval/5_tasks/common_manipulation.py",
    "eval/5_tasks/long-horizon_planning.py",
    "eval/5_tasks/multi-entity_collaboration.py",
    "eval/5_tasks/spatial_relationship.py",
    "eval/5_tasks/summary_scores.py",
    "eval/5_tasks/visual_reasoning.py",
    "scripts/rbench_eval_4embodiments.sh",
    "scripts/rbench_eval_5tasks.sh",
)
DATASET_ALLOWLIST = (
    "README.md",
    "prompts/common_manipulation_prompts.json",
    "prompts/dual_arm_prompts.json",
    "prompts/humanoid_prompts.json",
    "prompts/long-horizon_planning_prompts.json",
    "prompts/multi-entity_collaboration_prompts.json",
    "prompts/quad_prompts.json",
    "prompts/single_arm_prompts.json",
    "prompts/spatial_relationship_prompts.json",
    "prompts/visual_reasoning_prompts.json",
)
PAPER_SPACE_ALLOWLIST = ("README.md", "app.py", "utils.py", "leaderboard.json", "requirements.txt")
CURRENT_SPACE_ALLOWLIST = (*PAPER_SPACE_ALLOWLIST, "leaderboard_qwen.json")

SOURCE_SPECS = (
    SourceSpec(
        label="revidgen",
        kind="git",
        repo_id="DAGroup-PKU/ReVidgen",
        canonical_url="https://github.com/DAGroup-PKU/ReVidgen",
        requested_revision="b03df27f0376faa148dcd8cd620a1989a32ca979",
        allowlist=REVIDGEN_ALLOWLIST,
        license_id="NOASSERTION",
        license_source="Pinned tree has no repository-level license; GitHub reports license: null.",
        redistributable=False,
        command="git clone --filter=blob:none --no-checkout https://github.com/DAGroup-PKU/ReVidgen.git \"$workdir/ReVidgen\" && git -C \"$workdir/ReVidgen\" fetch --no-tags origin b03df27f0376faa148dcd8cd620a1989a32ca979 && git -C \"$workdir/ReVidgen\" sparse-checkout init --cone && git -C \"$workdir/ReVidgen\" sparse-checkout set eval scripts && git -C \"$workdir/ReVidgen\" checkout --detach b03df27f0376faa148dcd8cd620a1989a32ca979 && git -C \"$workdir/ReVidgen\" rev-parse --verify HEAD",
    ),
    SourceSpec(
        label="rbench-dataset",
        kind="dataset",
        repo_id="DAGroup-PKU/RBench",
        canonical_url="https://huggingface.co/datasets/DAGroup-PKU/RBench",
        requested_revision="6bdccf349ff5a8f68302428351e94f34ecd62450",
        allowlist=DATASET_ALLOWLIST,
        license_id="cc-by-4.0",
        license_source="README.md at the pinned revision",
        redistributable=True,
        command="hf download DAGroup-PKU/RBench --type dataset --revision 6bdccf349ff5a8f68302428351e94f34ecd62450 --include README.md --include 'prompts/*.json' --local-dir \"$workdir/RBench\"",
    ),
    SourceSpec(
        label="rbench-leaderboard-paper-era",
        kind="space",
        repo_id="DAGroup-PKU/RBench-Leaderboard",
        canonical_url="https://huggingface.co/spaces/DAGroup-PKU/RBench-Leaderboard",
        requested_revision="6b66282843a5d863af4271fb07ba1641d1d33334",
        allowlist=PAPER_SPACE_ALLOWLIST,
        license_id="NOASSERTION",
        license_source="Pinned card declares MIT but pinned tree has no complete permission notice.",
        redistributable=False,
        command="hf download DAGroup-PKU/RBench-Leaderboard --type space --revision 6b66282843a5d863af4271fb07ba1641d1d33334 --include README.md --include app.py --include utils.py --include leaderboard.json --include requirements.txt --local-dir \"$workdir/RBench-Leaderboard-paper-era\"",
    ),
    SourceSpec(
        label="rbench-leaderboard-current",
        kind="space",
        repo_id="DAGroup-PKU/RBench-Leaderboard",
        canonical_url="https://huggingface.co/spaces/DAGroup-PKU/RBench-Leaderboard",
        requested_revision="5dd6d55e454e22dbf7bd34ea5fbbeda5bc0f9b07",
        allowlist=CURRENT_SPACE_ALLOWLIST,
        license_id="NOASSERTION",
        license_source="Pinned card declares MIT but pinned tree has no complete permission notice.",
        redistributable=False,
        command="hf download DAGroup-PKU/RBench-Leaderboard --type space --revision 5dd6d55e454e22dbf7bd34ea5fbbeda5bc0f9b07 --include README.md --include app.py --include utils.py --include leaderboard.json --include leaderboard_qwen.json --include requirements.txt --local-dir \"$workdir/RBench-Leaderboard-current\"",
    ),
)
