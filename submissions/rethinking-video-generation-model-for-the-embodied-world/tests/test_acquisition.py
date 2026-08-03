import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from huggingface_hub import RepoFile, RepoFolder
import pytest

from rbench_repro import acquisition
from rbench_repro.acquisition import SOURCE_SPECS, TreeEntry, acquire_all, acquire_source, load_acquired
from rbench_repro.model import CLAIMS, canonical_json, sha256_bytes


ACQUIRED_AT = "2026-07-25T00:00:00+00:00"
EXACT_HUB_URL = (
    "https://huggingface.co/datasets/owner/repo/resolve/"
    + "a" * 40
    + "/README.md"
)


def test_acquire_rejects_revision_drift(source_spec, source_reader, tmp_path):
    source_reader.resolved = "f" * 40
    with pytest.raises(ValueError, match="resolved revision"):
        acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)
    assert source_reader.tree_calls == 0


@pytest.mark.parametrize(
    ("path", "kind", "error"),
    [("../escape.json", "file", "unsafe path"), ("prompts/link.json", "symlink", "symlink")],
)
def test_acquire_rejects_unsafe_tree_entries(source_spec, source_reader, tmp_path, path, kind, error):
    source_reader.entries.append((path, kind, 1))
    with pytest.raises(ValueError, match=error):
        acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)


def test_acquire_rejects_missing_allowlisted_payload(source_spec, source_reader, tmp_path):
    source_reader.entries.pop()
    with pytest.raises(ValueError, match="allowlist mismatch"):
        acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)


def test_acquire_rejects_duplicate_tree_entries(source_spec, source_reader, tmp_path):
    source_reader.entries.append(source_reader.entries[0])
    with pytest.raises(ValueError, match="duplicate tree entry"):
        acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)


def test_acquire_rejects_unknown_tree_entry_kind(source_spec, source_reader, tmp_path):
    path, _kind, size = source_reader.entries[0]
    source_reader.entries[0] = (path, "submodule", size)
    with pytest.raises(ValueError, match="invalid tree entry kind"):
        acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)


@pytest.mark.parametrize("size", [1_048_577, 8_388_609])
def test_acquire_enforces_file_and_total_limits(source_spec, source_reader, tmp_path, size):
    path = source_spec.allowlist[0]
    source_reader.payloads[path] = b"x" * size
    source_reader.entries[0] = (path, "file", size)
    with pytest.raises(ValueError, match="byte limit"):
        acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)


def test_acquire_rejects_declared_and_actual_size_mismatch(source_spec, source_reader, tmp_path):
    path, kind, size = source_reader.entries[0]
    source_reader.entries[0] = (path, kind, size + 1)
    with pytest.raises(ValueError, match="declared size"):
        acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)


def test_acquire_checks_actual_payload_limit_before_size_mismatch(source_spec, source_reader, tmp_path):
    source_reader.payloads[source_spec.allowlist[0]] = b"x" * 1_048_577
    with pytest.raises(ValueError, match="byte limit"):
        acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)


def test_acquire_hashes_stable_bytes_and_uses_content_addressed_cache(source_spec, source_reader, tmp_path):
    first = acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)
    second = acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)
    assert first.manifest == second.manifest
    assert first.root == second.root
    assert all(record.sha256 for record in first.manifest.files)
    assert source_reader.read_calls == len(source_spec.allowlist)


def test_manifest_persists_full_tree_without_fetching_media(source_spec, source_reader, tmp_path):
    source_reader.entries.extend(
        [("imgs", "directory", 0), ("imgs/known.png", "file", 22_000_000_000)]
    )
    acquired = acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)
    assert [record.path for record in acquired.manifest.files] == list(source_spec.allowlist)
    tree = {entry.path: entry.to_dict() for entry in acquired.manifest.tree}
    assert tree["imgs"] == {"kind": "directory", "path": "imgs", "size": 0}
    assert tree["imgs/known.png"] == {
        "kind": "file",
        "path": "imgs/known.png",
        "size": 22_000_000_000,
    }
    assert not (acquired.root / "imgs/known.png").exists()
    assert source_reader.read_calls == len(source_spec.allowlist)


def test_cached_manifest_is_bound_to_exact_full_tree(source_spec, source_reader, tmp_path):
    acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)
    source_reader.entries.append(("imgs/new.png", "file", 10))
    with pytest.raises(ValueError, match="tree metadata mismatch"):
        acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)


def test_acquire_rejects_tampered_cache(source_spec, source_reader, tmp_path):
    acquired = acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)
    (acquired.root / source_spec.allowlist[0]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cache hash"):
        acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)


def test_acquire_rejects_unindexed_existing_cache_corruption(source_spec, source_reader, tmp_path):
    acquired = acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)
    next((tmp_path / ".indexes").iterdir()).unlink()
    cached = acquired.root / source_spec.allowlist[0]
    cached.write_bytes(b"x" * cached.stat().st_size)
    with pytest.raises(ValueError, match="cache hash"):
        acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)


def test_acquire_rejects_cache_symlinks(source_spec, source_reader, tmp_path):
    acquired = acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)
    cached = acquired.root / source_spec.allowlist[0]
    external = tmp_path / "external"
    external.write_bytes(cached.read_bytes())
    cached.unlink()
    cached.symlink_to(external)
    with pytest.raises(ValueError, match="cache symlink"):
        acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_acquire_rejects_undeclared_cache_entries(source_spec, source_reader, tmp_path, kind):
    acquired = acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)
    undeclared = acquired.root / "undeclared"
    if kind == "file":
        undeclared.write_bytes(b"extra")
    elif kind == "directory":
        undeclared.mkdir()
    else:
        external = tmp_path / "external-extra"
        external.write_bytes(b"extra")
        undeclared.symlink_to(external)
    with pytest.raises(ValueError, match="undeclared cache entry"):
        acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("label", "other-label"),
        ("kind", "space"),
        ("canonical_url", "https://other.test/repo"),
        ("license_id", "other-license"),
        ("license_source", "OTHER.md"),
        ("redistributable", False),
        ("command", "other command"),
    ],
)
def test_cache_identity_includes_complete_source_spec(source_spec, source_reader, tmp_path, field, value):
    acquire_source(source_spec, tmp_path, source_reader, ACQUIRED_AT)
    changed = replace(source_spec, **{field: value})
    acquired = acquire_source(changed, tmp_path, source_reader, ACQUIRED_AT)
    assert getattr(acquired.manifest, field) == value
    assert source_reader.read_calls == 2 * len(source_spec.allowlist)


def test_manifest_is_exact_and_contains_no_source_bytes(source_spec, source_reader, tmp_path):
    result = acquire_source(replace(source_spec, redistributable=False), tmp_path, source_reader, ACQUIRED_AT)
    manifest = result.manifest.to_dict()
    assert manifest == {
        "acquired_at": ACQUIRED_AT,
        "canonical_url": source_spec.canonical_url,
        "command": source_spec.command,
        "files": [record.to_dict() for record in result.manifest.files],
        "kind": "dataset",
        "label": "fixture",
        "license_id": "cc-by-4.0",
        "license_source": "README.md",
        "redistributable": False,
        "repo_id": "owner/repo",
        "requested_revision": "a" * 40,
        "resolved_revision": "a" * 40,
        "tree": [entry.to_dict() for entry in result.manifest.tree],
    }
    assert "source text sentinel" not in json.dumps(manifest)
    assert result.root.is_relative_to(tmp_path)


def test_load_acquired_rehashes_files(monkeypatch, tmp_path):
    monkeypatch.setattr(acquisition, "GitReader", lambda staging_root: SpecReader())
    monkeypatch.setattr(acquisition, "HubReader", lambda staging_root: SpecReader())
    manifest_path = tmp_path / "project" / "evidence" / "input-manifest.json"
    cache_root = tmp_path / "cache"
    results = acquire_all(cache_root, manifest_path, ACQUIRED_AT)
    loaded = load_acquired(manifest_path, tmp_path / "cache")
    assert loaded == {result.manifest.label: result for result in results}
    result = results[0]
    (result.root / result.manifest.files[0].path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cache hash"):
        load_acquired(manifest_path, tmp_path / "cache")


def manifest_for_spec(spec):
    return {
        "acquired_at": ACQUIRED_AT,
        "canonical_url": spec.canonical_url,
        "command": spec.command,
        "files": [
            {"bytes": 0, "path": path, "sha256": sha256_bytes(b"")} for path in spec.allowlist
        ],
        "kind": spec.kind,
        "label": spec.label,
        "license_id": spec.license_id,
        "license_source": spec.license_source,
        "redistributable": spec.redistributable,
        "repo_id": spec.repo_id,
        "requested_revision": spec.requested_revision,
        "resolved_revision": spec.requested_revision,
        "tree": [{"kind": "file", "path": path, "size": 0} for path in spec.allowlist],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "space"),
        ("repo_id", "substituted/repository"),
        ("canonical_url", "https://example.test/substituted"),
        ("requested_revision", "f" * 40),
        ("files", None),
        ("command", "substituted command"),
        ("license_id", "substituted-license"),
        ("license_source", "SUBSTITUTED.md"),
        ("redistributable", True),
    ],
)
def test_load_acquired_rejects_substituted_source_spec_before_cache(
    monkeypatch, tmp_path, field, value
):
    sources = [manifest_for_spec(spec) for spec in SOURCE_SPECS]
    source = sources[0]
    if field == "requested_revision":
        source["requested_revision"] = value
        source["resolved_revision"] = value
    elif field == "files":
        source["files"] = source["files"][:-1]
    elif field == "redistributable":
        source[field] = not source[field]
    else:
        source[field] = value
    manifest_path = tmp_path / "input-manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "sources": sources}))
    monkeypatch.setattr(
        acquisition,
        "_verify_cache",
        lambda *args: pytest.fail("substituted specs must fail before cache verification"),
    )
    with pytest.raises(ValueError, match="source manifest mismatch"):
        load_acquired(manifest_path, tmp_path / "cache")


@pytest.mark.parametrize("change", ["missing", "extra", "substituted"])
def test_load_acquired_requires_exact_source_labels_before_cache(monkeypatch, tmp_path, change):
    sources = [manifest_for_spec(spec) for spec in SOURCE_SPECS]
    if change == "missing":
        sources.pop()
    elif change == "extra":
        extra = deepcopy(sources[-1])
        extra["label"] = "extra-source"
        sources.append(extra)
    else:
        sources[-1]["label"] = "substituted-source"
    manifest_path = tmp_path / "input-manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "sources": sources}))
    monkeypatch.setattr(
        acquisition,
        "_verify_cache",
        lambda *args: pytest.fail("label mismatches must fail before cache verification"),
    )
    with pytest.raises(ValueError, match="source labels mismatch"):
        load_acquired(manifest_path, tmp_path / "cache")


def test_load_acquired_rejects_malformed_json(tmp_path):
    manifest_path = tmp_path / "input-manifest.json"
    manifest_path.write_text("{")
    with pytest.raises(ValueError, match="invalid input manifest"):
        load_acquired(manifest_path, tmp_path / "cache")


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        {"schema_version": True, "sources": [{}]},
        {"schema_version": "1", "sources": [{}]},
        {"schema_version": 2, "sources": [{}]},
        {"schema_version": 1, "sources": []},
        {"schema_version": 1, "sources": {}},
        {"schema_version": 1, "sources": [{}], "extra": True},
    ],
)
def test_load_acquired_rejects_invalid_top_level_schema(tmp_path, value):
    manifest_path = tmp_path / "input-manifest.json"
    manifest_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="invalid input manifest"):
        load_acquired(manifest_path, tmp_path / "cache")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("label", 1),
        ("label", ""),
        ("kind", "model"),
        ("kind", []),
        ("repo_id", None),
        ("repo_id", ""),
        ("canonical_url", []),
        ("requested_revision", "a" * 39),
        ("resolved_revision", "b" * 40),
        ("acquired_at", 1),
        ("license_id", ""),
        ("license_source", False),
        ("redistributable", 0),
        ("redistributable", "false"),
        ("command", ""),
        ("files", []),
        ("files", {}),
        ("tree", []),
        ("tree", {}),
    ],
)
def test_load_acquired_rejects_invalid_source_fields(source_spec, source_reader, tmp_path, field, value):
    result = acquire_source(source_spec, tmp_path / "cache", source_reader, ACQUIRED_AT)
    source = result.manifest.to_dict()
    source[field] = value
    manifest_path = tmp_path / "input-manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "sources": [source]}))
    with pytest.raises(ValueError, match="invalid source manifest"):
        load_acquired(manifest_path, tmp_path / "cache")


def test_load_acquired_rejects_missing_and_extra_source_fields(source_spec, source_reader, tmp_path):
    result = acquire_source(source_spec, tmp_path / "cache", source_reader, ACQUIRED_AT)
    source = result.manifest.to_dict()
    source.pop("command")
    source["extra"] = True
    manifest_path = tmp_path / "input-manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "sources": [source]}))
    with pytest.raises(ValueError, match="invalid source manifest"):
        load_acquired(manifest_path, tmp_path / "cache")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "../escape"),
        ("path", 1),
        ("bytes", True),
        ("bytes", "15"),
        ("bytes", -1),
        ("sha256", "f" * 63),
        ("sha256", "F" * 64),
        ("sha256", 1),
    ],
)
def test_load_acquired_rejects_invalid_file_records(source_spec, source_reader, tmp_path, field, value):
    result = acquire_source(source_spec, tmp_path / "cache", source_reader, ACQUIRED_AT)
    source = result.manifest.to_dict()
    source["files"][0][field] = value
    manifest_path = tmp_path / "input-manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "sources": [source]}))
    with pytest.raises(ValueError, match="invalid file record"):
        load_acquired(manifest_path, tmp_path / "cache")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "../escape"),
        ("path", 1),
        ("kind", "submodule"),
        ("kind", 1),
        ("size", True),
        ("size", "15"),
        ("size", -1),
    ],
)
def test_load_acquired_rejects_invalid_tree_entries(
    source_spec, source_reader, tmp_path, field, value
):
    result = acquire_source(source_spec, tmp_path / "cache", source_reader, ACQUIRED_AT)
    source = result.manifest.to_dict()
    source["tree"][0][field] = value
    manifest_path = tmp_path / "input-manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "sources": [source]}))
    with pytest.raises(ValueError, match="invalid tree entry"):
        load_acquired(manifest_path, tmp_path / "cache")


def test_load_acquired_rejects_duplicate_and_unbound_tree_entries(
    source_spec, source_reader, tmp_path
):
    result = acquire_source(source_spec, tmp_path / "cache", source_reader, ACQUIRED_AT)
    source = result.manifest.to_dict()
    source["tree"].append(deepcopy(source["tree"][0]))
    manifest_path = tmp_path / "input-manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "sources": [source]}))
    with pytest.raises(ValueError, match="duplicate tree entry"):
        load_acquired(manifest_path, tmp_path / "cache")

    source = result.manifest.to_dict()
    source["tree"][0]["size"] += 1
    manifest_path.write_text(json.dumps({"schema_version": 1, "sources": [source]}))
    with pytest.raises(ValueError, match="file/tree mismatch"):
        load_acquired(manifest_path, tmp_path / "cache")


def test_load_acquired_rejects_duplicate_files_and_sources(source_spec, source_reader, tmp_path):
    result = acquire_source(source_spec, tmp_path / "cache", source_reader, ACQUIRED_AT)
    source = result.manifest.to_dict()
    source["files"].append(deepcopy(source["files"][0]))
    manifest_path = tmp_path / "input-manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "sources": [source]}))
    with pytest.raises(ValueError, match="duplicate file"):
        load_acquired(manifest_path, tmp_path / "cache")

    valid_source = result.manifest.to_dict()
    manifest_path.write_text(json.dumps({"schema_version": 1, "sources": [valid_source, valid_source]}))
    with pytest.raises(ValueError, match="duplicate source"):
        load_acquired(manifest_path, tmp_path / "cache")


class HubApi:
    def __init__(self, entries):
        self.entries = entries

    def list_repo_tree(self, **kwargs):
        return self.entries


class StreamResponse:
    def __init__(
        self,
        chunks,
        captured,
        url=EXACT_HUB_URL,
        history=(),
        status_code=200,
        headers=None,
    ):
        self.chunks = chunks
        self.captured = captured
        self.url = httpx.URL(url)
        self.history = tuple(history)
        self.status_code = status_code
        self.headers = httpx.Headers(headers)

    async def __aenter__(self):
        self.captured["response_entered"] = True
        return self

    async def __aexit__(self, *args):
        self.captured["response_entered"] = False
        self.captured["response_closed"] = True

    def raise_for_status(self):
        self.captured["status_checked"] = True

    async def aiter_bytes(self, chunk_size=None):
        assert self.captured["response_entered"]
        self.captured["body_reads"] = self.captured.get("body_reads", 0) + 1
        for chunk in self.chunks:
            yield chunk


class StreamClient:
    def __init__(self, response, captured, **kwargs):
        self.response = response
        self.captured = captured
        self.captured["client_options"] = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.captured["client_closed"] = True

    def stream(self, method, url, **kwargs):
        self.captured["request"] = (method, url, kwargs)
        return self.response


class RedirectClient(StreamClient):
    def __init__(self, routes, captured, **kwargs):
        super().__init__(None, captured, **kwargs)
        self.routes = routes

    def stream(self, method, url, **kwargs):
        current_url = httpx.URL(url)
        self.captured.setdefault("requests", []).append(str(current_url))
        response = self.routes[str(current_url)]
        if not self.captured["client_options"]["follow_redirects"]:
            return response

        history = []
        seen = {str(current_url)}
        while response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("location")
            if location is None:
                break
            history.append(SimpleNamespace(url=response.url))
            current_url = current_url.join(location)
            self.captured["requests"].append(str(current_url))
            if str(current_url) in seen:
                break
            seen.add(str(current_url))
            response = self.routes[str(current_url)]
        response.history = tuple(history)
        return response


def test_hub_reader_rejects_unknown_file_size(source_spec, tmp_path):
    reader = acquisition.HubReader(tmp_path)
    reader.api = HubApi((RepoFile(path="README.md", size=None, oid="f" * 40),))
    spec = replace(source_spec, allowlist=("README.md",))
    with pytest.raises(ValueError, match="invalid Hub tree entry"):
        reader.tree(spec, spec.requested_revision)


def test_hub_reader_returns_full_file_and_directory_metadata(source_spec, tmp_path):
    reader = acquisition.HubReader(tmp_path)
    reader.api = HubApi(
        (
            RepoFile(path="README.md", size=7, oid="f" * 40),
            RepoFolder(path="imgs", oid="f" * 40),
            RepoFile(path="imgs/known.png", size=22_000_000_000, oid="f" * 40),
        )
    )
    spec = replace(source_spec, allowlist=("README.md",))
    assert reader.tree(spec, spec.requested_revision) == (
        TreeEntry("README.md", "file", 7),
        TreeEntry("imgs", "directory", 0),
        TreeEntry("imgs/known.png", "file", 22_000_000_000),
    )


def test_hub_reader_enforces_payload_timeout(monkeypatch, source_spec, tmp_path):
    captured = {}
    response = StreamResponse((b"pay", b"load"), captured)

    def hub_url(**kwargs):
        captured["hub_url"] = kwargs
        return EXACT_HUB_URL

    def headers(**kwargs):
        captured["header_options"] = kwargs
        return {"authorization": "Bearer test"}

    monkeypatch.setattr(
        acquisition,
        "set_client_factory",
        lambda factory: pytest.fail("must not mutate the process-global Hub client"),
        raising=False,
    )
    monkeypatch.setattr(acquisition, "hf_hub_url", hub_url, raising=False)
    monkeypatch.setattr(acquisition, "build_hf_headers", headers, raising=False)
    monkeypatch.setattr(
        acquisition.httpx,
        "Client",
        lambda **kwargs: pytest.fail("Hub payload reads must use cancellable async I/O"),
    )
    monkeypatch.setattr(
        acquisition.httpx,
        "AsyncClient",
        lambda **kwargs: StreamClient(response, captured, **kwargs),
    )
    reader = acquisition.HubReader(tmp_path / "staging")
    reader.api = HubApi((RepoFile(path="README.md", size=7, oid="f" * 40),))
    spec = replace(source_spec, allowlist=("README.md",))
    reader.tree(spec, spec.requested_revision)
    assert reader.read(spec, spec.requested_revision, "README.md", 30) == b"payload"
    assert captured["hub_url"] == {
        "filename": "README.md",
        "repo_id": source_spec.repo_id,
        "repo_type": source_spec.kind,
        "revision": source_spec.requested_revision,
    }
    timeout = captured["client_options"]["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (30, 30, 30, 30)
    assert captured["client_options"]["follow_redirects"] is False
    assert captured["request"] == (
        "GET",
        EXACT_HUB_URL,
        {"headers": {"authorization": "Bearer test"}},
    )
    assert captured["status_checked"]
    assert captured["response_closed"]
    assert captured["client_closed"]


@pytest.mark.parametrize(
    ("declared_size", "chunks", "error"),
    [
        (3, (b"ab", b"cd"), "declared size"),
        (1_048_586, (b"x" * 1_048_576, b"x"), "byte limit"),
    ],
)
def test_hub_reader_bounds_stream_before_buffering(
    monkeypatch,
    source_spec,
    tmp_path,
    declared_size,
    chunks,
    error,
):
    captured = {}
    response = StreamResponse(chunks, captured)
    monkeypatch.setattr(acquisition, "hf_hub_url", lambda **kwargs: EXACT_HUB_URL)
    monkeypatch.setattr(acquisition, "build_hf_headers", lambda **kwargs: {})
    monkeypatch.setattr(
        acquisition.httpx,
        "Client",
        lambda **kwargs: pytest.fail("Hub payload reads must use cancellable async I/O"),
    )
    monkeypatch.setattr(
        acquisition.httpx,
        "AsyncClient",
        lambda **kwargs: StreamClient(response, captured, **kwargs),
    )
    reader = acquisition.HubReader(tmp_path)
    reader.api = HubApi((RepoFile(path="README.md", size=declared_size, oid="f" * 40),))
    spec = replace(source_spec, allowlist=("README.md",))
    reader.tree(spec, spec.requested_revision)
    with pytest.raises(ValueError, match=error):
        reader.read(spec, spec.requested_revision, "README.md", 30)
    assert captured["response_closed"]
    assert captured["client_closed"]


def test_hub_reader_enforces_absolute_transfer_deadline(monkeypatch, source_spec, tmp_path):
    captured = {}

    class BlockedResponse(StreamResponse):
        async def aiter_bytes(self, chunk_size=None):
            yield b"first"
            await asyncio.Event().wait()

    response = BlockedResponse((), captured)
    monkeypatch.setattr(acquisition, "hf_hub_url", lambda **kwargs: EXACT_HUB_URL)
    monkeypatch.setattr(acquisition, "build_hf_headers", lambda **kwargs: {})
    monkeypatch.setattr(
        acquisition.httpx,
        "Client",
        lambda **kwargs: pytest.fail("wall-clock bound must use cancellable async I/O"),
    )
    monkeypatch.setattr(
        acquisition.httpx,
        "AsyncClient",
        lambda **kwargs: StreamClient(response, captured, **kwargs),
    )
    reader = acquisition.HubReader(tmp_path)
    reader.api = HubApi((RepoFile(path="README.md", size=11, oid="f" * 40),))
    spec = replace(source_spec, allowlist=("README.md",))
    reader.tree(spec, spec.requested_revision)
    with pytest.raises(TimeoutError, match="absolute transfer deadline"):
        reader.read(spec, spec.requested_revision, "README.md", 0.01)
    assert captured["response_closed"]
    assert captured["client_closed"]


@pytest.mark.parametrize(
    ("location", "final_url"),
    [
        (
            "/api/resolve-cache/datasets/owner/repo/" + "a" * 40 + "/README.md?etag=test",
            "https://huggingface.co/api/resolve-cache/datasets/owner/repo/"
            + "a" * 40
            + "/README.md?etag=test",
        ),
        (
            "https://cdn-lfs.hf.co/repos/ab/cd/content?signature=test",
            "https://cdn-lfs.hf.co/repos/ab/cd/content?signature=test",
        ),
        (
            "https://cdn-lfs-us-1.hf.co/repos/ab/cd/content?signature=test",
            "https://cdn-lfs-us-1.hf.co/repos/ab/cd/content?signature=test",
        ),
        (
            "https://cdn-lfs-eu-1.hf.co/repos/ab/cd/content?signature=test",
            "https://cdn-lfs-eu-1.hf.co/repos/ab/cd/content?signature=test",
        ),
        (
            "https://cas-bridge.xethub.hf.co/xet-bridge-us/ab/cd/content?signature=test",
            "https://cas-bridge.xethub.hf.co/xet-bridge-us/ab/cd/content?signature=test",
        ),
    ],
)
def test_hub_reader_accepts_approved_redirect_chain(
    monkeypatch, source_spec, tmp_path, location, final_url
):
    captured = {}
    redirect = StreamResponse(
        (b"redirect body must not be read",),
        captured,
        status_code=302,
        headers={"location": location},
    )
    response = StreamResponse(
        (b"payload",),
        captured,
        url=final_url,
    )
    routes = {EXACT_HUB_URL: redirect, final_url: response}
    monkeypatch.setattr(acquisition, "hf_hub_url", lambda **kwargs: EXACT_HUB_URL)
    monkeypatch.setattr(acquisition, "build_hf_headers", lambda **kwargs: {})
    monkeypatch.setattr(
        acquisition.httpx,
        "AsyncClient",
        lambda **kwargs: RedirectClient(routes, captured, **kwargs),
    )
    reader = acquisition.HubReader(tmp_path)
    reader.api = HubApi((RepoFile(path="README.md", size=7, oid="f" * 40),))
    spec = replace(source_spec, allowlist=("README.md",))
    reader.tree(spec, spec.requested_revision)
    assert reader.read(spec, spec.requested_revision, "README.md", 30) == b"payload"
    assert captured["body_reads"] == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/datasets/owner/repo/resolve/" + "a" * 40 + "/README.md",
        "https://evil.example/repos/ab/cd/content",
        "https://cdn-lfs-us-1.hf.co/not-repos/content",
        "https://cas-bridge.xethub.hf.co/not-xet/content",
        "https://user@cdn-lfs.hf.co/repos/ab/cd/content",
        "https://cdn-lfs.hf.co:444/repos/ab/cd/content",
        "https://cdn-lfs.hf.co/repos/ab/%2e%2e/content",
        "https://cdn-lfs.hf.co/repos/ab/%252e%252e/content",
        "not a URL",
    ],
)
def test_hub_reader_rejects_unapproved_final_urls(monkeypatch, source_spec, tmp_path, url):
    captured = {}
    response = StreamResponse(
        (),
        captured,
        status_code=302,
        headers={"location": url},
    )
    routes = {
        EXACT_HUB_URL: response,
        url: StreamResponse((b"payload",), captured, url=url),
    }
    monkeypatch.setattr(acquisition, "hf_hub_url", lambda **kwargs: EXACT_HUB_URL)
    monkeypatch.setattr(acquisition, "build_hf_headers", lambda **kwargs: {})
    monkeypatch.setattr(
        acquisition.httpx,
        "AsyncClient",
        lambda **kwargs: RedirectClient(routes, captured, **kwargs),
    )
    reader = acquisition.HubReader(tmp_path)
    reader.api = HubApi((RepoFile(path="README.md", size=7, oid="f" * 40),))
    spec = replace(source_spec, allowlist=("README.md",))
    reader.tree(spec, spec.requested_revision)
    with pytest.raises(ValueError, match="unsafe Hub URL"):
        reader.read(spec, spec.requested_revision, "README.md", 30)


def test_hub_reader_rejects_unapproved_intermediate_redirect(monkeypatch, source_spec, tmp_path):
    captured = {}
    unsafe_url = "https://evil.example/intermediate"
    response = StreamResponse(
        (),
        captured,
        status_code=302,
        headers={"location": unsafe_url},
    )
    routes = {
        EXACT_HUB_URL: response,
        unsafe_url: StreamResponse((b"payload",), captured, url=unsafe_url),
    }
    monkeypatch.setattr(acquisition, "hf_hub_url", lambda **kwargs: EXACT_HUB_URL)
    monkeypatch.setattr(acquisition, "build_hf_headers", lambda **kwargs: {})
    monkeypatch.setattr(
        acquisition.httpx,
        "AsyncClient",
        lambda **kwargs: RedirectClient(routes, captured, **kwargs),
    )
    reader = acquisition.HubReader(tmp_path)
    reader.api = HubApi((RepoFile(path="README.md", size=7, oid="f" * 40),))
    spec = replace(source_spec, allowlist=("README.md",))
    reader.tree(spec, spec.requested_revision)
    with pytest.raises(ValueError, match="unsafe Hub URL"):
        reader.read(spec, spec.requested_revision, "README.md", 30)
    assert captured["requests"] == [EXACT_HUB_URL]
    assert captured.get("body_reads", 0) == 0


def test_hub_reader_rejects_redirect_without_location(monkeypatch, source_spec, tmp_path):
    captured = {}
    routes = {
        EXACT_HUB_URL: StreamResponse((), captured, status_code=302),
    }
    monkeypatch.setattr(acquisition, "hf_hub_url", lambda **kwargs: EXACT_HUB_URL)
    monkeypatch.setattr(acquisition, "build_hf_headers", lambda **kwargs: {})
    monkeypatch.setattr(
        acquisition.httpx,
        "AsyncClient",
        lambda **kwargs: RedirectClient(routes, captured, **kwargs),
    )
    reader = acquisition.HubReader(tmp_path)
    reader.api = HubApi((RepoFile(path="README.md", size=7, oid="f" * 40),))
    spec = replace(source_spec, allowlist=("README.md",))
    reader.tree(spec, spec.requested_revision)
    with pytest.raises(ValueError, match="missing redirect Location"):
        reader.read(spec, spec.requested_revision, "README.md", 30)
    assert captured.get("body_reads", 0) == 0


def test_hub_reader_rejects_redirect_loop(monkeypatch, source_spec, tmp_path):
    captured = {}
    cache_url = (
        "https://huggingface.co/api/resolve-cache/datasets/owner/repo/"
        + "a" * 40
        + "/README.md"
    )
    routes = {
        EXACT_HUB_URL: StreamResponse(
            (), captured, status_code=302, headers={"location": cache_url}
        ),
        cache_url: StreamResponse(
            (),
            captured,
            url=cache_url,
            status_code=307,
            headers={"location": EXACT_HUB_URL},
        ),
    }
    monkeypatch.setattr(acquisition, "hf_hub_url", lambda **kwargs: EXACT_HUB_URL)
    monkeypatch.setattr(acquisition, "build_hf_headers", lambda **kwargs: {})
    monkeypatch.setattr(
        acquisition.httpx,
        "AsyncClient",
        lambda **kwargs: RedirectClient(routes, captured, **kwargs),
    )
    reader = acquisition.HubReader(tmp_path)
    reader.api = HubApi((RepoFile(path="README.md", size=7, oid="f" * 40),))
    spec = replace(source_spec, allowlist=("README.md",))
    reader.tree(spec, spec.requested_revision)
    with pytest.raises(ValueError, match="redirect loop"):
        reader.read(spec, spec.requested_revision, "README.md", 30)
    assert captured.get("body_reads", 0) == 0


def test_hub_reader_rejects_excess_redirects(monkeypatch, source_spec, tmp_path):
    captured = {}
    urls = [EXACT_HUB_URL] + [f"https://cdn-lfs.hf.co/repos/{index}/content" for index in range(7)]
    routes = {
        current: StreamResponse(
            (),
            captured,
            url=current,
            status_code=302,
            headers={"location": following},
        )
        for current, following in zip(urls[:-1], urls[1:], strict=True)
    }
    routes[urls[-1]] = StreamResponse((b"payload",), captured, url=urls[-1])
    monkeypatch.setattr(acquisition, "hf_hub_url", lambda **kwargs: EXACT_HUB_URL)
    monkeypatch.setattr(acquisition, "build_hf_headers", lambda **kwargs: {})
    monkeypatch.setattr(
        acquisition.httpx,
        "AsyncClient",
        lambda **kwargs: RedirectClient(routes, captured, **kwargs),
    )
    reader = acquisition.HubReader(tmp_path)
    reader.api = HubApi((RepoFile(path="README.md", size=7, oid="f" * 40),))
    spec = replace(source_spec, allowlist=("README.md",))
    reader.tree(spec, spec.requested_revision)
    with pytest.raises(ValueError, match="too many Hub redirects"):
        reader.read(spec, spec.requested_revision, "README.md", 30)
    assert len(captured["requests"]) == acquisition.MAX_HUB_REDIRECTS + 1
    assert captured.get("body_reads", 0) == 0


def test_hub_reader_rejects_malformed_original_resolve_url(monkeypatch, source_spec, tmp_path):
    monkeypatch.setattr(acquisition, "hf_hub_url", lambda **kwargs: "https://evil.example/file")
    monkeypatch.setattr(
        acquisition.httpx,
        "AsyncClient",
        lambda **kwargs: pytest.fail("unsafe original URL must fail before the request"),
    )
    reader = acquisition.HubReader(tmp_path)
    reader.api = HubApi((RepoFile(path="README.md", size=7, oid="f" * 40),))
    spec = replace(source_spec, allowlist=("README.md",))
    reader.tree(spec, spec.requested_revision)
    with pytest.raises(ValueError, match="unsafe Hub resolve URL"):
        reader.read(spec, spec.requested_revision, "README.md", 30)


class FailingReader:
    def resolve(self, spec):
        raise RuntimeError("secret?token=do-not-leak")


def test_acquire_sanitizes_upstream_failures(source_spec, tmp_path):
    with pytest.raises(ValueError, match=r"^acquisition failed: fixture: RuntimeError$") as error:
        acquire_source(source_spec, tmp_path, FailingReader(), ACQUIRED_AT)
    assert "token" not in str(error.value)


def test_shared_constants_and_canonical_json_are_stable():
    assert len(CLAIMS) == 3
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'
    assert sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    with pytest.raises(ValueError):
        canonical_json({"invalid": float("nan")})


def test_source_specs_use_exact_distinct_pins_and_allowlists():
    assert [(spec.label, spec.requested_revision) for spec in SOURCE_SPECS] == [
        ("revidgen", "b03df27f0376faa148dcd8cd620a1989a32ca979"),
        ("rbench-dataset", "6bdccf349ff5a8f68302428351e94f34ecd62450"),
        ("rbench-leaderboard-paper-era", "6b66282843a5d863af4271fb07ba1641d1d33334"),
        ("rbench-leaderboard-current", "5dd6d55e454e22dbf7bd34ea5fbbeda5bc0f9b07"),
    ]
    assert len(SOURCE_SPECS[1].allowlist) == 10
    assert "leaderboard_qwen.json" not in SOURCE_SPECS[2].allowlist
    assert "leaderboard_qwen.json" in SOURCE_SPECS[3].allowlist
    assert SOURCE_SPECS[1].redistributable
    assert not any(spec.redistributable for spec in (SOURCE_SPECS[0], *SOURCE_SPECS[2:]))


class SpecReader:
    def resolve(self, spec):
        return spec.requested_revision

    def tree(self, spec, revision):
        return tuple(TreeEntry(path, "file", len(path.encode())) for path in spec.allowlist)

    def read(self, spec, revision, path, timeout_seconds):
        return path.encode()


def test_repository_has_no_invalid_empty_manifest_placeholder():
    project_root = Path(__file__).parents[1]
    manifest_path = project_root / "evidence" / "input-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        assert manifest.get("schema_version") == 1
        assert manifest.get("sources")


def test_acquire_all_vendors_only_redistributable_dataset_files(monkeypatch, tmp_path):
    monkeypatch.setattr(acquisition, "GitReader", lambda staging_root: SpecReader())
    monkeypatch.setattr(acquisition, "HubReader", lambda staging_root: SpecReader())
    manifest_path = tmp_path / "project" / "evidence" / "input-manifest.json"
    cache_root = tmp_path / "cache"
    assert not manifest_path.exists()
    acquired = acquire_all(cache_root, manifest_path, ACQUIRED_AT)
    assert len(acquired) == 4
    assert json.loads(manifest_path.read_bytes())["sources"] == [source.manifest.to_dict() for source in acquired]
    assert load_acquired(manifest_path, cache_root) == {source.manifest.label: source for source in acquired}
    inputs = manifest_path.parent.parent / "inputs" / "rbench"
    assert (inputs / "UPSTREAM_README.md").read_bytes() == b"README.md"
    assert sorted(path.name for path in (inputs / "prompts").iterdir()) == sorted(
        path.rsplit("/", 1)[1] for path in SOURCE_SPECS[1].allowlist if path.startswith("prompts/")
    )
    assert not (inputs / "app.py").exists()
