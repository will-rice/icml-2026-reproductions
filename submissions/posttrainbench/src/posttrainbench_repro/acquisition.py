"""Pinned artifact acquisition for PostTrainBench reproduction.

Fetches metadata and evidence files from GitHub and Hugging Face at exact
pinned revisions.  All network access is confined to this module; downstream
consumers work exclusively from the returned data structures.

No mutable ``main`` reference, no unpinned URL, and no credential is used.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

from posttrainbench_repro.constants import (
    GITHUB_PINNED_COMMIT,
    GITHUB_REPO,
    GIT_TREE_DIGEST,
    GIT_TREE_ENTRY_COUNT,
    GIT_TREE_ID,
    HF_ALLOWLISTED_FILES,
    HF_DATASET_ID,
    HF_PINNED_REVISION,
    HF_TREE_DIR_COUNT,
    HF_TREE_FILE_COUNT,
    HF_TREE_PAGE_SIZE,
    HF_TREE_TOTAL_ENTRIES,
    HF_TREE_TOTAL_PAGES,
    PINNED_BLOBS,
    TRACE_EXCERPTS,
)

_TIMEOUT = httpx.Timeout(60.0, connect=30.0)

_EXPECTED_HF_BASE = (
    f"https://huggingface.co/api/datasets/{HF_DATASET_ID}"
    f"/tree/{HF_PINNED_REVISION}"
)


# ---------------------------------------------------------------------------
# Canonical tree digest computation
# ---------------------------------------------------------------------------

def compute_canonical_tree_digest(
    entries: list[dict[str, Any]],
) -> str:
    """Compute SHA-256 over the canonical ``<path>\\t<type>\\t<sha>\\t<size>\\n`` form."""
    lines: list[str] = []
    for e in sorted(entries, key=lambda x: x["path"]):
        size_field = str(e["size"]) if e.get("size") is not None else ""
        lines.append(f"{e['path']}\t{e['type']}\t{e['sha']}\t{size_field}\n")
    canonical = "".join(lines).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# ---------------------------------------------------------------------------
# GitHub tree fetch
# ---------------------------------------------------------------------------

def fetch_github_tree_entries(
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Fetch the recursive Git tree at the pinned commit via GitHub API."""
    url = (
        f"https://api.github.com/repos/{GITHUB_REPO}"
        f"/git/trees/{GITHUB_PINNED_COMMIT}?recursive=1"
    )
    if client is None:
        with httpx.Client(timeout=_TIMEOUT) as c:
            resp = c.get(url, headers={"Accept": "application/vnd.github+json"})
    else:
        resp = client.get(url, headers={"Accept": "application/vnd.github+json"})
    resp.raise_for_status()
    data = resp.json()
    if data.get("truncated"):
        raise RuntimeError("GitHub recursive tree was truncated")
    if data["sha"] != GIT_TREE_ID:
        raise ValueError(
            f"Tree SHA mismatch: got {data['sha']}, expected {GIT_TREE_ID}"
        )
    return data["tree"]


def fetch_github_raw_file(
    path: str,
    *,
    client: httpx.Client | None = None,
) -> bytes:
    """Fetch raw file bytes from GitHub at the pinned commit."""
    url = (
        f"https://raw.githubusercontent.com/{GITHUB_REPO}"
        f"/{GITHUB_PINNED_COMMIT}/{path}"
    )
    if client is None:
        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as c:
            resp = c.get(url)
    else:
        resp = client.get(url, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def fetch_github_metadata(
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Return verified tree metadata, blob digests, and blob contents.

    Blob bytes are fetched once and cached in the returned dict so that
    downstream protocol audit does not need to refetch them.
    """
    entries = fetch_github_tree_entries(client)
    if len(entries) != GIT_TREE_ENTRY_COUNT:
        raise ValueError(
            f"Expected {GIT_TREE_ENTRY_COUNT} tree entries, got {len(entries)}"
        )
    tree_digest = compute_canonical_tree_digest(entries)
    if tree_digest != GIT_TREE_DIGEST:
        raise ValueError(
            f"Canonical tree digest mismatch: {tree_digest} != {GIT_TREE_DIGEST}"
        )

    # Verify pinned blobs by fetching raw content (once only)
    blobs: dict[str, dict[str, str | int]] = {}
    blob_contents: dict[str, bytes] = {}
    for path, (expected_git_sha, expected_raw_sha256) in PINNED_BLOBS.items():
        # Verify git blob SHA from the tree listing
        tree_entry = next((e for e in entries if e["path"] == path), None)
        if tree_entry is None:
            raise ValueError(f"Pinned blob not found in tree: {path}")
        if tree_entry["sha"] != expected_git_sha:
            raise ValueError(
                f"Git blob SHA mismatch for {path}: "
                f"{tree_entry['sha']} != {expected_git_sha}"
            )
        raw = fetch_github_raw_file(path, client=client)
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != expected_raw_sha256:
            raise ValueError(
                f"Raw SHA-256 mismatch for {path}: "
                f"{actual_sha256} != {expected_raw_sha256}"
            )
        blobs[path] = {
            "git_object": expected_git_sha,
            "raw_sha256": actual_sha256,
            "size": len(raw),
        }
        blob_contents[path] = raw

    return {
        "commit": GITHUB_PINNED_COMMIT,
        "tree_id": GIT_TREE_ID,
        "entry_count": len(entries),
        "canonical_tree_digest": tree_digest,
        "entries": entries,
        "blobs": blobs,
        "blob_contents": blob_contents,
    }


# ---------------------------------------------------------------------------
# Canonical path digest computation
# ---------------------------------------------------------------------------

def compute_canonical_path_digest(paths: list[str]) -> str:
    """Compute SHA-256 of sorted paths, each followed by ``\\n``."""
    canonical = "".join(p + "\n" for p in sorted(paths)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# ---------------------------------------------------------------------------
# HF tree pagination (hardened)
# ---------------------------------------------------------------------------

def _parse_link_next(link_header: str | None) -> str | None:
    """Parse ``Link: <url>; rel="next"`` and return url, or None."""
    if not link_header:
        return None
    for part in link_header.split(","):
        part = part.strip()
        m = re.match(r'<([^>]+)>;\s*rel="next"', part)
        if m:
            return m.group(1)
    return None


def _validate_next_url(next_url: str) -> None:
    """Reject next URLs that point to foreign hosts or wrong endpoints."""
    parsed = urlparse(next_url)
    if parsed.scheme != "https":
        raise ValueError(f"Non-HTTPS next URL: {next_url}")
    if parsed.hostname != "huggingface.co":
        raise ValueError(
            f"Foreign host in next URL: {parsed.hostname} "
            f"(expected huggingface.co)"
        )
    expected_prefix = f"/api/datasets/{HF_DATASET_ID}/tree/{HF_PINNED_REVISION}"
    if not parsed.path.startswith(expected_prefix):
        raise ValueError(
            f"Wrong endpoint/revision in next URL path: {parsed.path}"
        )


def fetch_hf_tree_pages(
    client: httpx.Client | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch ALL pages of the HF recursive tree at the pinned revision.

    Returns (entries, page_count).  Hardened pagination:
    - follows only parsed Link rel="next" URLs on huggingface.co
    - rejects foreign or wrong-revision next URLs
    - rejects page 113 (max 112 pages expected)
    - rejects short interior pages (non-final pages must have 1000 entries)
    - rejects cursor cycles
    - rejects unknown entry types
    """
    base_url = (
        f"https://huggingface.co/api/datasets/{HF_DATASET_ID}"
        f"/tree/{HF_PINNED_REVISION}"
    )
    params = {"recursive": "true", "expand": "false", "limit": str(HF_TREE_PAGE_SIZE)}
    url: str | None = f"{base_url}?{urlencode(params)}"

    all_entries: list[dict[str, Any]] = []
    page_count = 0
    seen_urls: set[str] = set()

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=_TIMEOUT, follow_redirects=True)

    try:
        while url is not None:
            # Reject cursor cycles
            if url in seen_urls:
                raise RuntimeError(f"Cursor cycle detected at page {page_count + 1}")
            seen_urls.add(url)

            # Reject page 113+
            if page_count >= HF_TREE_TOTAL_PAGES:
                raise RuntimeError(
                    f"Pagination exceeded expected {HF_TREE_TOTAL_PAGES} pages "
                    f"(attempted page {page_count + 1})"
                )

            resp = client.get(url)
            resp.raise_for_status()
            page_entries = resp.json()

            if not isinstance(page_entries, list):
                raise ValueError(f"Page {page_count + 1} returned non-list")

            # Validate entry types
            for entry in page_entries:
                etype = entry.get("type")
                if etype not in ("file", "directory"):
                    raise ValueError(
                        f"Unknown entry type {etype!r} for {entry.get('path')}"
                    )

            all_entries.extend(page_entries)
            page_count += 1

            next_url = _parse_link_next(resp.headers.get("link"))

            # Check for short interior page
            if next_url is not None and len(page_entries) < HF_TREE_PAGE_SIZE:
                raise RuntimeError(
                    f"Short interior page {page_count}: "
                    f"{len(page_entries)} < {HF_TREE_PAGE_SIZE}"
                )

            if next_url is not None:
                _validate_next_url(next_url)

            url = next_url
    finally:
        if own_client:
            client.close()

    return all_entries, page_count


def fetch_hf_path_inventory(
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch and classify the complete HF tree inventory.

    Verifies exact counts (111,326 entries, 97,209 files, 14,117 dirs,
    112 pages) and all three approved canonical path digests.
    Rejects duplicate or conflicting paths.
    """
    entries, page_count = fetch_hf_tree_pages(client)

    if page_count != HF_TREE_TOTAL_PAGES:
        raise ValueError(
            f"Expected {HF_TREE_TOTAL_PAGES} pages, got {page_count}"
        )

    all_paths: list[str] = []
    file_paths: list[str] = []
    dir_paths: list[str] = []

    seen: dict[str, str] = {}  # path -> type for conflict detection
    for e in entries:
        path = e["path"]
        etype = e["type"]
        if path in seen:
            if seen[path] != etype:
                raise ValueError(
                    f"Conflicting entry types for {path}: "
                    f"{seen[path]} vs {etype}"
                )
            raise ValueError(f"Duplicate path in inventory: {path}")
        seen[path] = etype
        all_paths.append(path)
        if etype == "directory":
            dir_paths.append(path)
        else:
            file_paths.append(path)

    if len(all_paths) != HF_TREE_TOTAL_ENTRIES:
        raise ValueError(
            f"Expected {HF_TREE_TOTAL_ENTRIES} unique entries, "
            f"got {len(all_paths)}"
        )
    if len(file_paths) != HF_TREE_FILE_COUNT:
        raise ValueError(
            f"Expected {HF_TREE_FILE_COUNT} files, got {len(file_paths)}"
        )
    if len(dir_paths) != HF_TREE_DIR_COUNT:
        raise ValueError(
            f"Expected {HF_TREE_DIR_COUNT} dirs, got {len(dir_paths)}"
        )

    # Verify canonical digests
    all_digest = compute_canonical_path_digest(all_paths)
    file_digest = compute_canonical_path_digest(file_paths)
    dir_digest = compute_canonical_path_digest(dir_paths)

    from posttrainbench_repro.constants import (
        CANONICAL_ALL_ENTRIES_SHA256,
        CANONICAL_DIRS_SHA256,
        CANONICAL_FILES_SHA256,
    )

    if all_digest != CANONICAL_ALL_ENTRIES_SHA256:
        raise ValueError(f"All-entries digest mismatch: {all_digest}")
    if file_digest != CANONICAL_FILES_SHA256:
        raise ValueError(f"Files digest mismatch: {file_digest}")
    if dir_digest != CANONICAL_DIRS_SHA256:
        raise ValueError(f"Dirs digest mismatch: {dir_digest}")

    return {
        "revision": HF_PINNED_REVISION,
        "page_count": page_count,
        "total_entries": len(all_paths),
        "file_count": len(file_paths),
        "dir_count": len(dir_paths),
        "all_paths": all_paths,
        "file_paths": file_paths,
        "dir_paths": dir_paths,
        "canonical_all_digest": all_digest,
        "canonical_file_digest": file_digest,
        "canonical_dir_digest": dir_digest,
    }


# ---------------------------------------------------------------------------
# Allowlisted file fetch
# ---------------------------------------------------------------------------

def fetch_allowlisted_file(
    path: str,
    *,
    client: httpx.Client | None = None,
) -> bytes:
    """Fetch a single file from the HF dataset at the pinned revision.

    Rejects any path not in ``HF_ALLOWLISTED_FILES`` before I/O.
    """
    if path not in HF_ALLOWLISTED_FILES:
        raise ValueError(
            f"Path {path!r} is not in the approved allowlist. "
            f"Only {sorted(HF_ALLOWLISTED_FILES)} are permitted."
        )
    url = (
        f"https://huggingface.co/datasets/{HF_DATASET_ID}"
        f"/raw/{HF_PINNED_REVISION}/{path}"
    )
    if client is None:
        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as c:
            resp = c.get(url)
    else:
        resp = client.get(url, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# JSONL trace excerpt extraction
# ---------------------------------------------------------------------------

def extract_trace_excerpts(trace_bytes: bytes) -> list[dict[str, Any]]:
    """Parse the JSONL trace and derive the three approved excerpts.

    Each excerpt is identified by its 0-indexed JSONL record number and
    JSON pointer path.  The function reads the trace, locates each record,
    navigates the pointer, extracts the text, and verifies its SHA-256.

    Raises on mutated, missing, or malformed records.
    """
    lines = trace_bytes.split(b"\n")
    # Filter out empty trailing lines
    lines = [ln for ln in lines if ln.strip()]

    results: list[dict[str, Any]] = []
    for spec in TRACE_EXCERPTS:
        record_idx = spec["record"]
        pointer = spec["pointer"]
        expected_sha = spec["sha256"]

        if record_idx >= len(lines):
            raise IndexError(
                f"Trace has only {len(lines)} records, "
                f"expected record {record_idx}"
            )

        try:
            record = json.loads(lines[record_idx])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON at record {record_idx}: {exc}"
            ) from exc

        # Navigate the JSON pointer
        value = _resolve_pointer(record, pointer, record_idx)
        text = str(value)

        # Verify hash
        actual_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"Trace excerpt hash mismatch at record {record_idx} "
                f"pointer {pointer}: {actual_sha} != {expected_sha}"
            )

        results.append({
            "record": record_idx,
            "json_pointer": pointer,
            "text": text,
            "sha256": actual_sha,
        })

    return results


def _resolve_pointer(
    obj: Any, pointer: str, record_idx: int
) -> Any:
    """Resolve a JSON pointer (RFC 6901) against an object."""
    if not pointer.startswith("/"):
        raise ValueError(f"Invalid JSON pointer: {pointer}")
    parts = pointer.lstrip("/").split("/")
    current = obj
    for part in parts:
        # Unescape ~1 and ~0
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(
                    f"Key {part!r} not found in record {record_idx} "
                    f"(available keys: {list(current.keys())})"
                )
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
            except ValueError:
                raise KeyError(
                    f"Non-integer index {part!r} for list at record {record_idx}"
                )
            if idx >= len(current):
                raise IndexError(
                    f"Index {idx} out of range for list of length "
                    f"{len(current)} at record {record_idx}"
                )
            current = current[idx]
        else:
            raise ValueError(
                f"Cannot navigate pointer into {type(current).__name__} "
                f"at record {record_idx}"
            )
    return current


# ---------------------------------------------------------------------------
# Full acquisition bundle
# ---------------------------------------------------------------------------

def acquire_all(
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Run all acquisition steps and return a verified data bundle.

    This is the single entry point used by the production pipeline.
    All data is fetched once and verified before returning.
    """
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=_TIMEOUT, follow_redirects=True)

    try:
        # 1. GitHub tree + blobs (fetched once, reused downstream)
        github = fetch_github_metadata(client)

        # 2. Complete HF paginated tree inventory
        hf_inventory = fetch_hf_path_inventory(client)

        # 3. Allowlisted HF files
        from posttrainbench_repro.constants import (
            CONTAMINATION_WITNESS_PATH,
            CONTAMINATION_WITNESS_SHA256,
            CONTAMINATION_WITNESS_BYTES,
            INSTRUCTION_MODEL_JUDGMENT_PATH,
            INSTRUCTION_MODEL_JUDGMENT_SHA256,
            INSTRUCTION_MODEL_JUDGMENT_BYTES,
            INSTRUCTION_MODEL_TRACE_PATH,
            INSTRUCTION_MODEL_TRACE_SHA256,
            TIME_TAKEN_WITNESS_PATH,
            TIME_TAKEN_WITNESS_SHA256,
            TIME_TAKEN_WITNESS_BYTES,
        )

        contam = fetch_allowlisted_file(
            CONTAMINATION_WITNESS_PATH, client=client
        )
        _verify_bytes(contam, CONTAMINATION_WITNESS_SHA256,
                       "contamination witness")
        if contam != CONTAMINATION_WITNESS_BYTES:
            raise ValueError("Contamination witness bytes mismatch")

        time_taken = fetch_allowlisted_file(
            TIME_TAKEN_WITNESS_PATH, client=client
        )
        _verify_bytes(time_taken, TIME_TAKEN_WITNESS_SHA256,
                       "time_taken witness")
        if time_taken != TIME_TAKEN_WITNESS_BYTES:
            raise ValueError("Time taken witness bytes mismatch")

        judgment = fetch_allowlisted_file(
            INSTRUCTION_MODEL_JUDGMENT_PATH, client=client
        )
        _verify_bytes(judgment, INSTRUCTION_MODEL_JUDGMENT_SHA256,
                       "instruction judgment")
        if judgment != INSTRUCTION_MODEL_JUDGMENT_BYTES:
            raise ValueError("Instruction judgment bytes mismatch")

        trace = fetch_allowlisted_file(
            INSTRUCTION_MODEL_TRACE_PATH, client=client
        )
        _verify_bytes(trace, INSTRUCTION_MODEL_TRACE_SHA256,
                       "instruction trace")

        # 4. Parse and verify trace excerpts
        trace_excerpts = extract_trace_excerpts(trace)

    finally:
        if own_client:
            client.close()

    return {
        "github": github,
        "hf_inventory": hf_inventory,
        "contamination_content": contam,
        "time_taken_content": time_taken,
        "instruction_judgment_content": judgment,
        "instruction_trace_sha256": hashlib.sha256(trace).hexdigest(),
        "instruction_trace_size": len(trace),
        "trace_excerpts": trace_excerpts,
    }


def _verify_bytes(data: bytes, expected_sha256: str, label: str) -> None:
    """Verify SHA-256 of fetched bytes."""
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: {actual} != {expected_sha256}"
        )
