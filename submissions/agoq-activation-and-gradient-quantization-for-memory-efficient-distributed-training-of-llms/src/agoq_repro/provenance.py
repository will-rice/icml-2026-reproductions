"""Fail-closed verification of AGoQ paper transcription and source inputs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY = "https://github.com/Eutenacity/AGoQ.git"
COMMIT = "006fa0f6318228d1fcd6727f0578c0e548e5cbff"
COMPONENTS = {
    "qkv",
    "attention",
    "linear_1",
    "rmsnorm",
    "ffn_1",
    "activation",
    "ffn_2",
}


class IntegrityError(ValueError):
    """Pinned input bytes or metadata failed verification."""


@dataclass(frozen=True)
class VerifiedFile:
    path: str
    git_blob: str
    sha256: str
    size_bytes: int


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read {path.name}: {exc}") from exc
    if type(value) is not dict:
        raise IntegrityError(f"{path.name} must contain a JSON object")
    return value


def canonical_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_id(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _safe_relative_path(value: object) -> str:
    if type(value) is not str or not value:
        raise IntegrityError("source path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise IntegrityError(f"unsafe source path: {value}")
    return value


def load_verified_sources(project_root: Path) -> tuple[VerifiedFile, ...]:
    manifest = _load_json(project_root / "evidence/inputs/upstream_manifest.json")
    if set(manifest) != {
        "schema_version",
        "repository",
        "commit",
        "license_file",
        "files",
    }:
        raise IntegrityError("upstream manifest schema")
    if manifest["schema_version"] != 1:
        raise IntegrityError("upstream manifest schema_version")
    if manifest["repository"] != REPOSITORY:
        raise IntegrityError("upstream repository")
    if manifest["commit"] != COMMIT or not re.fullmatch(
        r"[0-9a-f]{40}", str(manifest["commit"])
    ):
        raise IntegrityError("upstream commit")
    if manifest["license_file"] != "LICENSE":
        raise IntegrityError("upstream license_file")
    entries = manifest["files"]
    if type(entries) is not list or len(entries) != 10:
        raise IntegrityError("upstream files")

    verified: list[VerifiedFile] = []
    seen: set[str] = set()
    source_root = project_root / "evidence/inputs/upstream"
    for entry in entries:
        if type(entry) is not dict or set(entry) != {
            "path",
            "size_bytes",
            "git_blob",
            "sha256",
        }:
            raise IntegrityError("upstream file entry schema")
        relative = _safe_relative_path(entry["path"])
        if relative in seen:
            raise IntegrityError(f"duplicate source path: {relative}")
        seen.add(relative)
        size = entry["size_bytes"]
        blob = entry["git_blob"]
        sha256 = entry["sha256"]
        if type(size) is not int or isinstance(size, bool) or size <= 0:
            raise IntegrityError(f"invalid size for {relative}")
        if type(blob) is not str or not re.fullmatch(r"[0-9a-f]{40}", blob):
            raise IntegrityError(f"invalid Git blob for {relative}")
        if type(sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise IntegrityError(f"invalid SHA-256 for {relative}")
        path = source_root.joinpath(*PurePosixPath(relative).parts)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise IntegrityError(f"missing source file: {relative}") from exc
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != sha256:
            raise IntegrityError(
                f"SHA-256 mismatch for {relative}: {actual_sha} != {sha256}"
            )
        if len(payload) != size:
            raise IntegrityError(f"size mismatch for {relative}")
        if _git_blob_id(payload) != blob:
            raise IntegrityError(f"Git blob mismatch for {relative}")
        verified.append(VerifiedFile(relative, blob, sha256, size))
    return tuple(verified)


def load_verified_transcription(project_root: Path) -> dict[str, object]:
    value = _load_json(project_root / "evidence/inputs/paper_transcription.json")
    if set(value) != {
        "schema_version",
        "paper",
        "table_1_units",
        "pipeline",
        "training_tables",
    }:
        raise IntegrityError("paper transcription schema")
    if value["schema_version"] != 1:
        raise IntegrityError("paper transcription schema_version")
    paper = value["paper"]
    if type(paper) is not dict or paper != {
        "arxiv_id": "2605.00539v2",
        "pdf_sha256": "6a5095edf64e730a824fc076a0cbf3d97922b370dc827f173e872e17eb95e0d7",
        "pdf_size_bytes": 3196252,
        "license": "CC BY 4.0",
    }:
        raise IntegrityError("paper identity")
    table = value["table_1_units"]
    if type(table) is not dict or set(table) != {"bf16", "coat", "agoq"}:
        raise IntegrityError("table_1_units methods")
    for method, components in table.items():
        if type(components) is not dict or set(components) != COMPONENTS:
            raise IntegrityError(f"table_1_units components for {method}")
        if any(type(item) is not str for item in components.values()):
            raise IntegrityError(f"table_1_units values for {method}")
    pipeline = value["pipeline"]
    if type(pipeline) is not dict or set(pipeline) != {
        "printed_equation",
        "stored_batches_device_order_n4",
        "minimum_bits",
        "reported_bits_device_order_n4",
    }:
        raise IntegrityError("pipeline transcription")
    training = value["training_tables"]
    if type(training) is not dict or training != {
        "table_2_required_hardware": "64 GPUs",
        "table_3_required_hardware": "16 NVIDIA Blackwell GPUs",
    }:
        raise IntegrityError("training table transcription")
    return value
