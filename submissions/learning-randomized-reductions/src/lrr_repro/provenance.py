"""Provenance and input integrity verification module."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


class IntegrityError(ValueError):
    """A pinned input or manifest failed verification."""


@dataclass(frozen=True)
class VerifiedInput:
    artifact_id: str
    relative_path: str | None
    sha256: str
    size_bytes: int
    git_blob: str | None


def git_blob_id(payload: bytes) -> str:
    """Compute git object SHA-1 blob ID for payload."""
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def read_manifest(project_root: Path) -> dict[str, object]:
    manifest_path = project_root / "evidence/inputs/upstream_manifest.json"
    if not manifest_path.exists():
        raise IntegrityError(f"Manifest file not found at {manifest_path}")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise IntegrityError(f"Failed to parse manifest JSON: {exc}") from exc


def validate_manifest(manifest: dict[str, object]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise IntegrityError("Manifest missing 'artifacts' list")

    seen_ids = set()
    for art in artifacts:
        if not isinstance(art, dict):
            raise IntegrityError("Invalid artifact item in manifest")
        art_id = art.get("artifact_id")
        if not art_id or art_id in seen_ids:
            raise IntegrityError(f"Duplicate or missing artifact_id: {art_id}")
        seen_ids.add(art_id)

        rel_path = art.get("relative_path")
        if rel_path is not None:
            if not isinstance(rel_path, str):
                raise IntegrityError(f"Invalid relative_path type for {art_id}")
            if Path(rel_path).is_absolute() or ".." in Path(rel_path).parts:
                raise IntegrityError(f"Must be a safe relative path: {rel_path}")


def load_verified_inputs(
    project_root: Path, cache_dir: Path, verify_files: bool = True
) -> tuple[VerifiedInput, ...]:
    manifest = read_manifest(project_root)
    validate_manifest(manifest)

    verified_list = []
    artifacts = manifest["artifacts"]
    inputs_dir = project_root / "evidence/inputs"

    for art in artifacts:
        art_id = art["artifact_id"]
        rel_path = art.get("relative_path")
        expected_sha = art["sha256"]
        expected_size = art["size_bytes"]
        expected_blob = art.get("git_blob")

        if rel_path is not None:
            target_path = inputs_dir / rel_path
        else:
            # Derived PDF name in cache dir
            if "v1" in art_id:
                target_path = cache_dir / "2412.18134v1.pdf"
            elif "v5" in art_id:
                target_path = cache_dir / "2412.18134v5.pdf"
            else:
                target_path = cache_dir / f"{art_id}.pdf"

        if verify_files:
            if not target_path.exists():
                raise IntegrityError(f"Artifact {art_id} file missing at {target_path}")
            payload = target_path.read_bytes()
            if len(payload) != expected_size:
                raise IntegrityError(
                    f"Artifact {art_id} size mismatch: expected {expected_size}, got {len(payload)}"
                )
            actual_sha = hashlib.sha256(payload).hexdigest()
            if actual_sha != expected_sha:
                raise IntegrityError(
                    f"Artifact {art_id} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
                )
            if expected_blob is not None:
                actual_blob = git_blob_id(payload)
                if actual_blob != expected_blob:
                    raise IntegrityError(
                        f"Artifact {art_id} Git blob mismatch: expected {expected_blob}, got {actual_blob}"
                    )

        verified_list.append(
            VerifiedInput(
                artifact_id=art_id,
                relative_path=rel_path,
                sha256=expected_sha,
                size_bytes=expected_size,
                git_blob=expected_blob,
            )
        )

    return tuple(verified_list)


def load_paper_context(project_root: Path) -> Mapping[str, object]:
    ctx_path = project_root / "evidence/inputs/paper_context.json"
    if not ctx_path.exists():
        raise IntegrityError(f"Paper context file missing at {ctx_path}")
    try:
        return json.loads(ctx_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise IntegrityError(f"Failed to parse paper context JSON: {exc}") from exc
