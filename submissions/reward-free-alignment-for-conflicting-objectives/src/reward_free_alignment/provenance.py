from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


class IntegrityError(ValueError):
    """Pinned identity, artifact, or live-claim bytes failed verification."""


@dataclass(frozen=True)
class LiveClaim:
    ordinal: int
    text: str
    sha256: str
    targeted: bool


@dataclass(frozen=True)
class VerifiedArtifact:
    artifact_id: str
    relative_path: str
    sha256: str
    git_blob: str | None
    size_bytes: int


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise IntegrityError(f"unsafe relative path: {value}")
    return path


def verify_bytes(payload: bytes, expected_sha256: str, expected_size: int) -> None:
    if len(payload) != expected_size:
        raise IntegrityError("artifact size mismatch")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise IntegrityError("artifact SHA-256 mismatch")


def load_live_claims(path: Path) -> tuple[LiveClaim, ...]:
    if not path.is_file():
        raise IntegrityError(f"live_claims file missing: {path}")

    content = path.read_text(encoding="utf-8")
    data = json.loads(content)
    if not isinstance(data, list):
        raise IntegrityError("live_claims.json must contain a list")

    claims: list[LiveClaim] = []
    seen_ordinals: set[int] = set()
    seen_hashes: set[str] = set()

    for item in data:
        if not isinstance(item, dict):
            raise IntegrityError("Claim item must be a dictionary")
        ordinal = item.get("ordinal")
        text = item.get("text")
        sha256_hex = item.get("sha256")
        targeted = item.get("targeted")

        if not isinstance(ordinal, int) or not isinstance(text, str) or not isinstance(sha256_hex, str) or not isinstance(targeted, bool):
            raise IntegrityError(f"Invalid claim fields in item: {item}")

        if ordinal in seen_ordinals:
            raise IntegrityError(f"Duplicate ordinal: {ordinal}")
        if sha256_hex in seen_hashes:
            raise IntegrityError(f"Duplicate sha256: {sha256_hex}")

        computed_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if computed_hash != sha256_hex:
            raise IntegrityError(f"Hash mismatch for claim {ordinal}: expected {sha256_hex}, got {computed_hash}")

        seen_ordinals.add(ordinal)
        seen_hashes.add(sha256_hex)
        claims.append(LiveClaim(ordinal=ordinal, text=text, sha256=sha256_hex, targeted=targeted))

    return tuple(claims)


def load_manifest(project_root: Path | None = None, path: Path | None = None) -> dict[str, Any]:
    if path is None:
        if project_root is None:
            raise ValueError("Either project_root or path must be provided")
        path = project_root / "evidence/inputs/upstream_manifest.json"

    if not path.is_file():
        raise IntegrityError(f"Manifest file missing: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise IntegrityError("Manifest must be a JSON object")

    required_keys = {"attempt_id", "paper_id", "snapshot_id", "upstream_revision"}
    if not required_keys.issubset(data.keys()):
        raise IntegrityError(f"Manifest missing required keys: {required_keys - data.keys()}")

    return data


def load_verified_artifacts(project_root: Path) -> tuple[VerifiedArtifact, ...]:
    manifest = load_manifest(project_root)
    raw_artifacts = manifest.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise IntegrityError("artifacts must be a list")

    verified: list[VerifiedArtifact] = []
    for item in raw_artifacts:
        rel_str = item.get("relative_path")
        safe_rel = _safe_relative_path(rel_str)
        abs_path = project_root / safe_rel
        if not abs_path.is_file():
            raise IntegrityError(f"Artifact file missing: {abs_path}")

        payload = abs_path.read_bytes()
        verify_bytes(payload, item["sha256"], item["size_bytes"])
        verified.append(
            VerifiedArtifact(
                artifact_id=item["artifact_id"],
                relative_path=rel_str,
                sha256=item["sha256"],
                git_blob=item.get("git_blob"),
                size_bytes=item["size_bytes"],
            )
        )
    return tuple(verified)
