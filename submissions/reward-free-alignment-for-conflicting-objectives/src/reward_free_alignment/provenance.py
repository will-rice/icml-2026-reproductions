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


_EXPECTED_CLAIMS = (
    LiveClaim(1, "RACO is an offline, reward-free preference-alignment method that accepts user-specified objective weights and explicitly handles conflicting objectives (Table 1).", "e9a35e34b57a7273bf84d3d5981ab19f8ff1088adef8363f4640dcf436183944", False),
    LiveClaim(2, "The method uses CAGrad-Clip to limit correction gradients so updates better respect preferred objective trade-offs (Figure 1, Algorithm 1).", "7c0aa54e034d03f2d0905417a024af4db41338cd9a803a0b42e441945c307cf9", False),
    LiveClaim(3, "On TL;DR summarization, RACO achieves better Pareto frontiers for conciseness-quality and faithfulness-quality trade-offs than AMoPO and weighted-loss DPO baselines (Figure 2, Figure 3).", "85abbc8a21d5c4537409f6e9f2af6bffc7e4c15e2311dfa078bf816ea0cffc9e", False),
    LiveClaim(4, "On BeaverTails safety alignment, RACO improves harmlessness-helpfulness Pareto trade-offs across Qwen3 and Gemma3 setups (Figure 4).", "dac93f364ac0469302894920781b034bfcd205816fbe16537c2f8e7c10d8995d", False),
    LiveClaim(5, "Ablations show clipping and the correction-radius constant affect validation margins and Pareto frontiers (Figure 5, Figure 6).", "269d8a5053e224206036399bccb2435455565149086de6439a9046d89682772b", False),
    LiveClaim(6, "RACO directly applies conflict-averse gradient descent to objective-specific pairwise preference losses instead of relying on explicit reward models (Section 3).", "0d457572ea8a502fa8a489fef3e15da21b13cc39dd3a3730843d1cbe833059b0", True),
    LiveClaim(7, "The clipped CAGrad update is introduced to stabilize multi-objective LLM alignment while respecting user-specified objective weights (Section 3.2).", "50719d645042a500f9c4d53fbdfedf719ee56429ddb73a231912f1eaeadb1b31", True),
    LiveClaim(8, "The paper proves convergence of clipped CAGrad to Pareto-critical points that respect user-specified weights in nonconvex smooth settings (Theorem 3.1).", "5ec835ce150ff60d1e2bbd4fbdf7d1ebacf91bb2b6b2d65f72c44c3b3ed65229", True),
    LiveClaim(9, "For two objectives, the analysis shows clipping can strictly improve the convergence rate (Theorem 3.2).", "b74a0ea75967144b210934fd40fd23449d3ef985df878d5a9e14c4b04025ba4b", True),
    LiveClaim(10, "Experiments on multi-objective summarization and safety alignment across Qwen 3, Llama 3, and Gemma 3 report better Pareto trade-offs than reward-free baselines (Section 4).", "58b31f527bb5e1bccb05c0dab775a74c2f2bdcd8e92ef2c0dd578733b5fb058e", False),
)


@dataclass(frozen=True)
class VerifiedArtifact:
    artifact_id: str
    relative_path: str
    sha256: str
    git_blob: str | None
    size_bytes: int
    source_url: str
    acquisition_command: str
    license: str


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


def _git_blob_id(payload: bytes) -> str:
    """Compute the Git blob object ID for a byte payload."""
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """JSON object_pairs_hook that raises IntegrityError on duplicate keys."""
    seen: set[str] = set()
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise IntegrityError(f"Duplicate JSON key: {key!r}")
        seen.add(key)
        result[key] = value
    return result


def load_live_claims(path: Path) -> tuple[LiveClaim, ...]:
    if not path.is_file():
        raise IntegrityError(f"live_claims file missing: {path}")

    content = path.read_text(encoding="utf-8")
    data = json.loads(content, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(data, list):
        raise IntegrityError("live_claims.json must contain a list")

    claims: list[LiveClaim] = []
    seen_ordinals: set[int] = set()
    seen_hashes: set[str] = set()
    seen_texts: set[str] = set()

    for item in data:
        if not isinstance(item, dict):
            raise IntegrityError("Claim item must be a dictionary")

        # Reject extra keys
        allowed_keys = {"ordinal", "text", "sha256", "targeted"}
        extra = set(item.keys()) - allowed_keys
        if extra:
            raise IntegrityError(f"Extra keys in claim: {extra}")

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
        if text in seen_texts:
            raise IntegrityError(f"Duplicate claim text: {text!r}")

        computed_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if computed_hash != sha256_hex:
            raise IntegrityError(f"Hash mismatch for claim {ordinal}: expected {sha256_hex}, got {computed_hash}")

        seen_ordinals.add(ordinal)
        seen_hashes.add(sha256_hex)
        seen_texts.add(text)
        claims.append(LiveClaim(ordinal=ordinal, text=text, sha256=sha256_hex, targeted=targeted))

    if len(claims) != len(_EXPECTED_CLAIMS):
        raise IntegrityError(f"Expected {len(_EXPECTED_CLAIMS)} live claims, got {len(claims)}")

    for idx, (claim, expected) in enumerate(zip(claims, _EXPECTED_CLAIMS)):
        if claim != expected:
            raise IntegrityError(
                f"Loaded claim at index {idx} does not match admitted claim identity"
            )

    return tuple(claims)


_EXPECTED_ATTEMPT_ID = "97e213a5-7ca3-4a1b-a500-1ec52d94d87a"
_EXPECTED_PAPER_ID = "vSzRJyg6k0"
_EXPECTED_SNAPSHOT_ID = "09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199"
_EXPECTED_UPSTREAM_REV = "arxiv:2602.02495v3+github:PeterLauLukChen/RACO@84a943c34f38520c7e0c9dd3066517c111b3c8fa"
_EXPECTED_COMMIT_HASH = "84a943c34f38520c7e0c9dd3066517c111b3c8fa"

_MANIFEST_REQUIRED_KEYS = {
    "attempt_id", "paper_id", "snapshot_id", "upstream_revision", "artifacts"
}
_MANIFEST_ALLOWED_KEYS = _MANIFEST_REQUIRED_KEYS

_ARTIFACT_REQUIRED_KEYS = {
    "artifact_id", "relative_path", "sha256", "git_blob", "size_bytes",
    "source_url", "acquisition_command", "license"
}
_ARTIFACT_ALLOWED_KEYS = _ARTIFACT_REQUIRED_KEYS


def load_manifest(project_root: Path | None = None, path: Path | None = None) -> dict[str, Any]:
    if path is None:
        if project_root is None:
            raise ValueError("Either project_root or path must be provided")
        path = project_root / "evidence/inputs/upstream_manifest.json"

    if not path.is_file():
        raise IntegrityError(f"Manifest file missing: {path}")

    data = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(data, dict):
        raise IntegrityError("Manifest must be a JSON object")

    # Reject missing required keys
    missing = _MANIFEST_REQUIRED_KEYS - data.keys()
    if missing:
        raise IntegrityError(f"Manifest missing required keys: {missing}")

    # Reject extra keys (fail-closed)
    extra = data.keys() - _MANIFEST_ALLOWED_KEYS
    if extra:
        raise IntegrityError(f"Manifest has extra keys: {extra}")

    # Verify identity bindings
    if data["attempt_id"] != _EXPECTED_ATTEMPT_ID:
        raise IntegrityError(f"Manifest attempt_id mismatch: {data['attempt_id']}")
    if data["paper_id"] != _EXPECTED_PAPER_ID:
        raise IntegrityError(f"Manifest paper_id mismatch: {data['paper_id']}")
    if data["snapshot_id"] != _EXPECTED_SNAPSHOT_ID:
        raise IntegrityError(f"Manifest snapshot_id mismatch: {data['snapshot_id']}")
    if data["upstream_revision"] != _EXPECTED_UPSTREAM_REV:
        raise IntegrityError(f"Manifest upstream_revision mismatch: {data['upstream_revision']}")

    return data


def load_verified_artifacts(project_root: Path) -> tuple[VerifiedArtifact, ...]:
    manifest = load_manifest(project_root)
    raw_artifacts = manifest.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise IntegrityError("artifacts must be a list")

    if not raw_artifacts:
        raise IntegrityError("Empty artifact list: manifest must contain pinned upstream artifacts")

    # Pass 1: duplicate ID and path check across all entries
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for item in raw_artifacts:
        if isinstance(item, dict):
            art_id = item.get("artifact_id")
            rel_p = item.get("relative_path")
            if isinstance(art_id, str) and art_id in seen_ids:
                raise IntegrityError(f"Duplicate artifact_id: {art_id}")
            if isinstance(rel_p, str) and rel_p in seen_paths:
                raise IntegrityError(f"Duplicate artifact path: {rel_p}")
            if isinstance(art_id, str):
                seen_ids.add(art_id)
            if isinstance(rel_p, str):
                seen_paths.add(rel_p)

    verified: list[VerifiedArtifact] = []

    for item in raw_artifacts:
        if not isinstance(item, dict):
            raise IntegrityError("Artifact entry must be a dictionary")

        if not item:
            raise IntegrityError("Empty artifact entry")

        # Reject missing required artifact keys or extra keys
        missing_art = _ARTIFACT_REQUIRED_KEYS - item.keys()
        if missing_art:
            raise IntegrityError(f"Artifact entry missing required keys: {missing_art}")
        extra_art = item.keys() - _ARTIFACT_ALLOWED_KEYS
        if extra_art:
            raise IntegrityError(f"Artifact entry has extra keys: {extra_art}")

        artifact_id = item["artifact_id"]
        rel_str = item["relative_path"]
        sha256 = item["sha256"]
        git_blob = item["git_blob"]
        size_bytes = item["size_bytes"]
        source_url = item["source_url"]
        acquisition_command = item["acquisition_command"]
        license_str = item["license"]

        if not isinstance(artifact_id, str) or not artifact_id:
            raise IntegrityError("artifact_id must be a non-empty string")
        if not isinstance(rel_str, str):
            raise IntegrityError("relative_path must be a string")
        if not isinstance(sha256, str):
            raise IntegrityError("sha256 must be a string")
        if not isinstance(git_blob, str) or not git_blob:
            raise IntegrityError("git_blob must be a non-empty string")
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            raise IntegrityError("size_bytes must be a positive integer (empty files forbidden)")

        safe_rel = _safe_relative_path(rel_str)
        abs_path = project_root / safe_rel
        if abs_path.is_symlink():
            raise IntegrityError(f"Symlink escapes forbidden: {rel_str}")
        if not abs_path.is_file():
            raise IntegrityError(f"Artifact file missing: {abs_path}")

        payload = abs_path.read_bytes()
        if len(payload) == 0:
            raise IntegrityError(f"Artifact file is empty: {abs_path}")

        verify_bytes(payload, sha256, size_bytes)

        computed_blob = _git_blob_id(payload)
        if computed_blob != git_blob:
            raise IntegrityError(
                f"Git blob mismatch for {rel_str}: "
                f"expected {git_blob}, got {computed_blob}"
            )

        raw_prefix = f"https://raw.githubusercontent.com/PeterLauLukChen/RACO/{_EXPECTED_COMMIT_HASH}/"
        if not isinstance(source_url, str) or not source_url.startswith(raw_prefix) or "/blob/" in source_url:
            raise IntegrityError(f"source_url must be exact raw URL starting with {raw_prefix}: got {source_url}")

        if not isinstance(acquisition_command, str) or "git checkout" not in acquisition_command or _EXPECTED_COMMIT_HASH not in acquisition_command:
            raise IntegrityError(f"acquisition_command must contain git checkout {_EXPECTED_COMMIT_HASH}: got {acquisition_command}")

        if license_str != "Apache-2.0":
            raise IntegrityError(f"license must be Apache-2.0: got {license_str}")

        verified.append(
            VerifiedArtifact(
                artifact_id=artifact_id,
                relative_path=rel_str,
                sha256=sha256,
                git_blob=git_blob,
                size_bytes=size_bytes,
                source_url=source_url,
                acquisition_command=acquisition_command,
                license=license_str,
            )
        )
    return tuple(verified)
