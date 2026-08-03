"""Standard-library validation for independently reviewed tensor outputs."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


_APPROVAL_KEYS = {
    "format",
    "status",
    "receipt",
    "receipt_sha256",
    "converter",
    "reviewer",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "model_revision",
    "schema_sha256",
    "output_sha256",
    "output_bytes",
    "approval_sha256",
}


def validate_approval_record(
    approval: Mapping[str, object],
    *,
    receipt: Mapping[str, object],
    output_path: Path,
    expected_schema_sha256: str,
) -> dict[str, object]:
    """Validate independent review and every content-addressed identity."""
    if set(approval) != _APPROVAL_KEYS:
        raise ValueError("approval schema")
    if approval["format"] != "timerewarder-conversion-approval-v1":
        raise ValueError("approval schema")
    signature = approval["approval_sha256"]
    unsigned = {
        key: value for key, value in approval.items() if key != "approval_sha256"
    }
    if not _is_sha256(signature) or signature != _canonical_sha256(unsigned):
        raise ValueError("approval hash")
    if approval["status"] != "approved":
        raise ValueError("approval status")

    converter = approval["converter"]
    reviewer = approval["reviewer"]
    if (
        not isinstance(converter, str)
        or not converter
        or not isinstance(reviewer, str)
        or not reviewer
        or reviewer == converter
    ):
        raise ValueError("approval reviewer")

    embedded_receipt = approval["receipt"]
    if not isinstance(embedded_receipt, dict) or embedded_receipt != receipt:
        raise ValueError("approval receipt")
    receipt_sha256 = _canonical_sha256(receipt)
    if approval["receipt_sha256"] != receipt_sha256:
        raise ValueError("approval receipt hash")
    if receipt.get("approval_status") != "pending_independent_reviewer":
        raise ValueError("approval receipt status")
    if receipt.get("status") != "success" or receipt.get("converter") != converter:
        raise ValueError("approval receipt identity")

    identity_fields = (
        "checkpoint_sha256",
        "checkpoint_bytes",
        "model_revision",
        "schema_sha256",
        "output_sha256",
        "output_bytes",
    )
    if any(approval[field] != receipt.get(field) for field in identity_fields):
        raise ValueError("approval identity")
    if approval["schema_sha256"] != expected_schema_sha256 or not _is_sha256(
        expected_schema_sha256
    ):
        raise ValueError("approval schema identity")
    if not _is_sha256(approval["checkpoint_sha256"]) or not _is_sha256(
        approval["output_sha256"]
    ):
        raise ValueError("approval identity")
    if (
        type(approval["checkpoint_bytes"]) is not int
        or approval["checkpoint_bytes"] <= 0
        or type(approval["output_bytes"]) is not int
        or approval["output_bytes"] <= 0
        or not isinstance(approval["model_revision"], str)
        or len(approval["model_revision"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in approval["model_revision"]
        )
    ):
        raise ValueError("approval identity")

    if (
        output_path.suffix != ".safetensors"
        or not output_path.is_file()
        or output_path.is_symlink()
    ):
        raise ValueError("approval safetensors output")
    if (
        output_path.stat().st_size != approval["output_bytes"]
        or _sha256_file(output_path) != approval["output_sha256"]
    ):
        raise ValueError("approval output identity")
    return dict(approval)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
