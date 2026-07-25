"""Pinned paper identity and authenticated transcription loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

PAPER = {
    "challenge_id": "a3GdvuPItd",
    "revision": "arxiv:2606.12913v2",
    "source_url": "https://arxiv.org/pdf/2606.12913v2",
    "pdf_byte_count": 683737,
    "pdf_sha256": (
        "26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090"
    ),
    "license": "CC BY-NC-SA 4.0",
}

TARGET_CLAIMS = (
    "The paper casts dataset pruning as a graph problem with node weights "
    "for intrinsic importance and edge weights for extrinsic "
    "diversity/interaction, yielding a Maximum Weight Clique formulation "
    "(Section 3.3).",
    "Under mild conditions, the unified objective becomes submodular and "
    "admits a greedy approximation guarantee (Section 3.6; Appendix F).",
)

PDF_ACQUISITION_COMMAND = (
    "curl --fail --location --proto '=https' --tlsv1.2 "
    "--output /tmp/2606.12913v2.pdf "
    "https://export.arxiv.org/pdf/2606.12913v2 && "
    'test "$(wc -c < /tmp/2606.12913v2.pdf)" -eq 683737 && '
    "printf '%s  %s\\n' "
    "26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090 "
    "/tmp/2606.12913v2.pdf | sha256sum --check --strict"
)

TRANSCRIPTION_SET_SHA256 = (
    "817d3bb2e9f2b1136961d5c846b0c1880ff5221ee0d8b37b491ed50ce257680c"
)

_MANIFEST_KEYS = {
    "record_id",
    "equation",
    "pdf_page",
    "section",
    "normalized_expression",
    "source_excerpt_path",
    "source_excerpt_byte_count",
    "source_excerpt_sha256",
    "reviewed_by",
}
_DESIGN_AUTHOR = "codex-graph-pruning-design-author-v2"


def verify_pdf(path: Path) -> None:
    """Reject any bytes that are not the pinned arXiv v2 PDF."""

    data = path.read_bytes()
    if len(data) != PAPER["pdf_byte_count"]:
        raise ValueError("pinned PDF byte count mismatch")
    if hashlib.sha256(data).hexdigest() != PAPER["pdf_sha256"]:
        raise ValueError("pinned PDF SHA-256 mismatch")


def transcription_set_sha256(
    records: Sequence[Mapping[str, object]],
) -> str:
    """Hash the ordered immutable identity tuple for every transcription."""

    identity_rows = [
        [
            record["record_id"],
            record["source_excerpt_path"],
            record["source_excerpt_byte_count"],
            record["source_excerpt_sha256"],
        ]
        for record in records
    ]
    payload = json.dumps(
        identity_rows,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _resolve_excerpt_path(root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str):
        raise ValueError("transcription excerpt path must be a string")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != raw_path
        or not relative.parts
        or relative.parts[0] != "paper_transcriptions"
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("transcription excerpt path is not canonical")

    transcription_root = (root / "paper_transcriptions").resolve()
    resolved = (root / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(transcription_root):
        raise ValueError("transcription excerpt path escapes its root")
    return resolved


def load_transcriptions(root: Path) -> tuple[dict[str, object], ...]:
    """Load and authenticate the complete file-backed transcription set."""

    project_root = root.resolve()
    transcription_root = project_root / "paper_transcriptions"
    manifest_path = transcription_root / "manifest.json"
    raw_records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("transcription manifest must be a nonempty array")

    record_ids: set[str] = set()
    referenced_paths: set[Path] = set()
    records: list[dict[str, object]] = []

    for raw_record in raw_records:
        if not isinstance(raw_record, dict) or set(raw_record) != _MANIFEST_KEYS:
            raise ValueError("transcription manifest record keys mismatch")
        record = dict(raw_record)

        record_id = record["record_id"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("transcription record ID must be a nonempty string")
        if record_id in record_ids:
            raise ValueError("duplicate transcription record ID")
        record_ids.add(record_id)

        excerpt_path = _resolve_excerpt_path(
            project_root,
            record["source_excerpt_path"],
        )
        if excerpt_path in referenced_paths:
            raise ValueError("duplicate transcription excerpt path")
        referenced_paths.add(excerpt_path)

        excerpt = excerpt_path.read_bytes()
        excerpt.decode("utf-8")
        if record["source_excerpt_byte_count"] != len(excerpt):
            raise ValueError("transcription excerpt byte count mismatch")
        if record["source_excerpt_sha256"] != hashlib.sha256(excerpt).hexdigest():
            raise ValueError("transcription excerpt SHA-256 mismatch")

        reviewed_by = record["reviewed_by"]
        if (
            not isinstance(reviewed_by, list)
            or len(reviewed_by) != 2
            or reviewed_by[0] != _DESIGN_AUTHOR
            or not isinstance(reviewed_by[1], str)
            or reviewed_by[1] == reviewed_by[0]
        ):
            raise ValueError("transcription requires two distinct reviewers")

        records.append(record)

    actual_excerpt_paths = {
        path.resolve()
        for path in transcription_root.rglob("*.txt")
        if path.is_file()
    }
    if referenced_paths != actual_excerpt_paths:
        raise ValueError("transcription referenced-file set mismatch")

    result = tuple(records)
    if transcription_set_sha256(result) != TRANSCRIPTION_SET_SHA256:
        raise ValueError("transcription aggregate SHA-256 mismatch")
    return result
