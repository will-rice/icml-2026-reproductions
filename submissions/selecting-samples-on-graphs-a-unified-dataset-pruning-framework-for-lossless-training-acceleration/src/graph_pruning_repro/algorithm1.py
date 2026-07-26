"""Literal, non-repairing state audit for authenticated Algorithm 1 bytes."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from pathlib import Path

from .provenance import load_transcriptions

_NUMBERED_LINE = re.compile(r"^(?P<number>[1-9][0-9]*):(?P<body>.*)$")


def _numbered_lines(transcription: Sequence[str]) -> dict[int, str]:
    numbered: dict[int, str] = {}
    for line in transcription:
        match = _NUMBERED_LINE.match(line)
        if match is None:
            continue
        number = int(match.group("number"))
        if number in numbered:
            raise ValueError(f"duplicate numbered line {number}")
        numbered[number] = line
    return numbered


def _algorithm_record(project_root: Path) -> dict[str, object]:
    records = load_transcriptions(project_root)
    matches = [
        record for record in records if record["record_id"] == "algorithm-1"
    ]
    if len(matches) != 1:
        raise ValueError("Task 1 provenance lacks unique Algorithm 1 record")
    return matches[0]


def _authenticated_bytes(
    source: Path | bytes,
    *,
    project_root: Path,
) -> tuple[bytes, dict[str, object]]:
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be a Path")
    project_root_absolute = Path(os.path.abspath(project_root))
    try:
        project_root_resolved = project_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("project root is not the canonical project root") from exc
    if project_root_absolute != project_root_resolved:
        raise ValueError("project root is not the canonical project root")

    record = _algorithm_record(project_root_resolved)
    expected_path = Path(
        os.path.abspath(
            project_root_resolved / str(record["source_excerpt_path"])
        )
    )

    if isinstance(source, Path):
        if source.is_symlink() or source != expected_path:
            raise ValueError("source is not the canonical Algorithm 1 path")
        raw = source.read_bytes()
    elif type(source) is bytes:
        raw = source
    else:
        raise TypeError("transcription source must be a Path or raw bytes")

    expected_count = record["source_excerpt_byte_count"]
    expected_digest = record["source_excerpt_sha256"]
    if (
        len(raw) != expected_count
        or hashlib.sha256(raw).hexdigest() != expected_digest
    ):
        raise ValueError(
            "authenticated Algorithm 1 bytes do not match Task 1 provenance"
        )
    return raw, record


def audit_literal_algorithm1(
    transcription: Path | bytes,
    *,
    project_root: Path,
) -> dict[str, object]:
    """Authenticate raw input, then stop at the first undefined read.

    The original Sequence[str] interface cannot distinguish CRLF, a missing
    final newline, or substituted paths. This security-hardened interface
    authenticates exact bytes through Task 1 provenance before UTF-8 decoding
    or literal state inspection.
    """

    raw, record = _authenticated_bytes(
        transcription,
        project_root=project_root,
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("authenticated Algorithm 1 bytes are not UTF-8") from exc
    numbered = _numbered_lines(text.splitlines())
    if 8 not in numbered:
        raise ValueError(
            "authenticated reviewed transcription is missing numbered line 8"
        )
    line_eight = numbered[8]
    if "x⋆" not in line_eight and "x*" not in line_eight:
        raise ValueError(
            "authenticated reviewed line 8 must contain the literal x* read"
        )
    required_prefix = set(range(1, 8))
    if not required_prefix.issubset(numbered):
        missing = sorted(required_prefix.difference(numbered))
        raise ValueError(
            "authenticated reviewed transcription is missing numbered lines "
            f"{missing}"
        )

    return {
        "greedy_path": "paper_algorithm1_literal",
        "status": "undefined_read",
        "line": 8,
        "source_line": line_eight,
        "symbol": "x*",
        "transcription_sha256": record["source_excerpt_sha256"],
        "transcription_byte_count": len(raw),
        "provenance_binding": {
            field: record[field]
            for field in (
                "record_id",
                "source_excerpt_path",
                "source_excerpt_byte_count",
                "source_excerpt_sha256",
                "reviewed_by",
            )
        },
        "last_executed_line": 7,
        "selected": None,
        "executable_resolution": None,
        "repairs": [],
        "state_snapshot": {
            "executed_numbered_lines": list(range(1, 8)),
            "defined_symbols": [
                "T",
                "p",
                "G",
                "I^in",
                "{N_k}",
                "D",
                "S_0",
                "I^ex(x_i | S_0)",
                "t",
                "x_i",
            ],
            "selected_symbol_defined": False,
            "S_t_defined": False,
            "unindexed_S_defined": False,
            "unified_score_at_S_0_defined": False,
        },
        "static_ambiguities": [
            {
                "line": 5,
                "kind": "missing_unified_score_initialization",
                "executed": True,
            },
            {
                "line": 8,
                "kind": "candidate_score_carry_forward_unspecified",
                "executed": False,
            },
            {
                "line": 10,
                "kind": "S_t_read_before_construction",
                "executed": False,
            },
            {
                "line": 11,
                "kind": "S_t_read_before_construction",
                "executed": False,
            },
            {
                "line": 14,
                "kind": "unindexed_S_is_undefined",
                "executed": False,
            },
        ],
    }
