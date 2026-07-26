"""Literal, non-repairing state audit for the paper's Algorithm 1."""

from __future__ import annotations

import re
from collections.abc import Sequence

_NUMBERED_LINE = re.compile(r"^(?P<number>[1-9][0-9]*):(?P<body>.*)$")


def _numbered_lines(transcription: Sequence[str]) -> dict[int, str]:
    if isinstance(transcription, (str, bytes)) or not isinstance(
        transcription,
        Sequence,
    ):
        raise TypeError("transcription must be a sequence of lines")
    if any(type(line) is not str for line in transcription):
        raise TypeError("transcription lines must be strings")

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


def audit_literal_algorithm1(
    transcription: Sequence[str],
) -> dict[str, object]:
    """Execute literal lines only until the first undefined symbol read.

    The routine intentionally does not initialize, reorder, or carry forward
    any score that the transcription itself leaves unspecified.
    """

    numbered = _numbered_lines(transcription)
    if 8 not in numbered:
        raise ValueError("transcription is missing numbered line 8")
    line_eight = numbered[8]
    if "x⋆" not in line_eight and "x*" not in line_eight:
        raise ValueError("line 8 must contain the literal x* read")
    required_prefix = set(range(1, 8))
    if not required_prefix.issubset(numbered):
        missing = sorted(required_prefix.difference(numbered))
        raise ValueError(f"transcription is missing numbered lines {missing}")

    return {
        "greedy_path": "paper_algorithm1_literal",
        "status": "undefined_read",
        "line": 8,
        "source_line": line_eight,
        "symbol": "x*",
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
