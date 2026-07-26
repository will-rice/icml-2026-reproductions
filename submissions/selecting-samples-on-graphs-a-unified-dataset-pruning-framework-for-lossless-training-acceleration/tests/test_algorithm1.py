from pathlib import Path

import pytest

from graph_pruning_repro.algorithm1 import audit_literal_algorithm1


TRANSCRIPTION = (
    Path(__file__).parents[1] / "paper_transcriptions" / "algorithm1.txt"
)


def test_literal_algorithm1_stops_before_first_selection() -> None:
    algorithm_lines = TRANSCRIPTION.read_text().splitlines()

    result = audit_literal_algorithm1(algorithm_lines)

    assert result["greedy_path"] == "paper_algorithm1_literal"
    assert result["status"] == "undefined_read"
    assert result["line"] == 8
    assert result["symbol"] == "x*"
    assert result["selected"] is None
    assert result["repairs"] == []
    assert result["state_snapshot"]["selected_symbol_defined"] is False
    assert result["state_snapshot"]["executed_numbered_lines"] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]


def test_literal_audit_records_later_ambiguities_without_executing_them() -> None:
    result = audit_literal_algorithm1(
        TRANSCRIPTION.read_text().splitlines(),
    )

    assert result["static_ambiguities"] == [
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
    ]
    assert result["last_executed_line"] == 7
    assert result["selected"] is None
    assert result["executable_resolution"] is None


def test_literal_audit_rejects_missing_or_mutated_line_eight() -> None:
    lines = TRANSCRIPTION.read_text().splitlines()
    without_line = [line for line in lines if not line.startswith("8:")]
    mutated = [
        "8:     if initialized_candidate in N(x_i) then"
        if line.startswith("8:")
        else line
        for line in lines
    ]

    with pytest.raises(ValueError, match="line 8"):
        audit_literal_algorithm1(without_line)
    with pytest.raises(ValueError, match="x"):
        audit_literal_algorithm1(mutated)


@pytest.mark.parametrize("transcription", ("not-a-sequence", [1, 2, 3]))
def test_literal_audit_rejects_invalid_transcription(
    transcription: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        audit_literal_algorithm1(transcription)  # type: ignore[arg-type]
