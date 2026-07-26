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
    assert result["transcription_sha256"] == (
        "d76e4ad27e1db3256341079e159115938"
        "51962da2e04bd8b5913cb774ee79249"
    )
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


def _replace_numbered_line(
    lines: list[str],
    number: int,
    replacement: str | None,
) -> list[str]:
    prefix = f"{number}:"
    result = [line for line in lines if not line.startswith(prefix)]
    if replacement is not None:
        index = next(
            index
            for index, line in enumerate(lines)
            if line.startswith(prefix)
        )
        result.insert(index, replacement)
    return result


@pytest.mark.parametrize(
    "mutation",
    (
        "line8-defines-before-read",
        "line8-wrong-symbol",
        "line8-wrong-condition",
        "line10-absent",
        "line10-rewritten",
        "line11-absent",
        "line11-rewritten",
        "line14-absent",
        "line14-rewritten",
        "numbered-lines-reordered",
    ),
)
def test_literal_audit_authenticates_every_required_semantic_line(
    mutation: str,
) -> None:
    lines = TRANSCRIPTION.read_text().splitlines()
    if mutation == "line8-defines-before-read":
        mutated = _replace_numbered_line(
            lines,
            8,
            "8:     x⋆ ← first(T \\ S_{t-1}).",
        )
    elif mutation == "line8-wrong-symbol":
        mutated = _replace_numbered_line(
            lines,
            8,
            "8:     if y⋆ ∈ N(x_i) then",
        )
    elif mutation == "line8-wrong-condition":
        mutated = _replace_numbered_line(
            lines,
            8,
            "8:     if x⋆ ∉ N(x_i) then",
        )
    elif mutation.endswith("-absent"):
        number = int(mutation.removeprefix("line").split("-", 1)[0])
        mutated = _replace_numbered_line(lines, number, None)
    elif mutation == "line10-rewritten":
        mutated = _replace_numbered_line(
            lines,
            10,
            "10:      Carry I^ex(x_i | S_{t-1}) forward unchanged.",
        )
    elif mutation == "line11-rewritten":
        mutated = _replace_numbered_line(
            lines,
            11,
            "11:      I(x_i | S_{t-1}) ← α I^in(x_i).",
        )
    elif mutation == "line14-rewritten":
        mutated = _replace_numbered_line(
            lines,
            14,
            "14:  Select x⋆ ← first(T \\ S_{t-1}).",
        )
    else:
        mutated = list(lines)
        line10 = next(
            i for i, line in enumerate(mutated) if line.startswith("10:")
        )
        line11 = next(
            i for i, line in enumerate(mutated) if line.startswith("11:")
        )
        mutated[line10], mutated[line11] = mutated[line11], mutated[line10]

    with pytest.raises(ValueError, match="reviewed transcription"):
        audit_literal_algorithm1(mutated)


@pytest.mark.parametrize("transcription", ("not-a-sequence", [1, 2, 3]))
def test_literal_audit_rejects_invalid_transcription(
    transcription: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        audit_literal_algorithm1(transcription)  # type: ignore[arg-type]
