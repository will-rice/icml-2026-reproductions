from pathlib import Path

import pytest

import graph_pruning_repro.algorithm1 as algorithm1_module
from graph_pruning_repro.algorithm1 import audit_literal_algorithm1


PROJECT_ROOT = Path(__file__).parents[1]
TRANSCRIPTION = PROJECT_ROOT / "paper_transcriptions" / "algorithm1.txt"


def _audit(source: Path | bytes) -> dict[str, object]:
    return audit_literal_algorithm1(source, project_root=PROJECT_ROOT)


def test_literal_algorithm1_authenticates_raw_path_and_stops() -> None:
    result = _audit(TRANSCRIPTION)

    assert result["greedy_path"] == "paper_algorithm1_literal"
    assert result["status"] == "undefined_read"
    assert result["line"] == 8
    assert result["symbol"] == "x*"
    assert result["selected"] is None
    assert result["repairs"] == []
    assert result["provenance_binding"] == {
        "record_id": "algorithm-1",
        "source_excerpt_path": "paper_transcriptions/algorithm1.txt",
        "source_excerpt_byte_count": 926,
        "source_excerpt_sha256": (
            "d76e4ad27e1db3256341079e159115938"
            "51962da2e04bd8b5913cb774ee79249"
        ),
        "reviewed_by": [
            "codex-graph-pruning-design-author-v2",
            "codex-graph-pruning-design-reviewer-v2",
        ],
    }
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


def test_literal_algorithm1_accepts_exact_authenticated_raw_bytes() -> None:
    result = _audit(TRANSCRIPTION.read_bytes())

    assert result["status"] == "undefined_read"
    assert result["line"] == 8
    assert result["transcription_byte_count"] == 926


def test_literal_audit_records_later_ambiguities_without_executing_them() -> None:
    result = _audit(TRANSCRIPTION)

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


def _encoded_lines(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8")


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

    with pytest.raises(ValueError, match="authenticated Algorithm 1 bytes"):
        _audit(_encoded_lines(mutated))


@pytest.mark.parametrize(
    "raw_mutation",
    (
        "crlf",
        "missing-final-newline",
        "utf8-bom",
        "extra-bytes",
    ),
)
def test_raw_byte_mutations_reject_before_literal_audit(
    monkeypatch: pytest.MonkeyPatch,
    raw_mutation: str,
) -> None:
    raw = TRANSCRIPTION.read_bytes()
    if raw_mutation == "crlf":
        mutated = raw.replace(b"\n", b"\r\n")
    elif raw_mutation == "missing-final-newline":
        assert raw.endswith(b"\n")
        mutated = raw[:-1]
    elif raw_mutation == "utf8-bom":
        mutated = b"\xef\xbb\xbf" + raw
    else:
        mutated = raw + b"\x00"
    audits = 0

    def forbidden_numbered_lines(*_args: object, **_kwargs: object) -> object:
        nonlocal audits
        audits += 1
        raise AssertionError("literal audit started before authentication")

    monkeypatch.setattr(
        algorithm1_module,
        "_numbered_lines",
        forbidden_numbered_lines,
    )

    with pytest.raises(ValueError, match="authenticated Algorithm 1 bytes"):
        _audit(mutated)
    assert audits == 0


@pytest.mark.parametrize("substitution", ("regular-copy", "symlink"))
def test_path_substitution_rejects_before_literal_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    substitution: str,
) -> None:
    substitute = tmp_path / "algorithm1.txt"
    if substitution == "regular-copy":
        substitute.write_bytes(TRANSCRIPTION.read_bytes())
    else:
        substitute.symlink_to(TRANSCRIPTION)
    audits = 0

    def forbidden_numbered_lines(*_args: object, **_kwargs: object) -> object:
        nonlocal audits
        audits += 1
        raise AssertionError("literal audit started before path authentication")

    monkeypatch.setattr(
        algorithm1_module,
        "_numbered_lines",
        forbidden_numbered_lines,
    )

    with pytest.raises(ValueError, match="canonical Algorithm 1 path"):
        _audit(substitute)
    assert audits == 0


def test_symlinked_project_root_path_rejects_before_literal_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_alias = tmp_path / "project-alias"
    project_alias.symlink_to(PROJECT_ROOT, target_is_directory=True)
    substitute = (
        project_alias / "paper_transcriptions" / "algorithm1.txt"
    )
    audits = 0

    def forbidden_numbered_lines(*_args: object, **_kwargs: object) -> object:
        nonlocal audits
        audits += 1
        raise AssertionError("literal audit started through symlinked root")

    monkeypatch.setattr(
        algorithm1_module,
        "_numbered_lines",
        forbidden_numbered_lines,
    )

    with pytest.raises(ValueError, match="canonical project root"):
        audit_literal_algorithm1(
            substitute,
            project_root=project_alias,
        )
    assert audits == 0


@pytest.mark.parametrize(
    "transcription",
    ("not-a-path", ["sequence", "of", "lines"], 7),
)
def test_literal_audit_rejects_invalid_transcription(
    transcription: object,
) -> None:
    with pytest.raises(TypeError, match="Path or raw bytes"):
        audit_literal_algorithm1(
            transcription,  # type: ignore[arg-type]
            project_root=PROJECT_ROOT,
        )
