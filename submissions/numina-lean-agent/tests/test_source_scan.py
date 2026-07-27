from pathlib import Path

from numina_lean import scan_lean_sources


def test_source_scan_counts_only_live_sorry_tokens(tmp_path: Path) -> None:
    (tmp_path / "Proof.lean").write_text(
        """
theorem admitted : True := by
  sorry

-- sorry in a line comment
/- sorry in a block comment
   /- sorry in a nested block comment -/
-/
def message := "sorry in a string"
def sorryAxName := 1
"""
    )
    (tmp_path / "Complete.lean").write_text(
        "theorem complete : True := by\n  trivial\n"
    )

    result = scan_lean_sources(
        tmp_path,
        [Path("Proof.lean"), Path("Complete.lean")],
    )

    assert result == {
        "file_count": 2,
        "files_with_sorry": {"Proof.lean": 1},
        "method": "nested-comment/string-aware sorry token scan",
        "sorry_count": 1,
    }
