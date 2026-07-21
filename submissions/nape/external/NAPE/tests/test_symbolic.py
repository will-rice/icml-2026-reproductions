"""
Tests for symbolic compress/uncompress round-trip and parsing utilities.
"""

import pytest

from next_action_pred_eval.core.symbolic import (
    symbolic_to_operations,
    operations_to_symbolic,
    compress_symbolic,
    compress_symbolic_inputs,
    uncompress_symbolic_inputs,
)


class TestCompressUncompress:
    """Test compress_symbolic_inputs and uncompress_symbolic_inputs round-trip."""

    def test_compress_adjacent_values(self):
        """Adjacent VALUE ops on same row should compress to range."""
        ops = [
            "VALUE | Sheet1!A1 | 1",
            "VALUE | Sheet1!B1 | 2",
            "VALUE | Sheet1!C1 | 3",
        ]
        compressed = compress_symbolic_inputs(ops)
        assert len(compressed) == 1
        assert "A1:C1" in compressed[0]
        assert "MERGED_CONTENT" in compressed[0]

    def test_compress_adjacent_column(self):
        """Adjacent VALUE ops in same column should compress to range."""
        ops = [
            "VALUE | Sheet1!A1 | 1",
            "VALUE | Sheet1!A2 | 2",
            "VALUE | Sheet1!A3 | 3",
        ]
        compressed = compress_symbolic_inputs(ops)
        assert len(compressed) == 1
        assert "A1:A3" in compressed[0]

    def test_non_adjacent_not_compressed(self):
        """Non-adjacent VALUE ops should remain separate."""
        ops = [
            "VALUE | Sheet1!A1 | 1",
            "VALUE | Sheet1!C1 | 3",
        ]
        compressed = compress_symbolic_inputs(ops)
        assert len(compressed) == 2

    def test_non_value_ops_not_compressed(self):
        """Non-VALUE/FORMULA ops should pass through unchanged."""
        ops = [
            "FONT_BOLD | Sheet1!A1 | True",
            "FONT_BOLD | Sheet1!B1 | True",
        ]
        compressed = compress_symbolic_inputs(ops)
        assert len(compressed) == 2  # Not compressed

    def test_uncompress_range(self):
        """Uncompress should split range back to individual cells."""
        ops = ["VALUE | Sheet1!A1:A3 | MERGED_CONTENT"]
        uncompressed = uncompress_symbolic_inputs(ops)
        assert len(uncompressed) == 3
        assert "Sheet1!A1" in uncompressed[0]
        assert "Sheet1!A2" in uncompressed[1]
        assert "Sheet1!A3" in uncompressed[2]

    def test_compress_uncompress_round_trip(self):
        """Compress then uncompress should produce same number of ops."""
        ops = [
            "VALUE | Sheet1!A1 | 1",
            "VALUE | Sheet1!A2 | 2",
            "VALUE | Sheet1!A3 | 3",
        ]
        compressed = compress_symbolic_inputs(ops)
        uncompressed = uncompress_symbolic_inputs(compressed)
        assert len(uncompressed) == 3


class TestCompressSymbolic:
    """Test the compress_symbolic utility with various options."""

    def test_remove_sheet_name(self):
        ops = ["VALUE | Sheet1!A1 | 1"]
        compressed = compress_symbolic(ops, remove_sheet_name=True)
        assert "Sheet1" not in compressed[0]
        assert "A1" in compressed[0]

    def test_remove_inputs(self):
        ops = [
            "VALUE | Sheet1!A1 | 1",
            "FONT_BOLD | Sheet1!A1 | True",
            "FORMULA | Sheet1!B1 | =SUM(A1)",
        ]
        compressed = compress_symbolic(ops, remove_inputs=True)
        assert len(compressed) == 1
        assert "FONT_BOLD" in compressed[0]

    def test_remove_args(self):
        ops = ["FONT_BOLD | Sheet1!A1 | True"]
        compressed = compress_symbolic(ops, remove_args=True)
        assert compressed[0] == "FONT_BOLD | Sheet1!A1"

    def test_compress_args(self):
        ops = ["VALUE | Sheet1!A1 | This is a very long value that should be truncated"]
        compressed = compress_symbolic(ops, compress_args=True, max_len_args=10)
        assert len(compressed[0].split(" | ")[2]) <= 13  # 10 + "..."


class TestSymbolicToOperations:
    """Test the main symbolic_to_operations function."""

    def test_empty_list(self):
        ops = symbolic_to_operations([])
        assert ops == []

    def test_comments_skipped(self):
        ops = symbolic_to_operations([
            "# This is a comment",
            "VALUE | Sheet1!A1 | 1",
        ])
        assert len(ops) == 1

    def test_blank_lines_skipped(self):
        ops = symbolic_to_operations(["", "  ", "VALUE | Sheet1!A1 | 1", ""])
        assert len(ops) == 1

    def test_unknown_operation_skipped(self):
        """Unknown operation types should be skipped with warning, not raise."""
        ops = symbolic_to_operations([
            "VALUE | Sheet1!A1 | 1",
            "UNKNOWN_OP | Sheet1!A1 | something",
            "VALUE | Sheet1!A2 | 2",
        ])
        assert len(ops) == 2  # UNKNOWN_OP skipped
