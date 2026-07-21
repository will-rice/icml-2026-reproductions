"""
Tests for full trajectory state building.

Verifies that applying a complete sequence of operations produces
a non-empty, well-structured state. Spot-checks high-level invariants
(merged cells exist, at least one cell has a value, etc.) rather than
trajectory-specific cell values so the suite stays valid as the
shipped benchmark evolves.
"""

import pytest

from next_action_pred_eval.core.symbolic import symbolic_to_operations
from next_action_pred_eval.core.state import StateBuilder


def _first_sheet(state):
    sheets = list(state["worksheets"].keys())
    assert sheets, "state has no worksheets"
    return sheets[0]


class TestFullTrajectoryState:
    """Build state from a complete trajectory and spot-check."""

    def test_all_operations_parse(self, trajectory_0000afae):
        """All operations should parse without error."""
        ops = symbolic_to_operations(trajectory_0000afae)
        assert len(ops) > 0

    def test_state_building_completes(self, trajectory_0000afae):
        """Applying all operations should produce a non-empty state."""
        ops = symbolic_to_operations(trajectory_0000afae)
        builder = StateBuilder()
        builder.apply_operations(ops)
        state = builder.state

        sheet = _first_sheet(state)
        cells = state["worksheets"][sheet]["cells"]
        assert len(cells) > 0

    def test_merged_cells_present(self, trajectory_0000afae):
        """Trajectory should produce at least one merged region."""
        ops = symbolic_to_operations(trajectory_0000afae)
        builder = StateBuilder()
        builder.apply_operations(ops)
        state = builder.state

        sheet = _first_sheet(state)
        merged = state["worksheets"][sheet]["worksheetProperties"].get("merged_cells", [])
        assert len(merged) >= 1

    def test_has_string_value(self, trajectory_0000afae):
        """At least one cell should hold a string value."""
        ops = symbolic_to_operations(trajectory_0000afae)
        builder = StateBuilder()
        builder.apply_operations(ops)
        state = builder.state

        sheet = _first_sheet(state)
        cells = state["worksheets"][sheet]["cells"]
        string_cells = [c for c in cells.values() if isinstance(c.get("value"), str)]
        assert len(string_cells) > 0

    def test_font_format_applied(self, trajectory_0000afae):
        """At least one cell should have a font format applied."""
        ops = symbolic_to_operations(trajectory_0000afae)
        builder = StateBuilder()
        builder.apply_operations(ops)
        state = builder.state

        sheet = _first_sheet(state)
        cells = state["worksheets"][sheet]["cells"]
        font_cells = [
            c for c in cells.values()
            if c.get("Format", {}).get("font")
        ]
        assert len(font_cells) > 0

    def test_border_applied(self, trajectory_0000afae):
        """At least one cell should have a border applied."""
        ops = symbolic_to_operations(trajectory_0000afae)
        builder = StateBuilder()
        builder.apply_operations(ops)
        state = builder.state

        sheet = _first_sheet(state)
        cells = state["worksheets"][sheet]["cells"]
        border_cells = [
            c for c in cells.values()
            if c.get("Format", {}).get("borders")
        ]
        assert len(border_cells) > 0

    def test_number_format_applied(self, trajectory_0000afae):
        """At least one cell should have a non-default number format."""
        ops = symbolic_to_operations(trajectory_0000afae)
        builder = StateBuilder()
        builder.apply_operations(ops)
        state = builder.state

        sheet = _first_sheet(state)
        cells = state["worksheets"][sheet]["cells"]
        fmt_cells = [
            c for c in cells.values()
            if c.get("number_format") and c.get("number_format") != "General"
        ]
        assert len(fmt_cells) > 0
