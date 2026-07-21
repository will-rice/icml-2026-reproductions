"""Tests for composable symbolic transforms and TransformedSolver."""

import pytest

from next_action_pred_eval.core.transforms import (
    build_transforms,
    SymbolicTransform,
    RelativeRangeTransform,
    RelativeFormulaTransform,
    ValueLookupTransform,
)
from next_action_pred_eval.evaluation.transformed_solver import TransformedSolver
from next_action_pred_eval.evaluation.solver import ConstantSolver, PredictionResult


# ── RelativeRangeTransform ──────────────────────────────────────────


class TestRelativeRangeTransform:
    def test_first_op_delta_from_default(self):
        t = RelativeRangeTransform()
        result = t.encode_one("FILL_COLOR | Sheet1!B2 | #FFFF00")
        # Default is (1,1,1,1), B2 is (2,2,1,1) → delta (1,1,0,0)
        assert result == "FILL_COLOR | (1,1,0,0) | #FFFF00"

    def test_sequential_encoding(self):
        t = RelativeRangeTransform()
        t.encode_one("VALUE | Sheet1!A1 | 42")
        result = t.encode_one("VALUE | Sheet1!A2 | 43")
        assert result == "VALUE | (1,0,0,0) | 43"

    def test_range_encoding(self):
        t = RelativeRangeTransform()
        t.encode_one("VALUE | Sheet1!A1 | 42")
        result = t.encode_one("FILL_COLOR | Sheet1!A1:C3 | #FF0000")
        # A1:C3 is (1,1,3,3), previous was (1,1,1,1) → delta (0,0,2,2)
        assert result == "FILL_COLOR | (0,0,2,2) | #FF0000"

    def test_decode_predictions_single(self):
        t = RelativeRangeTransform()
        t.encode_one("VALUE | Sheet1!A1 | 42")
        t.encode_one("VALUE | Sheet1!A2 | 43")
        # State is at A2 (row=2, col=1). Predict (1,0,0,0) → A3
        decoded = t.decode_predictions(["VALUE | (1,0,0,0) | 44"])
        assert decoded == ["VALUE | Sheet1!A3 | 44"]

    def test_decode_predictions_multi_op(self):
        t = RelativeRangeTransform()
        t.encode_one("VALUE | Sheet1!A1 | 42")
        # State: row=1, col=1. Two predictions in sequence:
        decoded = t.decode_predictions([
            "VALUE | (1,0,0,0) | 43",  # → A2
            "VALUE | (1,0,0,0) | 44",  # → A3
        ])
        assert decoded == [
            "VALUE | Sheet1!A2 | 43",
            "VALUE | Sheet1!A3 | 44",
        ]

    def test_decode_does_not_modify_state(self):
        t = RelativeRangeTransform()
        t.encode_one("VALUE | Sheet1!A1 | 42")
        t.decode_predictions(["VALUE | (5,5,0,0) | far_away"])
        # State should still be at A1
        result = t.encode_one("VALUE | Sheet1!A2 | 43")
        assert result == "VALUE | (1,0,0,0) | 43"

    def test_encode_sequence(self):
        t = RelativeRangeTransform()
        ops = [
            "VALUE | Sheet1!A1 | 1",
            "VALUE | Sheet1!A2 | 2",
            "VALUE | Sheet1!A3 | 3",
        ]
        encoded = t.encode_sequence(ops)
        assert encoded[0] == "VALUE | (0,0,0,0) | 1"
        assert encoded[1] == "VALUE | (1,0,0,0) | 2"
        assert encoded[2] == "VALUE | (1,0,0,0) | 3"

    def test_roundtrip(self):
        t = RelativeRangeTransform()
        ops = [
            "VALUE | Sheet1!A1 | 42",
            "FILL_COLOR | Sheet1!B2 | #FF0000",
            "FONT_BOLD | Sheet1!B2:D4 | True",
        ]
        encoded = t.encode_sequence(ops)

        # Reset and re-encode to get state, then decode from end
        t2 = RelativeRangeTransform()
        for op in ops[:2]:
            t2.encode_one(op)
        decoded = t2.decode_predictions(encoded[2:])
        assert decoded[0] == ops[2]

    def test_non_delta_passthrough(self):
        t = RelativeRangeTransform()
        t.encode_one("VALUE | A1 | 1")
        # If prediction has standard range, pass through as-is
        decoded = t.decode_predictions(["VALUE | B2 | 2"])
        assert decoded == ["VALUE | B2 | 2"]

    def test_reset(self):
        t = RelativeRangeTransform()
        t.encode_one("VALUE | Sheet1!Z99 | far")
        t.reset()
        result = t.encode_one("VALUE | Sheet1!A1 | start")
        assert result == "VALUE | (0,0,0,0) | start"


# ── ValueLookupTransform ───────────────────────────────────────────


class TestValueLookupTransform:
    def test_default_value_gets_zero_token(self):
        t = ValueLookupTransform()
        result = t.encode_one("FONT_BOLD | A1 | True")
        assert result == "FONT_BOLD | A1 | font_bold_0"

    def test_non_default_gets_indexed_token(self):
        t = ValueLookupTransform()
        result = t.encode_one("FILL_COLOR | A1 | #FFFF00")
        assert result == "FILL_COLOR | A1 | fill_color_1"

    def test_same_value_reuses_token(self):
        t = ValueLookupTransform()
        t.encode_one("FILL_COLOR | A1 | #FFFF00")
        result = t.encode_one("FILL_COLOR | B1 | #FFFF00")
        assert result == "FILL_COLOR | B1 | fill_color_1"

    def test_different_values_get_different_tokens(self):
        t = ValueLookupTransform()
        r1 = t.encode_one("FILL_COLOR | A1 | #FFFF00")
        r2 = t.encode_one("FILL_COLOR | A2 | #00FF00")
        assert "fill_color_1" in r1
        assert "fill_color_2" in r2

    def test_content_classification(self):
        t = ValueLookupTransform()
        r1 = t.encode_one("VALUE | A1 | 42")
        r2 = t.encode_one('VALUE | A2 | "hello"')
        r3 = t.encode_one("FORMULA | A3 | =SUM(A1:A2)")
        r4 = t.encode_one("INPUT | A4 | [[1,2],[3,4]]")
        assert "inp_number_1" in r1
        assert "inp_string_1" in r2
        assert "inp_formula_1" in r3
        assert "inp_list_1" in r4

    def test_decode_resolves_tokens(self):
        t = ValueLookupTransform()
        t.encode_one("FILL_COLOR | A1 | #FFFF00")
        t.encode_one("FILL_COLOR | A2 | #00FF00")
        decoded = t.decode_predictions(["FILL_COLOR | A3 | fill_color_1"])
        assert decoded == ["FILL_COLOR | A3 | #FFFF00"]

    def test_decode_resolves_default_token(self):
        t = ValueLookupTransform()
        decoded = t.decode_predictions(["FONT_BOLD | A1 | font_bold_0"])
        # Default for FONT_BOLD is True
        assert " | True" in decoded[0] or " | true" in decoded[0]

    def test_decode_unknown_token_passthrough(self):
        t = ValueLookupTransform()
        decoded = t.decode_predictions(["VALUE | A1 | unknown_token_99"])
        assert decoded == ["VALUE | A1 | unknown_token_99"]

    def test_empty_value(self):
        t = ValueLookupTransform()
        result = t.encode_one("VALUE | A1 | ")
        assert "<empty>" in result
        decoded = t.decode_predictions(["VALUE | A1 | <empty>"])
        assert decoded[0].endswith("| ")

    def test_reset_clears_vocabulary(self):
        t = ValueLookupTransform()
        t.encode_one("FILL_COLOR | A1 | #FFFF00")
        t.reset()
        # Same value should get _1 again after reset
        result = t.encode_one("FILL_COLOR | A1 | #FFFF00")
        assert "fill_color_1" in result


# ── RelativeFormulaTransform ───────────────────────────────────────


class TestRelativeFormulaTransform:
    def test_simple_formula(self):
        t = RelativeFormulaTransform()
        result = t.encode_one("FORMULA | E2 | =SUM(A2:C2)")
        # A2 relative to E2: dr=0, dc=A(1)-E(5)=-4. C2: dc=-2
        assert result == "FORMULA | E2 | =SUM(R[0]C[-4]:R[0]C[-2])"

    def test_absolute_ref_converted(self):
        t = RelativeFormulaTransform()
        result = t.encode_one("FORMULA | B2 | =$A$1 + B1")
        # All refs converted: $A$1 relative to B2: dr=-1, dc=-1.  B1: dr=-1, dc=0
        assert "R[-1]C[-1]" in result
        assert "R[-1]C[0]" in result

    def test_non_formula_passthrough(self):
        t = RelativeFormulaTransform()
        result = t.encode_one("VALUE | A1 | 42")
        assert result == "VALUE | A1 | 42"

    def test_cross_sheet_ref_relativized(self):
        """Cross-sheet refs are relativized, sheet prefix preserved."""
        t = RelativeFormulaTransform()
        result = t.encode_one("FORMULA | A1 | =Sheet2!A1 + B1")
        # Sheet2!A1 relative to A1: dr=0, dc=0 with sheet prefix
        assert "Sheet2!R[0]C[0]" in result
        assert "R[0]C[1]" in result

    def test_decode_simple(self):
        t = RelativeFormulaTransform()
        t.encode_one("VALUE | E2 | 42")  # Set position to E2
        decoded = t.decode_predictions(["FORMULA | E3 | =SUM(R[0]C[-4]:R[0]C[-2])"])
        # R[0]C[-4] at E3(row=3, col=5) → col=1 → A3
        # R[0]C[-2] at E3 → col=3 → C3
        assert decoded == ["FORMULA | E3 | =SUM(A3:C3)"]

    def test_roundtrip_formula(self):
        t = RelativeFormulaTransform()
        original = "FORMULA | E2 | =SUM(A2:C2) - D2"
        encoded = t.encode_one(original)

        t2 = RelativeFormulaTransform()
        t2.encode_one("VALUE | A1 | dummy")  # Different position to set state
        # Re-encode to position E2
        t2.reset()
        # Decode from position E2 (same as where we encoded)
        # We need to set the decoder state to E2's position
        t2._row, t2._col = 2, 5
        decoded = t2.decode_predictions([encoded])
        assert decoded[0] == original

    def test_decode_does_not_modify_state(self):
        t = RelativeFormulaTransform()
        t.encode_one("VALUE | A1 | 1")
        t.decode_predictions(["FORMULA | Z99 | =R[0]C[-1]"])
        # State should still be at A1
        result = t.encode_one("FORMULA | B1 | =A1")
        assert "R[0]C[-1]" in result

    def test_function_name_not_matched(self):
        """SUM( should not be parsed as a cell reference."""
        t = RelativeFormulaTransform()
        result = t.encode_one("FORMULA | A1 | =SUM(B1:C1)")
        assert "SUM" in result  # SUM is preserved, not converted
        assert "R[0]C[1]" in result  # B1 is converted

    def test_with_delta_range(self):
        """When preceded by RelativeRangeTransform, range is in delta form."""
        t = RelativeFormulaTransform()
        t.encode_one("VALUE | E2 | 42")  # Set position
        # If this came second and range is already delta (from RelativeRange)
        # it should use saved position
        result = t.encode_one("FORMULA | (0,0,0,0) | =SUM(A2:C2)")
        # Position is still E2 (from delta), A2→R[0]C[-4], C2→R[0]C[-2]
        assert "R[0]C[-4]" in result
        assert "R[0]C[-2]" in result


# ── build_transforms factory ───────────────────────────────────────


class TestBuildTransforms:
    def test_empty_config(self):
        assert build_transforms([]) == []

    def test_single_transform(self):
        transforms = build_transforms([{"type": "relative_range"}])
        assert len(transforms) == 1
        assert isinstance(transforms[0], RelativeRangeTransform)

    def test_canonical_ordering(self):
        # Specify in wrong order — should be reordered
        transforms = build_transforms([
            {"type": "value_lookup"},
            {"type": "relative_range"},
            {"type": "relative_formula"},
        ])
        assert isinstance(transforms[0], RelativeFormulaTransform)
        assert isinstance(transforms[1], RelativeRangeTransform)
        assert isinstance(transforms[2], ValueLookupTransform)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown transform type"):
            build_transforms([{"type": "nonexistent"}])


# ── Composition ────────────────────────────────────────────────────


class TestTransformComposition:
    def test_all_three_encode(self):
        transforms = build_transforms([
            {"type": "relative_formula"},
            {"type": "relative_range"},
            {"type": "value_lookup"},
        ])
        op = "FORMULA | E2 | =SUM(A2:C2)"
        encoded = op
        for t in transforms:
            encoded = t.encode_one(encoded)
        # After RelativeFormula: FORMULA | E2 | =SUM(R[0]C[-4]:R[0]C[-2])
        # After RelativeRange: FORMULA | (1,4,0,0) | =SUM(R[0]C[-4]:R[0]C[-2])
        # After ValueLookup: FORMULA | (1,4,0,0) | inp_formula_1
        assert "(1,4,0,0)" in encoded
        assert "inp_formula_1" in encoded

    def test_all_three_roundtrip(self):
        transforms = build_transforms([
            {"type": "relative_formula"},
            {"type": "relative_range"},
            {"type": "value_lookup"},
        ])
        ops = [
            "VALUE | Sheet1!A1 | 42",
            "VALUE | Sheet1!A2 | 43",
            "FORMULA | Sheet1!B1 | =A1 + A2",
            "FILL_COLOR | Sheet1!B1 | #FFFF00",
        ]

        # Encode all
        encoded_ops = []
        for op in ops:
            encoded = op
            for t in transforms:
                encoded = t.encode_one(encoded)
            encoded_ops.append(encoded)

        # Try round-trip: decode the last op as a prediction
        decoded = [encoded_ops[-1]]
        for t in reversed(transforms):
            decoded = t.decode_predictions(decoded)
        assert decoded[0] == ops[-1]

    def test_format_ops_roundtrip(self):
        transforms = build_transforms([
            {"type": "relative_range"},
            {"type": "value_lookup"},
        ])
        ops = [
            "FONT_BOLD | Sheet1!A1 | True",
            "FILL_COLOR | Sheet1!A1 | #FF0000",
            "FILL_COLOR | Sheet1!A2 | #FF0000",
        ]
        # Encode first two ops (context), then encode the third separately
        for op in ops[:2]:
            for t in transforms:
                t.encode_one(op)

        # Encode 3rd op to get its transformed form
        encoded_last = ops[2]
        for t in transforms:
            encoded_last = t.encode_one(encoded_last)

        # Now reset transforms and re-encode only the first two (set decoder state)
        transforms2 = build_transforms([
            {"type": "relative_range"},
            {"type": "value_lookup"},
        ])
        for op in ops[:2]:
            for t in transforms2:
                t.encode_one(op)

        # Decode the 3rd encoded op as a "prediction" from state after op 2
        decoded = [encoded_last]
        for t in reversed(transforms2):
            decoded = t.decode_predictions(decoded)
        assert decoded[0] == ops[-1]


# ── TransformedSolver ──────────────────────────────────────────────


class _MockSolver:
    """Mock solver that returns first N ops as predictions."""

    def __init__(self, n_return=1):
        self.n_return = n_return
        self.last_history = None

    def predict(self, previous_actions, workbook_state=None, context=None):
        self.last_history = list(previous_actions)
        # Return the last N actions back as "predictions"
        preds = previous_actions[-self.n_return:] if previous_actions else []
        return PredictionResult(
            predicted_operations=[],
            predicted_symbolic=list(preds),
        )

    def reset(self):
        self.last_history = None

    def get_config(self):
        return {"solver_class": "MockSolver"}


class TestTransformedSolver:
    def test_encodes_history_for_inner_solver(self):
        inner = _MockSolver(n_return=0)
        solver = TransformedSolver(
            inner=inner,
            transforms=build_transforms([{"type": "relative_range"}]),
        )
        solver.predict(["VALUE | Sheet1!A1 | 42", "VALUE | Sheet1!A2 | 43"])
        # Inner solver should have received encoded history
        assert inner.last_history is not None
        assert "(0,0,0,0)" in inner.last_history[0]
        assert "(1,0,0,0)" in inner.last_history[1]

    def test_decodes_predictions_back(self):
        inner = _MockSolver(n_return=1)
        solver = TransformedSolver(
            inner=inner,
            transforms=build_transforms([{"type": "relative_range"}]),
        )
        result = solver.predict([
            "VALUE | Sheet1!A1 | 42",
            "VALUE | Sheet1!A2 | 43",
        ])
        # The mock returns the last encoded op as prediction.
        # TransformedSolver should decode it back.
        # Last encoded is (1,0,0,0) for A2, decoded from state at A2
        # → (1,0,0,0) from A2 = A3
        assert len(result.predicted_symbolic) == 1
        assert "A3" in result.predicted_symbolic[0]

    def test_incremental_encoding(self):
        inner = _MockSolver(n_return=0)
        solver = TransformedSolver(
            inner=inner,
            transforms=build_transforms([{"type": "relative_range"}]),
        )
        # First call: 2 ops
        solver.predict(["VALUE | Sheet1!A1 | 42", "VALUE | Sheet1!A2 | 43"])
        assert len(solver._encoded_history) == 2

        # Second call: 3 ops (1 new)
        solver.predict([
            "VALUE | Sheet1!A1 | 42",
            "VALUE | Sheet1!A2 | 43",
            "VALUE | Sheet1!A3 | 44",
        ])
        assert len(solver._encoded_history) == 3

    def test_reset_on_shorter_history(self):
        inner = _MockSolver(n_return=0)
        solver = TransformedSolver(
            inner=inner,
            transforms=build_transforms([{"type": "relative_range"}]),
        )
        solver.predict(["VALUE | Sheet1!A1 | 42", "VALUE | Sheet1!A2 | 43"])
        assert len(solver._encoded_history) == 2

        # Shorter history → reset
        solver.predict(["VALUE | Sheet1!B1 | 1"])
        assert len(solver._encoded_history) == 1

    def test_get_config(self):
        inner = ConstantSolver()
        solver = TransformedSolver(
            inner=inner,
            transforms=build_transforms([
                {"type": "relative_range"},
                {"type": "value_lookup"},
            ]),
        )
        config = solver.get_config()
        assert config["solver_class"] == "TransformedSolver"
        assert len(config["transforms"]) == 2
        assert "inner_solver" in config

    def test_metadata_contains_raw_predictions(self):
        inner = _MockSolver(n_return=1)
        solver = TransformedSolver(
            inner=inner,
            transforms=build_transforms([{"type": "relative_range"}]),
        )
        result = solver.predict(["VALUE | Sheet1!A1 | 42"])
        assert "raw_predictions" in result.metadata
        assert "transforms" in result.metadata
