"""
Tests for evaluation components.

Verifies that state comparison and heuristic acceptance behave correctly.
"""

import copy
import pytest

from next_action_pred_eval.core.symbolic import symbolic_to_operations
from next_action_pred_eval.core.state import StateBuilder
from next_action_pred_eval.core.operations import get_cells_in_range
from next_action_pred_eval.evaluation.state_comparator import StateComparator
from next_action_pred_eval.evaluation.acceptance import (
    HEURISTIC_IDEAL_USER,
    AcceptanceHeuristic,
)


def _collect_cells(operations):
    """Collect all (sheet, cell) pairs touched by operations."""
    cells = set()
    for op in operations:
        try:
            for addr in get_cells_in_range(op.cell_range):
                cells.add((op.cell_range.sheet, addr))
        except Exception:
            pass
    return cells


def _restrict_state(state, cells):
    """Restrict state to only the given cells (replicates evaluator logic)."""
    grouped = {}
    for sheet, cell in cells:
        grouped.setdefault(sheet, set()).add(cell)
    filtered = {"worksheets": {}}
    for sheet_name, cell_set in grouped.items():
        ws = state.get("worksheets", {}).get(sheet_name)
        if not ws:
            continue
        fc = {c: ws["cells"][c] for c in cell_set if c in ws.get("cells", {})}
        if not fc:
            continue
        ns = {"cells": fc}
        if "worksheetProperties" in ws:
            ns["worksheetProperties"] = ws["worksheetProperties"]
        filtered["worksheets"][sheet_name] = ns
    return filtered


def _evaluate_like_reference(entry):
    """
    Replicate the reference evaluator's exact flow:
    1. Build initial_state from history
    2. pred_state = initial_state + pred_ops
    3. lookahead_state = initial_state + lookahead_ops (future)
    4. Restrict to cells from pred_ops ∪ lookahead_ops
    5. Compare restricted states
    """
    history = entry["history_before"]
    predicted = entry["prediction_made"]
    future = entry["future_before"]

    history_ops = symbolic_to_operations(history)
    pred_ops = symbolic_to_operations(predicted)
    lookahead_ops = symbolic_to_operations(future)

    # Step 1: build initial state from history
    initial_builder = StateBuilder()
    initial_builder.apply_operations(history_ops)
    initial_state = initial_builder.state

    # Step 2: pred_state = initial + pred_ops
    pred_builder = StateBuilder(copy.deepcopy(initial_state))
    pred_builder.apply_operations(pred_ops)

    # Step 3: lookahead_state = initial + lookahead_ops
    lookahead_builder = StateBuilder(copy.deepcopy(initial_state))
    lookahead_builder.apply_operations(lookahead_ops)

    # Step 4: restrict to cells from pred_ops ∪ lookahead_ops
    final_cells = _collect_cells(pred_ops) | _collect_cells(lookahead_ops)
    fp = _restrict_state(pred_builder.state, final_cells)
    fl = _restrict_state(lookahead_builder.state, final_cells)

    # Step 5: compare
    comparator = StateComparator(ignore_defaults=True)
    return comparator.compare(fp, fl)


class TestAcceptanceHeuristic:
    """Test heuristic acceptance decisions against known results."""

    def test_ideal_user_rejects_low_precision(self):
        """ideal_user requires precision >= 0.9 AND ops_saved >= 1."""
        from next_action_pred_eval.evaluation.metrics import create_empty_metrics
        import dataclasses

        base = create_empty_metrics()
        metrics = dataclasses.replace(
            base,
            total_ground_truth=3,
            total_predicted=3,
            final_state_tp=5,
            final_state_fp=4,
            final_state_fn=589,
            final_state_precision=0.5556,
            final_state_recall=0.0084,
            final_state_f1_score=0.0166,
            final_state_ops_saved=-1,
        )

        accepted = HEURISTIC_IDEAL_USER.evaluate(metrics)
        assert accepted is False

    def test_ideal_user_accepts_high_precision_with_savings(self):
        """ideal_user should accept precision >= 0.9 AND ops_saved >= 1."""
        from next_action_pred_eval.evaluation.metrics import create_empty_metrics
        import dataclasses

        base = create_empty_metrics()
        metrics = dataclasses.replace(
            base,
            total_ground_truth=5,
            total_predicted=5,
            exact_matches=5,
            final_state_tp=10,
            final_state_fp=1,
            final_state_precision=0.91,
            final_state_recall=1.0,
            final_state_f1_score=0.95,
            final_state_ops_saved=3,
        )

        accepted = HEURISTIC_IDEAL_USER.evaluate(metrics)
        assert accepted is True
