"""
State Builder Module.

Builds workbook state by applying operations sequentially to a state dictionary.
State format: nested dict ``{worksheets: {<name>: {cells: {<A1>: {...}}}}}``.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from copy import deepcopy

from next_action_pred_eval.core.operation import Operation

logger = logging.getLogger(__name__)


class StateBuilder:
    """
    Builds workbook state by applying operations sequentially.

    Uses a standard format for state representation:
    {
        "worksheets": {
            "Sheet1": {
                "cells": {
                    "A1": {"value": ..., "formula": ..., "Format": {...}},
                    ...
                },
                "worksheetProperties": {"merged_cells": [...]}
            }
        }
    }

    Each operation directly modifies the state dict using its apply_to_state() method.
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        """
        Initialize StateBuilder.

        Args:
            initial_state: Initial workbook state (optional). If None, starts with empty state.
        """
        if initial_state:
            self.state = deepcopy(initial_state)
        else:
            self.state = {"worksheets": {}}

        logger.debug("StateBuilder initialized")

    @classmethod
    def from_workbook(cls, workbook_path: Union[str, Path]) -> 'StateBuilder':
        """
        Create StateBuilder from existing workbook using openpyxl.

        Args:
            workbook_path: Path to Excel workbook

        Returns:
            StateBuilder with initial state loaded from workbook
        """
        from next_action_pred_eval.utils.workbook.sheet_to_state import workbook_to_state

        initial_state = workbook_to_state(str(workbook_path))
        logger.debug(f"Loaded initial state from {workbook_path}")
        return cls(initial_state)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'StateBuilder':
        """
        Create StateBuilder from JSON state file.

        Args:
            json_path: Path to JSON file containing state

        Returns:
            StateBuilder with initial state loaded from JSON
        """
        import json

        with open(json_path, 'r', encoding='utf-8') as f:
            initial_state = json.load(f)

        logger.debug(f"Loaded initial state from {json_path}")
        return cls(initial_state)

    def apply_operations(
        self,
        operations: List[Operation],
        raise_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Apply operations sequentially to build state.

        Args:
            operations: List of operations to apply
            raise_on_error: If True (default), raise on the first operation
                that fails to apply.  If False, log a warning and skip the
                failed operation, continuing with the rest.  Use False when
                applying LLM-predicted operations that may contain invalid
                cell ranges.

        Returns:
            Final state after applying all operations
        """
        logger.debug(f"Applying {len(operations)} operations to build state")

        for i, op in enumerate(operations):
            try:
                op.apply_to_state(self.state)
            except Exception as e:
                if raise_on_error:
                    logger.error(f"Error applying operation {i}: {op} - {e}")
                    raise
                logger.warning(f"Skipping failed operation {i}: {op} - {e}")

        logger.debug("All operations applied successfully")
        return self.get_state()

    def apply_operation(self, operation: Operation) -> Dict[str, Any]:
        """
        Apply a single operation to the current state.

        Args:
            operation: Operation to apply

        Returns:
            Current state after applying the operation
        """
        try:
            operation.apply_to_state(self.state)
        except Exception as e:
            logger.error(f"Error applying operation: {operation} - {e}")
            raise

        return self.state

    def get_state(self) -> Dict[str, Any]:
        """Get a deep copy of the current state."""
        return deepcopy(self.state)

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the current state to a new state (deep copied)."""
        self.state = deepcopy(state)

    def reset(self) -> None:
        """Reset state to empty."""
        self.state = {"worksheets": {}}

    def build_state(
        self,
        initial_state: Dict[str, Any],
        operations: List[Operation],
    ) -> Dict[str, Any]:
        """
        Build a new state by applying operations to an initial state.

        This is a stateless operation that doesn't modify self.state.

        Args:
            initial_state: Starting workbook state
            operations: List of operations to apply

        Returns:
            New state after applying all operations
        """
        state = deepcopy(initial_state)
        for op in operations:
            try:
                op.apply_to_state(state)
            except Exception as e:
                logger.warning(f"Error applying operation in build_state: {op} - {e}")
                # Continue applying remaining operations
        return state

    def get_cell(self, sheet: str, cell_addr: str) -> Optional[Dict[str, Any]]:
        """
        Get cell data from the current state.

        Args:
            sheet: Sheet name
            cell_addr: Cell address (e.g., "A1")

        Returns:
            Cell data dict or None if not found
        """
        return (
            self.state.get("worksheets", {})
            .get(sheet, {})
            .get("cells", {})
            .get(cell_addr)
        )

    def get_sheet_names(self) -> List[str]:
        """Get list of sheet names in the current state."""
        return list(self.state.get("worksheets", {}).keys())

    def to_json(self, path: Union[str, Path]) -> None:
        """
        Save current state to JSON file.

        Args:
            path: Output path for JSON file
        """
        import json

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved state to {path}")
