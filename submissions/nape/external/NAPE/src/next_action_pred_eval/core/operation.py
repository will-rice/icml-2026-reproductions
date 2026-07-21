"""
Operation base class - Abstract base for all spreadsheet operations.
"""

from abc import abstractmethod
from typing import Any, Dict
import json

from pydantic import BaseModel, ConfigDict

from next_action_pred_eval.core.cell_range import CellRange


class Operation(BaseModel):
    """
    Base class for all Excel operations.

    All operations inherit from this class and must implement:
    - to_symbolic(): Convert to symbolic string representation
    - from_symbolic(): Create from symbolic string (classmethod)
    - apply_to_state(): Apply operation to a state dict
    - get_inverse(): Get the operation that reverses this one

    Attributes:
        cell_range: The cell range this operation affects
        value: The value/parameter for this operation
        is_inverse: Whether this is an inverse/clearing operation
    """

    cell_range: CellRange
    value: Any
    is_inverse: bool = False

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True
    )

    def __hash__(self) -> int:
        """Make Operation hashable by converting value to a hashable representation."""
        if isinstance(self.value, (str, int, float, bool, type(None))):
            hashable_value = self.value
        elif isinstance(self.value, (list, tuple)):
            try:
                hashable_value = json.dumps(self.value, sort_keys=True)
            except (TypeError, ValueError):
                hashable_value = str(self.value)
        elif isinstance(self.value, dict):
            hashable_value = json.dumps(self.value, sort_keys=True)
        else:
            hashable_value = str(self.value)

        return hash((self.__class__.__name__, self.cell_range, hashable_value, self.is_inverse))

    @abstractmethod
    def to_symbolic(self) -> str:
        """
        Convert to symbolic representation.

        Returns:
            String in format "OPERATION_NAME | Sheet!Range | value"
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement to_symbolic")

    @classmethod
    @abstractmethod
    def from_symbolic(cls, symbolic: str) -> "Operation":
        """
        Create operation from symbolic representation.

        Args:
            symbolic: String in format "OPERATION_NAME | Sheet!Range | value"

        Returns:
            Operation instance
        """
        raise NotImplementedError(f"{cls.__name__} must implement from_symbolic")

    @abstractmethod
    def apply_to_state(self, state: Dict[str, Any]) -> None:
        """
        Apply this operation to a workbook state dict (modifies in place).

        Args:
            state: Workbook state dict in the standard format:
                   {"worksheets": {"Sheet1": {"cells": {...}, "worksheetProperties": {...}}}}
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply_to_state")

    @abstractmethod
    def get_inverse(self) -> "Operation":
        """
        Get the inverse operation that resets/clears this operation's effect.

        Returns:
            An Operation instance that reverses/clears the effect of this operation
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_inverse")

    @property
    def modifies_format(self) -> bool:
        """
        Returns True if this operation modifies Format properties.
        Override in subclasses that modify font, fill, border, or alignment.
        """
        return False

    # Code generation methods - to be implemented by subclasses
    def to_officejs(self, sheet_var: str = "sheet") -> str:
        """
        Convert to Office.js code.

        Args:
            sheet_var: Variable name for the sheet object

        Returns:
            Office.js code string
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement to_officejs")

    def to_openpyxl(self, sheet_var: str = "ws") -> str:
        """
        Convert to OpenPyXL code.

        Args:
            sheet_var: Variable name for the worksheet object

        Returns:
            OpenPyXL Python code string
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement to_openpyxl")

    def to_xlwings(self, sheet_var: str = "sheet") -> str:
        """
        Convert to xlwings code.

        Args:
            sheet_var: Variable name for the sheet object

        Returns:
            xlwings Python code string
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement to_xlwings")
