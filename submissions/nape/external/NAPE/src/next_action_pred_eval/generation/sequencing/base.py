"""
Sequencing Base Classes.

Core abstractions for the operation sequencing pipeline:
- SequencingContext: Mutable context passed through the pipeline
- BaseTransformer: Abstract base class for all transformers
- Constraint: Represents before-after ordering requirements
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from next_action_pred_eval.core.operation import Operation


class Constraint(BaseModel):
    """
    Represents a before-after constraint between two operations.
    Uses operation fingerprints for stable references.
    """
    before_fingerprint: str  # Operation that must come first
    after_fingerprint: str   # Operation that must come after
    reason: Optional[str] = None  # Optional explanation

    def __repr__(self):
        if self.reason:
            return (
                f"Constraint({self.before_fingerprint} → "
                f"{self.after_fingerprint}: {self.reason})"
            )
        return f"Constraint({self.before_fingerprint} → {self.after_fingerprint})"


class SequencingContext:
    """
    Mutable context passed through transformation pipeline.
    Contains operations and metadata needed for sequencing decisions.
    """

    def __init__(
        self,
        operations: List[Operation],
        region_metadata: Dict = None,
        sheet_name: str = "Sheet1",
        transformation_log: List[str] = None,
        constraints: List[Constraint] = None,
        fingerprint_map: Dict[Operation, str] = None
    ):
        self.operations = operations
        self.region_metadata = region_metadata or {}
        self.sheet_name = sheet_name
        self.transformation_log = transformation_log or []
        self.constraints = constraints or []
        self.fingerprint_map = fingerprint_map or {}

        # Extract common metadata fields
        self.regions = self.region_metadata.get("regions", [])
        self.region_dependencies = self.region_metadata.get("region_dependencies", {})
        self.pasted_ranges = self.region_metadata.get("pasted_ranges", [])

        # Paste constraint metadata (survives merging, used by ConstraintEnforcer)
        self.paste_full_target_regions = []
        self.paste_template_target_regions = []
        self.paste_template_source_regions = []

        # Similarity group metadata
        self.paste_full_groups = []
        self.paste_template_groups = []

        # Immutable blocks: operations that must stay together in exact order
        # Maps block_id -> list of operation fingerprints in order
        self.immutable_blocks: Dict[str, List[str]] = {}
        # Reverse mapping: fingerprint -> block_id
        self.operation_to_block: Dict[str, str] = {}

    def copy_with_operations(
        self,
        new_operations: List[Operation],
        new_constraints: List[Constraint] = None,
        new_fingerprint_map: Dict[Operation, str] = None
    ) -> 'SequencingContext':
        """Create new context with updated operations and optionally constraints."""
        new_ctx = SequencingContext(
            operations=new_operations,
            region_metadata=self.region_metadata,
            sheet_name=self.sheet_name,
            transformation_log=self.transformation_log.copy(),
            constraints=(
                new_constraints if new_constraints is not None
                else self.constraints.copy()
            ),
            fingerprint_map=(
                new_fingerprint_map if new_fingerprint_map is not None
                else self.fingerprint_map.copy()
            )
        )

        # Preserve paste constraint metadata
        new_ctx.paste_full_target_regions = list(self.paste_full_target_regions)
        new_ctx.paste_template_target_regions = list(
            self.paste_template_target_regions
        )
        new_ctx.paste_template_source_regions = list(
            self.paste_template_source_regions
        )
        new_ctx.paste_full_groups = list(self.paste_full_groups)
        new_ctx.paste_template_groups = list(self.paste_template_groups)
        new_ctx.immutable_blocks = {
            k: list(v) for k, v in self.immutable_blocks.items()
        }
        new_ctx.operation_to_block = dict(self.operation_to_block)

        return new_ctx

    def add_constraint(self, before_fp: str, after_fp: str, reason: str = None):
        """Add a constraint between two operations (mutates in-place)."""
        self.constraints.append(Constraint(
            before_fingerprint=before_fp,
            after_fingerprint=after_fp,
            reason=reason
        ))

    def add_immutable_block(self, block_id: str, operation_fingerprints: List[str]):
        """
        Register a block of operations that must stay together in exact order.
        No other operations can be scheduled between them.
        """
        self.immutable_blocks[block_id] = operation_fingerprints
        for fp in operation_fingerprints:
            self.operation_to_block[fp] = block_id

    def log(self, message: str):
        """Add message to transformation log."""
        self.transformation_log.append(message)


class BaseTransformer(ABC):
    """
    Base class for all operation transformers.

    Each transformer receives a SequencingContext and returns
    a new context with transformed operations.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize transformer with configuration.

        Args:
            config: Dict of parameters for this transformer.
                    Merged with DEFAULT_CONFIG.
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.enabled = self.config.get("enabled", True)
        self.name = self.__class__.__name__

    @abstractmethod
    def transform(self, context: SequencingContext) -> SequencingContext:
        """
        Transform operations in context.

        Args:
            context: Current sequencing context

        Returns:
            New context with transformed operations
        """
        pass

    def can_skip(self, context: SequencingContext) -> bool:
        """
        Check if transformer should be skipped.

        Returns:
            True if transformer should be skipped
        """
        return not self.enabled or len(context.operations) == 0

    def log(self, context: SequencingContext, message: str):
        """Log transformation action."""
        context.log(f"[{self.name}] {message}")

    def __repr__(self) -> str:
        return f"{self.name}(enabled={self.enabled})"
