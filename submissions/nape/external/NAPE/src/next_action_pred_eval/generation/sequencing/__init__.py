"""
Sequencing module - Transform operations into human-like sequences.

This module provides a configurable pipeline for sequencing spreadsheet operations
in a way that mimics human editing patterns.

Example:
    from next_action_pred_eval.generation.sequencing import SequencingEngine

    engine = SequencingEngine.from_config("path/to/config.yaml")
    sequenced_ops = engine.sequence(operations, region_metadata)

Key components:
- SequencingEngine: Main orchestrator for the pipeline
- SequencingContext: Mutable context passed through transformers
- BaseTransformer: Abstract base for all transformers
- Transformers: RegionOrchestrator, OperationSequencer, ConstraintEnforcer
"""

from next_action_pred_eval.generation.sequencing.base import (
    BaseTransformer,
    Constraint,
    SequencingContext,
)
from next_action_pred_eval.generation.sequencing.engine import SequencingEngine
from next_action_pred_eval.generation.sequencing.transformers import (
    RegionOrchestrator,
    OperationSequencer,
    ConstraintEnforcer,
)

__all__ = [
    # Core classes
    "SequencingEngine",
    "SequencingContext",
    "BaseTransformer",
    "Constraint",
    # Transformers
    "RegionOrchestrator",
    "OperationSequencer",
    "ConstraintEnforcer",
]
