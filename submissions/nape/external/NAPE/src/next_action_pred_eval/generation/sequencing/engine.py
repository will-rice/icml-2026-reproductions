"""
Sequencing Engine.

Main orchestrator for the operation sequencing pipeline.
Builds and executes transformation pipelines from configuration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import yaml

from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.generation.sequencing.base import (
    BaseTransformer,
    SequencingContext,
)

logger = logging.getLogger(__name__)


class SequencingEngine:
    """
    Orchestrator for the operation sequencing pipeline.

    Takes operations and region metadata, applies a series of transformers,
    and returns reordered/modified operations.

    Example:
        engine = SequencingEngine.from_config("path/to/config.yaml")
        result = engine.sequence(operations, region_metadata)
    """

    # Registry of available transformers
    TRANSFORMER_REGISTRY: Dict[str, Type[BaseTransformer]] = {}

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize sequencing engine.

        Args:
            config: Configuration dict with 'pipeline' key listing transformers
        """
        self.config = config
        self.pipeline: List[BaseTransformer] = self._build_pipeline()

    @classmethod
    def register_transformer(cls, name: str, transformer_class: Type[BaseTransformer]):
        """
        Register a transformer class.

        Args:
            name: Name to register under
            transformer_class: Transformer class
        """
        cls.TRANSFORMER_REGISTRY[name] = transformer_class

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'SequencingEngine':
        """
        Create engine from YAML config file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configured SequencingEngine
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'SequencingEngine':
        """
        Create engine from config dict.

        Args:
            config: Configuration dictionary

        Returns:
            Configured SequencingEngine
        """
        return cls(config)

    def _build_pipeline(self) -> List[BaseTransformer]:
        """Build transformer pipeline from config."""
        pipeline = []
        pipeline_config = self.config.get('pipeline', [])

        for transformer_config in pipeline_config:
            if isinstance(transformer_config, str):
                # Simple name reference
                name = transformer_config
                params = {}
            elif isinstance(transformer_config, dict):
                # Support both key formats:
                #   {"name": ..., "params": ...}  (engine format)
                #   {"type": ..., "config": ...}  (config_sampler format)
                name = transformer_config.get('name') or transformer_config.get('type')
                params = transformer_config.get('params') or transformer_config.get('config', {})
            else:
                logger.warning(f"Invalid transformer config: {transformer_config}")
                continue

            transformer_class = self.TRANSFORMER_REGISTRY.get(name)
            if transformer_class is None:
                logger.warning(f"Unknown transformer: {name}")
                continue

            transformer = transformer_class(params)
            pipeline.append(transformer)
            logger.debug(f"Added transformer: {name}")

        # Auto-add ConstraintEnforcer at end if not present
        if pipeline and not any(
            t.__class__.__name__ == 'ConstraintEnforcer' for t in pipeline
        ):
            enforcer_class = self.TRANSFORMER_REGISTRY.get('ConstraintEnforcer')
            if enforcer_class:
                pipeline.append(enforcer_class({}))
                logger.debug("Auto-added ConstraintEnforcer")

        return pipeline

    def sequence(
        self,
        operations: List[Operation],
        region_metadata: Optional[Dict[str, Any]] = None,
        sheet_name: str = "Sheet1"
    ) -> List[Operation]:
        """
        Apply sequencing transformations to operations.

        Args:
            operations: List of operations to sequence
            region_metadata: Optional region definitions and relationships
            sheet_name: Name of the sheet being processed

        Returns:
            Reordered/modified list of operations
        """
        if not operations:
            return []

        # Apply adaptive config overrides based on sheet characteristics
        self._apply_adaptive_overrides(operations, region_metadata or {})

        # Build initial context (auto-extracts regions, dependencies, etc.)
        context = SequencingContext(
            operations=list(operations),
            region_metadata=region_metadata or {},
            sheet_name=sheet_name,
        )

        context.log(f"Starting sequencing with {len(operations)} operations")

        # Apply transformers
        for transformer in self.pipeline:
            if transformer.can_skip(context):
                context.log(f"Skipping {transformer.name}")
                continue

            try:
                context = transformer.transform(context)
                context.log(
                    f"Applied {transformer.name}: "
                    f"{len(context.operations)} operations"
                )
            except Exception as e:
                logger.error(f"Error in {transformer.name}: {e}")
                context.log(f"Error in {transformer.name}: {e}")

        return context.operations

    def _apply_adaptive_overrides(
        self,
        operations: List[Operation],
        region_metadata: Dict[str, Any],
    ) -> None:
        """
        Analyze sheet characteristics and override pipeline config for realism.

        Applies heuristics:
        1. If pasted_ranges exist and any covers >128 cells → enable force_merge_pasted_ranges
        2. If no pasted_ranges and total input/value ops >1024 → force merge_inputs=True
        3. If sheet is very tall (rows > 8*cols) → force row-first ordering
        4. If sheet is very wide (cols > 8*rows) → force col-first ordering

        Mutates transformer configs in self.pipeline in-place.
        """
        from next_action_pred_eval.core.operations import SetInput, SetValue, SetFormula

        # --- Compute sheet bounding box ---
        max_row, max_col = 0, 0
        for op in operations:
            try:
                coords = op.cell_range.get_coordinates()
                max_row = max(max_row, coords[2])  # end_row
                max_col = max(max_col, coords[3])  # end_col
            except Exception:
                pass

        # --- Count input/value ops ---
        input_value_ops = [
            op for op in operations if isinstance(op, (SetInput, SetValue))
        ]
        total_input_ops = len(input_value_ops)

        # --- Count single-cell SetValue/SetFormula ops ---
        single_cell_count = 0
        for op in operations:
            if isinstance(op, (SetValue, SetFormula)) and not op.is_inverse:
                r1, c1, r2, c2 = op.cell_range.get_coordinates()
                if r1 == r2 and c1 == c2:
                    single_cell_count += 1

        # --- Check for pasted ranges ---
        pasted_ranges = region_metadata.get("pasted_ranges", [])
        has_pasted_ranges = bool(pasted_ranges)

        # Check if any pasted range is large (>128 cells)
        has_large_paste = False
        if has_pasted_ranges:
            from openpyxl.utils import range_boundaries
            for pr in pasted_ranges:
                range_str = pr.get("range") if isinstance(pr, dict) else pr
                if not range_str:
                    continue
                try:
                    min_c, min_r, max_c, max_r = range_boundaries(range_str)
                    cell_count = (max_c - min_c + 1) * (max_r - min_r + 1)
                    if cell_count > 128:
                        has_large_paste = True
                        break
                except Exception:
                    pass

        # --- Find relevant transformers ---
        merger = next(
            (t for t in self.pipeline if t.__class__.__name__ == 'OperationMerger'),
            None,
        )
        sequencer = next(
            (t for t in self.pipeline if t.__class__.__name__ == 'OperationSequencer'),
            None,
        )
        autofill_detector = next(
            (t for t in self.pipeline if t.__class__.__name__ == 'AutoFillDetector'),
            None,
        )

        overrides_applied = []

        # --- Override 1: Force merge large pasted ranges ---
        if has_large_paste and merger:
            merger.config.setdefault("merge_params", {})["force_merge_pasted_ranges"] = True
            overrides_applied.append(
                f"Large pasted range (>128 cells) → force_merge_pasted_ranges"
            )

        # --- Override 2: Force merge inputs when too many ---
        if total_input_ops > 1024 and merger:
            merger.config.setdefault("merge_params", {})["merge_inputs"] = True
            # Don't split by type — these are clearly all data cells
            merger.config["merge_params"]["sort_input_by_type"] = False
            overrides_applied.append(
                f"{total_input_ops} input ops → force merge_inputs + disable sort_by_type"
            )

        # --- Override 3: Force AutoFill value detection on large sheets ---
        if single_cell_count > 64 and autofill_detector:
            autofill_detector.config["detect_values"] = True
            overrides_applied.append(
                f"{single_cell_count} single-cell ops → force detect_values"
            )

        # --- Override 4/5: Dimension-based ordering ---
        if max_row > 0 and max_col > 0:
            if max_row > 8 * max_col:
                # Very tall sheet (data table with many rows)
                if merger:
                    merger.config.setdefault("merge_params", {})["row_first"] = True
                if sequencer:
                    sequencer.config["custom_key"] = (
                        "(min_row, min_col, op_priority, max_row, max_col)"
                    )
                overrides_applied.append(
                    f"Tall sheet ({max_row}r x {max_col}c) → row-first ordering"
                )
            elif max_col > 8 * max_row:
                # Very wide sheet
                if merger:
                    merger.config.setdefault("merge_params", {})["row_first"] = False
                if sequencer:
                    sequencer.config["custom_key"] = (
                        "(min_col, min_row, op_priority, max_col, max_row)"
                    )
                overrides_applied.append(
                    f"Wide sheet ({max_row}r x {max_col}c) → col-first ordering"
                )

        if overrides_applied:
            for msg in overrides_applied:
                logger.info(f"Adaptive override: {msg}")

    def get_transformation_log(self) -> List[str]:
        """
        Get log from last transformation.

        Note: This returns empty list if no transformation has been run yet,
        or if you need the log, capture it from the returned context.
        """
        return []  # Log is stored in context, not engine


def register_default_transformers():
    """Register default transformers with the engine."""
    # Import here to avoid circular imports
    try:
        from next_action_pred_eval.generation.sequencing.transformers import (
            RegionOrchestrator,
            OperationSequencer,
            ConstraintEnforcer,
        )

        SequencingEngine.register_transformer('RegionOrchestrator', RegionOrchestrator)
        SequencingEngine.register_transformer('OperationSequencer', OperationSequencer)
        SequencingEngine.register_transformer('ConstraintEnforcer', ConstraintEnforcer)
    except ImportError as e:
        logger.debug(f"Could not register default transformers: {e}")

    # Register additional transformers
    try:
        from next_action_pred_eval.generation.sequencing.transformers.date_noise_filter import DateNoiseFilter
        SequencingEngine.register_transformer('DateNoiseFilter', DateNoiseFilter)
    except ImportError as e:
        logger.debug(f"Could not register DateNoiseFilter: {e}")

    try:
        from next_action_pred_eval.generation.sequencing.transformers.border_consolidator import BorderConsolidator
        SequencingEngine.register_transformer('BorderConsolidator', BorderConsolidator)
    except ImportError as e:
        logger.debug(f"Could not register BorderConsolidator: {e}")

    try:
        from next_action_pred_eval.generation.sequencing.transformers.default_value_filter import DefaultValueFilter
        SequencingEngine.register_transformer('DefaultValueFilter', DefaultValueFilter)
    except ImportError as e:
        logger.debug(f"Could not register DefaultValueFilter: {e}")

    try:
        from next_action_pred_eval.generation.sequencing.transformers.operation_merger import OperationMerger
        SequencingEngine.register_transformer('OperationMerger', OperationMerger)
    except ImportError as e:
        logger.debug(f"Could not register OperationMerger: {e}")

    try:
        from next_action_pred_eval.generation.sequencing.transformers.formatting_consolidator import FormattingConsolidator
        SequencingEngine.register_transformer('FormattingConsolidator', FormattingConsolidator)
    except ImportError as e:
        logger.debug(f"Could not register FormattingConsolidator: {e}")

    try:
        from next_action_pred_eval.generation.sequencing.transformers.autofill_detector import AutoFillDetector
        SequencingEngine.register_transformer('AutoFillDetector', AutoFillDetector)
    except ImportError as e:
        logger.debug(f"Could not register AutoFillDetector: {e}")

    try:
        from next_action_pred_eval.generation.sequencing.transformers.input_splitter import InputSplitter
        SequencingEngine.register_transformer('InputSplitter', InputSplitter)
    except ImportError as e:
        logger.debug(f"Could not register InputSplitter: {e}")


# Register transformers on module load
register_default_transformers()
