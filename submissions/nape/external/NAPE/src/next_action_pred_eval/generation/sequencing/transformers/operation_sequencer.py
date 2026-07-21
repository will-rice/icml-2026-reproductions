"""
Operation Sequencer Transformer.

Orders operations within regions or globally based on various strategies.

Handles:
- Priority-based ordering (data before formatting)
- Spatial ordering (row-first, col-first)
- Custom ordering keys
- Hybrid ordering (priority + spatial within same priority)
- Tie-breaking (deterministic, random sampling, cached sampling)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict

from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.generation.sequencing.base import BaseTransformer, SequencingContext
from next_action_pred_eval.generation.sequencing.utils import (
    get_operation_bounds,
    is_operation_in_region,
    TieBreaker,
)


# Default priority map (same as old system)
# Lower number = earlier in sequence
DEFAULT_PRIORITY_MAP = {
    "SetValue": 1,
    "SetFormula": 1,
    "SetInput": 1,
    "PasteFrom": 1,
    "AutoFill": 1,
    "MergeCells": 2,
    "SetNumberFormat": 3,
    "SetFontProperty": 5,
    "SetAlignment": 6,
    "SetWrapText": 6,
    "SetTextOrientation": 6,
    "SetFillColor": 4,
    "SetBorder": 4,
}


class OperationSequencer(BaseTransformer):
    """
    Orders operations based on priority, spatial position, or custom keys.

    Config keys (matching ConfigSampler output):
        scope: "global" | "per_region"
        ordering_strategy: "priority" | "spatial" | "custom" | "hybrid"

        # For priority strategy:
        priority_map: dict mapping operation type names to priority levels

        # For spatial strategy:
        spatial_mode: "row_first" | "col_first"

        # For custom strategy:
        custom_key: string like "(min_row, min_col, op_priority)"

        # Tie-breaking:
        tie_mode: "deterministic_sub_order" | "sample_random" | "sample_cached"
        tie_seed: int (for random modes)
        sub_order_key: string (for deterministic mode)

        # Closing operations:
        respect_closing_operations: bool
    """

    DEFAULT_CONFIG = {
        "enabled": True,
        "scope": "per_region",
        "ordering_strategy": "custom",
        "priority_map": DEFAULT_PRIORITY_MAP,
        "spatial_mode": "row_first",
        "custom_key": "(min_row, min_col, op_priority, max_row, max_col)",
        "use_priority": True,
        "use_spatial_within_priority": True,
        "tie_mode": "deterministic_sub_order",
        "tie_seed": None,
        "sub_order_key": None,
        "respect_closing_operations": False,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.tie_breaker = TieBreaker(
            mode=self.config.get("tie_mode", "deterministic_sub_order"),
            seed=self.config.get("tie_seed")
        )

    def transform(self, context: SequencingContext) -> SequencingContext:
        """Apply operation sequencing."""
        if self.can_skip(context):
            return context

        scope = self.config.get("scope", "per_region")

        if scope == "global":
            ordered_ops = self._sequence_operations(context.operations, context)
            self.log(
                context,
                f"Sequenced {len(ordered_ops)} operations globally"
            )
            return context.copy_with_operations(ordered_ops)

        elif scope == "per_region":
            if not context.regions:
                ordered_ops = self._sequence_operations(
                    context.operations, context
                )
                self.log(context, "No regions found, sequenced globally")
                return context.copy_with_operations(ordered_ops)

            return self._sequence_per_region(context)

        else:
            self.log(context, f"Unknown scope '{scope}', skipping")
            return context

    def _sequence_per_region(self, context: SequencingContext) -> SequencingContext:
        """Sequence operations within each region independently."""
        # Group operations by region
        ops_by_region = defaultdict(list)

        for op in context.operations:
            region_id = self._find_region_for_operation(op, context)
            ops_by_region[region_id].append(op)

        # Build region_id -> region dict for quick lookup
        region_by_id = {r.get("id"): r for r in context.regions}

        # Sequence each region
        ordered_ops = []
        regions_processed = 0
        closing_ops_moved = 0

        # Process in current region order (RegionOrchestrator should have
        # ordered them already)
        seen_regions = set()
        for op in context.operations:
            region_id = self._find_region_for_operation(op, context)
            if region_id not in seen_regions:
                if region_id in ops_by_region:
                    region_ops = ops_by_region[region_id]
                    sequenced = self._sequence_operations(region_ops, context)

                    # Apply closing operations reordering if enabled
                    if (
                        self.config.get("respect_closing_operations", False)
                        and region_id is not None
                    ):
                        region = region_by_id.get(region_id)
                        if region:
                            sequenced, num_moved = (
                                self._apply_closing_operations(sequenced, region)
                            )
                            closing_ops_moved += num_moved

                    ordered_ops.extend(sequenced)
                    seen_regions.add(region_id)
                    if region_id is not None:
                        regions_processed += 1

        log_msg = f"Sequenced operations in {regions_processed} regions"
        if closing_ops_moved > 0:
            log_msg += f", moved {closing_ops_moved} closing operations to end"
        self.log(context, log_msg)
        return context.copy_with_operations(ordered_ops)

    def _sequence_operations(
        self,
        operations: List[Operation],
        context: SequencingContext
    ) -> List[Operation]:
        """Apply sequencing strategy to a list of operations."""
        if not operations:
            return []

        strategy = self.config.get("ordering_strategy", "hybrid")

        if strategy == "priority":
            return self._sequence_by_priority(operations, context)
        elif strategy == "spatial":
            return self._sequence_by_spatial(operations, context)
        elif strategy == "custom":
            return self._sequence_by_custom_key(operations, context)
        elif strategy == "hybrid":
            return self._sequence_hybrid(operations, context)
        else:
            return operations

    def _sequence_by_priority(
        self,
        operations: List[Operation],
        context: SequencingContext
    ) -> List[Operation]:
        """Sequence by operation priority levels."""
        priority_map = self.config.get("priority_map", DEFAULT_PRIORITY_MAP)

        # Group by priority
        ops_by_priority = defaultdict(list)
        for op in operations:
            op_type = type(op).__name__
            priority = priority_map.get(op_type, 999)
            ops_by_priority[priority].append(op)

        # Sort by priority level, then break ties
        result = []
        for priority in sorted(ops_by_priority.keys()):
            group = ops_by_priority[priority]
            tied_groups = [group]
            sub_key = self._build_sub_order_key(context)
            broken = self.tie_breaker.break_ties(tied_groups, sub_key)
            result.extend(broken)

        return result

    def _sequence_by_spatial(
        self,
        operations: List[Operation],
        context: SequencingContext
    ) -> List[Operation]:
        """Sequence by spatial position (row-first or col-first)."""
        spatial_mode = self.config.get("spatial_mode", "row_first")

        if spatial_mode == "row_first":
            def spatial_key(op):
                min_col, min_row, max_col, max_row = get_operation_bounds(op)
                return (min_row, min_col, max_row, max_col)
        else:  # col_first
            def spatial_key(op):
                min_col, min_row, max_col, max_row = get_operation_bounds(op)
                return (min_col, min_row, max_col, max_row)

        # Group by spatial key
        ops_by_key = defaultdict(list)
        for op in operations:
            key = spatial_key(op)
            ops_by_key[key].append(op)

        # Sort and break ties
        result = []
        for key in sorted(ops_by_key.keys()):
            group = ops_by_key[key]
            sub_key = self._build_sub_order_key(context)
            broken = self.tie_breaker.break_ties([group], sub_key)
            result.extend(broken)

        return result

    def _sequence_by_custom_key(
        self,
        operations: List[Operation],
        context: SequencingContext
    ) -> List[Operation]:
        """Sequence by custom key string."""
        custom_key_str = self.config.get("custom_key")
        if not custom_key_str:
            return operations

        key_func = self._build_custom_key_function(custom_key_str, context)

        # Group by key
        ops_by_key = defaultdict(list)
        for op in operations:
            key = key_func(op)
            ops_by_key[key].append(op)

        # Sort and break ties
        result = []
        for key in sorted(ops_by_key.keys()):
            group = ops_by_key[key]
            sub_key = self._build_sub_order_key(context)
            broken = self.tie_breaker.break_ties([group], sub_key)
            result.extend(broken)

        return result

    def _sequence_hybrid(
        self,
        operations: List[Operation],
        context: SequencingContext
    ) -> List[Operation]:
        """
        Hybrid: Priority-based grouping, then spatial within each priority.
        """
        use_priority = self.config.get("use_priority", True)
        use_spatial = self.config.get("use_spatial_within_priority", True)

        if not use_priority:
            return self._sequence_by_spatial(operations, context)

        priority_map = self.config.get("priority_map", DEFAULT_PRIORITY_MAP)
        spatial_mode = self.config.get("spatial_mode", "row_first")

        if spatial_mode == "row_first":
            def spatial_key(op):
                min_col, min_row, max_col, max_row = get_operation_bounds(op)
                return (min_row, min_col, max_row, max_col)
        else:
            def spatial_key(op):
                min_col, min_row, max_col, max_row = get_operation_bounds(op)
                return (min_col, min_row, max_col, max_row)

        # Group by priority first
        ops_by_priority = defaultdict(list)
        for op in operations:
            op_type = type(op).__name__
            priority = priority_map.get(op_type, 999)
            ops_by_priority[priority].append(op)

        # Within each priority, sort spatially if enabled
        result = []
        for priority in sorted(ops_by_priority.keys()):
            priority_ops = ops_by_priority[priority]

            if use_spatial and len(priority_ops) > 1:
                ops_by_spatial = defaultdict(list)
                for op in priority_ops:
                    s_key = spatial_key(op)
                    ops_by_spatial[s_key].append(op)

                for s_key in sorted(ops_by_spatial.keys()):
                    group = ops_by_spatial[s_key]
                    sub_key = self._build_sub_order_key(context)
                    broken = self.tie_breaker.break_ties([group], sub_key)
                    result.extend(broken)
            else:
                sub_key = self._build_sub_order_key(context)
                broken = self.tie_breaker.break_ties(
                    [priority_ops], sub_key
                )
                result.extend(broken)

        return result

    def _build_custom_key_function(
        self,
        key_str: str,
        context: SequencingContext
    ) -> Callable:
        """Build key function from string like '(min_row, min_col, op_priority)'."""
        key_str = key_str.strip('()')
        components = [c.strip() for c in key_str.split(',')]

        priority_map = self.config.get("priority_map", DEFAULT_PRIORITY_MAP)

        def key_func(op):
            values = []
            for comp in components:
                if comp == "min_row":
                    values.append(get_operation_bounds(op)[1])
                elif comp == "min_col":
                    values.append(get_operation_bounds(op)[0])
                elif comp == "max_row":
                    values.append(get_operation_bounds(op)[3])
                elif comp == "max_col":
                    values.append(get_operation_bounds(op)[2])
                elif comp == "op_priority":
                    op_type = type(op).__name__
                    values.append(priority_map.get(op_type, 999))
                elif comp == "op_type":
                    values.append(type(op).__name__)
                else:
                    values.append(0)
            return tuple(values)

        return key_func

    def _build_sub_order_key(self, context: SequencingContext) -> Callable:
        """Build sub-ordering key for tie-breaking."""
        sub_order_str = self.config.get("sub_order_key")

        if sub_order_str:
            return self._build_custom_key_function(sub_order_str, context)
        else:
            return lambda op: type(op).__name__

    def _find_region_for_operation(
        self,
        operation: Operation,
        context: SequencingContext
    ):
        """Find which region an operation belongs to."""
        for region in context.regions:
            if is_operation_in_region(operation, region, mode="overlap"):
                return region["id"]
        return None

    def _apply_closing_operations(
        self,
        operations: List[Operation],
        region: Dict
    ) -> Tuple[List[Operation], int]:
        """
        Move operations matching region's closing_operations to the end
        of the sequence.

        Args:
            operations: Sequenced operations for a region
            region: Region dict with optional 'closing_operations' field

        Returns:
            Tuple of (reordered operations, number of operations moved)
        """
        from openpyxl.utils import range_boundaries

        closing_ops_spec = region.get("closing_operations", [])
        if not closing_ops_spec:
            return operations, 0

        # Map operation_type names from region analysis to actual class names
        OP_TYPE_MAPPING = {
            "INPUT": {"SetInput", "SetValue"},
            "FONT_NAME": {"SetFontProperty"},
            "FONT_SIZE": {"SetFontProperty"},
            "FONT_BOLD": {"SetFontProperty"},
            "FONT_ITALIC": {"SetFontProperty"},
            "FONT_UNDERLINE": {"SetFontProperty"},
            "FONT_COLOR": {"SetFontProperty"},
            "FILL_COLOR": {"SetFillColor"},
        }

        def matches_closing_op(op: Operation, closing_spec: Dict) -> bool:
            """Check if an operation matches a closing operation spec."""
            spec_type = closing_spec.get("operation_type", "")
            spec_range = closing_spec.get("range", "")

            op_class_name = type(op).__name__
            allowed_classes = OP_TYPE_MAPPING.get(spec_type, set())
            if op_class_name not in allowed_classes:
                return False

            if not spec_range:
                return True

            try:
                s_min_col, s_min_row, s_max_col, s_max_row = (
                    range_boundaries(spec_range)
                )
                o_min_col, o_min_row, o_max_col, o_max_row = (
                    range_boundaries(op.cell_range.range)
                )
                return not (
                    o_max_col < s_min_col or o_min_col > s_max_col or
                    o_max_row < s_min_row or o_min_row > s_max_row
                )
            except Exception:
                return False

        normal_ops = []
        closing_ops = []

        for op in operations:
            is_closing = any(
                matches_closing_op(op, spec) for spec in closing_ops_spec
            )
            if is_closing:
                closing_ops.append(op)
            else:
                normal_ops.append(op)

        return normal_ops + closing_ops, len(closing_ops)
