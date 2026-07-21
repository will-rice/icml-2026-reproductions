"""
Region Orchestrator Transformer.

Controls how regions are ordered and processed in the sequencing pipeline.

Modes:
- sequential: Process regions in order (row-first, col-first, id-order, custom)
- parallel: No region-based ordering (all operations treated equally)
- dependency: Use region_dependencies for topological sort
- mixed: Custom ordering with parallel groups
"""

from typing import Callable, Dict, List
from collections import defaultdict

from next_action_pred_eval.generation.sequencing.base import BaseTransformer, SequencingContext
from next_action_pred_eval.generation.sequencing.utils import (
    is_operation_in_region,
    get_region_bounds,
    topological_sort,
)


class RegionOrchestrator(BaseTransformer):
    """
    Controls the order in which regions are processed.

    Config keys (matching ConfigSampler output):
        mode: "sequential" | "parallel" | "dependency" | "mixed"
        sequential_ordering: "row_first" | "col_first" | "id_order" | "(key, ...)"
        dependency_override: bool - Use dependencies when available (default: True)
        custom_ordering: list or string for mixed mode
    """

    DEFAULT_CONFIG = {
        "enabled": True,
        "mode": "dependency",
        "sequential_ordering": "row_first",
        "dependency_override": True,
        "custom_ordering": None,
    }

    def transform(self, context: SequencingContext) -> SequencingContext:
        """Apply region orchestration."""
        if self.can_skip(context) or not context.regions:
            return context

        mode = self.config.get("mode", "dependency")

        if mode == "parallel":
            self.log(context, "Parallel mode - no region ordering")
            return context

        elif mode == "dependency":
            return self._order_by_dependencies(context)

        elif mode == "sequential":
            return self._order_sequentially(context)

        elif mode == "mixed":
            return self._order_mixed(context)

        else:
            self.log(context, f"Unknown mode '{mode}', skipping")
            return context

    def _order_by_dependencies(
        self, context: SequencingContext
    ) -> SequencingContext:
        """Order regions using topological sort on dependencies."""
        # Check if dependencies are available
        if (
            not context.region_dependencies
            and self.config.get("dependency_override", True)
        ):
            self.log(
                context,
                "No dependencies found, falling back to sequential"
            )
            return self._order_sequentially(context)

        # Get all region IDs
        region_ids = [region["id"] for region in context.regions]

        # Topological sort using old-style API (items, dependencies)
        sorted_ids = topological_sort(
            region_ids, context.region_dependencies
        )

        # Group operations by region
        ops_by_region = self._group_operations_by_region(context)

        # Reconstruct in dependency order
        ordered_ops = []
        for region_id in sorted_ids:
            if region_id in ops_by_region:
                ordered_ops.extend(ops_by_region[region_id])

        # Add unassigned operations at the end
        ordered_ops.extend(ops_by_region.get(None, []))

        self.log(
            context,
            f"Ordered {len(context.regions)} regions by dependencies"
        )
        return context.copy_with_operations(ordered_ops)

    def _order_sequentially(
        self, context: SequencingContext
    ) -> SequencingContext:
        """Order regions sequentially based on ordering parameter."""
        ordering = self.config.get("sequential_ordering", "row_first")

        # Build sorting key for regions
        if ordering == "row_first":
            sort_key = lambda r: (
                get_region_bounds(r)[1], get_region_bounds(r)[0]
            )
        elif ordering == "col_first":
            sort_key = lambda r: (
                get_region_bounds(r)[0], get_region_bounds(r)[1]
            )
        elif ordering == "id_order":
            sort_key = lambda r: r["id"]
        elif ordering.startswith("(") and ordering.endswith(")"):
            sort_key = self._build_region_sort_key(ordering)
        else:
            sort_key = lambda r: r["id"]

        sorted_regions = sorted(context.regions, key=sort_key)

        # Group operations by region
        ops_by_region = self._group_operations_by_region(context)

        # Reconstruct in sorted order
        ordered_ops = []
        for region in sorted_regions:
            region_id = region["id"]
            if region_id in ops_by_region:
                ordered_ops.extend(ops_by_region[region_id])

        # Add unassigned operations
        ordered_ops.extend(ops_by_region.get(None, []))

        self.log(
            context,
            f"Ordered {len(sorted_regions)} regions sequentially ({ordering})"
        )
        return context.copy_with_operations(ordered_ops)

    def _order_mixed(self, context: SequencingContext) -> SequencingContext:
        """Mixed mode: Order regions with custom rules."""
        custom_ordering = self.config.get("custom_ordering")

        if custom_ordering is None:
            self.log(
                context,
                "Mixed mode requires custom_ordering, falling back to sequential"
            )
            return self._order_sequentially(context)

        if isinstance(custom_ordering, list):
            return self._order_by_explicit_list(context, custom_ordering)

        if isinstance(custom_ordering, str):
            if custom_ordering == "row_first":
                key_func = lambda r: get_region_bounds(r)[1]
            elif custom_ordering == "col_first":
                key_func = lambda r: get_region_bounds(r)[0]
            elif custom_ordering.startswith("("):
                key_func = self._build_region_sort_key(custom_ordering)
            else:
                key_func = lambda r: r.get(custom_ordering, 0)

            groups = defaultdict(list)
            for region in context.regions:
                key_val = key_func(region)
                groups[key_val].append(region)

            sorted_keys = sorted(groups.keys())

            ops_by_region = self._group_operations_by_region(context)

            ordered_ops = []
            for key_val in sorted_keys:
                for region in groups[key_val]:
                    region_id = region["id"]
                    if region_id in ops_by_region:
                        ordered_ops.extend(ops_by_region[region_id])

            ordered_ops.extend(ops_by_region.get(None, []))

            parallel_groups = len(groups)
            self.log(
                context,
                f"Mixed mode: {len(context.regions)} regions in "
                f"{parallel_groups} sequential groups"
            )
            return context.copy_with_operations(ordered_ops)

        self.log(
            context,
            "Unknown custom_ordering format, falling back to sequential"
        )
        return self._order_sequentially(context)

    def _order_by_explicit_list(
        self, context: SequencingContext, order_list: List[int]
    ) -> SequencingContext:
        """Order regions by explicit list of region IDs."""
        ops_by_region = self._group_operations_by_region(context)

        ordered_ops = []
        for region_id in order_list:
            if region_id in ops_by_region:
                ordered_ops.extend(ops_by_region[region_id])

        specified_ids = set(order_list)
        for region in context.regions:
            region_id = region["id"]
            if region_id not in specified_ids and region_id in ops_by_region:
                ordered_ops.extend(ops_by_region[region_id])

        ordered_ops.extend(ops_by_region.get(None, []))

        self.log(context, f"Ordered by explicit list: {order_list}")
        return context.copy_with_operations(ordered_ops)

    def _group_operations_by_region(
        self, context: SequencingContext
    ) -> Dict[int, List]:
        """Group operations by which region they belong to."""
        ops_by_region = defaultdict(list)

        for op in context.operations:
            region_id = self._find_region_for_operation(op, context)
            ops_by_region[region_id].append(op)

        return ops_by_region

    def _find_region_for_operation(self, operation, context: SequencingContext):
        """Find which region an operation belongs to (None if unassigned)."""
        for region in context.regions:
            if is_operation_in_region(operation, region, mode="contain"):
                return region["id"]

        return None

    def _build_region_sort_key(self, key_string: str) -> Callable:
        """Build sorting key from string like '(region_type, min_row, min_col)'."""
        key_string = key_string.strip('()')
        components = [c.strip() for c in key_string.split(',')]

        def sort_key(region):
            values = []
            for comp in components:
                if comp == "min_row":
                    values.append(get_region_bounds(region)[1])
                elif comp == "min_col":
                    values.append(get_region_bounds(region)[0])
                elif comp == "max_row":
                    values.append(get_region_bounds(region)[3])
                elif comp == "max_col":
                    values.append(get_region_bounds(region)[2])
                elif comp in region:
                    values.append(region[comp])
                else:
                    values.append(0)
            return tuple(values)

        return sort_key

