"""
Sequencing Utilities.

Shared utility functions for the sequencing pipeline.
"""

from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Tuple
import random

from openpyxl.utils import range_boundaries

from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.core.operation import Operation


def ranges_overlap(range1: CellRange, range2: CellRange) -> bool:
    """
    Check if two CellRanges overlap.

    Args:
        range1: First cell range
        range2: Second cell range

    Returns:
        True if the ranges overlap
    """
    if range1.sheet != range2.sheet:
        return False

    try:
        min_col1, min_row1, max_col1, max_row1 = range_boundaries(range1.range)
        min_col2, min_row2, max_col2, max_row2 = range_boundaries(range2.range)
        return not (
            max_col1 < min_col2 or max_col2 < min_col1 or
            max_row1 < min_row2 or max_row2 < min_row1
        )
    except Exception:
        return False


def is_operation_in_region(
    op: Operation,
    region: Dict[str, Any],
    mode: str = "overlap"
) -> bool:
    """
    Check if operation relates to a region.

    Args:
        op: Operation to check
        region: Region dict with 'range' key (e.g., "A1:C10")
        mode: "overlap" (any intersection) or "contain" (fully contained)

    Returns:
        True if operation is in region according to mode
    """
    try:
        op_min_col, op_min_row, op_max_col, op_max_row = range_boundaries(
            op.cell_range.range
        )
        r_min_col, r_min_row, r_max_col, r_max_row = range_boundaries(
            region["range"]
        )
    except Exception:
        return False

    if mode == "contain":
        return (
            op_min_col >= r_min_col and op_max_col <= r_max_col and
            op_min_row >= r_min_row and op_max_row <= r_max_row
        )
    else:  # overlap
        return not (
            op_max_col < r_min_col or r_max_col < op_min_col or
            op_max_row < r_min_row or r_max_row < op_min_row
        )


def get_operation_bounds(op: Operation) -> Tuple[int, int, int, int]:
    """
    Get cell range boundaries for an operation.

    Args:
        op: Operation to get bounds from

    Returns:
        Tuple of (min_col, min_row, max_col, max_row)
    """
    try:
        return range_boundaries(op.cell_range.range)
    except Exception:
        return (float('inf'), float('inf'), float('inf'), float('inf'))


def get_region_bounds(region: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """
    Get cell range boundaries for a region.

    Args:
        region: Region dict with 'range' key (e.g., "A1:C10")

    Returns:
        Tuple of (min_col, min_row, max_col, max_row)
    """
    try:
        return range_boundaries(region["range"])
    except Exception:
        return (float('inf'), float('inf'), float('inf'), float('inf'))


def topological_sort(
    items: List[Any],
    dependencies: Dict[Any, Any]
) -> List[Any]:
    """
    Perform topological sort on items based on dependencies.

    Args:
        items: List of items to sort
        dependencies: Dict where dependencies[item] = item_that_must_come_before

    Returns:
        List of items in topological order
    """
    # Build adjacency list and in-degree count
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for item in items:
        in_degree[item] = 0

    # Build graph from dependencies
    for dependent, dependency in dependencies.items():
        if dependent in in_degree and dependency in in_degree:
            graph[dependency].append(dependent)
            in_degree[dependent] += 1

    # Kahn's algorithm
    queue = deque([item for item in items if in_degree[item] == 0])
    result = []

    while queue:
        current = queue.popleft()
        result.append(current)

        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If not all nodes processed, there's a cycle - return simple sorted order
    if len(result) != len(items):
        return sorted(items)

    return result


class TieBreaker:
    """
    Handles tie-breaking for operations with same sort key.

    Modes:
    - deterministic_sub_order: Use secondary ordering key
    - sample_random: Random with seed
    - sample_cached: Generate random order once, cache and reuse
    """

    def __init__(
        self,
        mode: str = "deterministic_sub_order",
        seed: Optional[int] = None
    ):
        """
        Initialize tie breaker.

        Args:
            mode: "deterministic_sub_order" | "sample_random" | "sample_cached"
            seed: Random seed for reproducibility (used in sample modes)
        """
        self.mode = mode
        self.seed = seed
        self.cache: Dict[Any, List[int]] = {}

        if seed is not None:
            random.seed(seed)

    def break_ties(
        self,
        groups: List[List[Operation]],
        sub_order_key: Callable = None
    ) -> List[Operation]:
        """
        Break ties for operations with same primary sort key.

        Args:
            groups: List of lists, where each inner list has operations
                    with the same primary key
            sub_order_key: Secondary ordering key (for deterministic mode)

        Returns:
            Flattened list with ties broken
        """
        result = []

        for group in groups:
            if len(group) <= 1:
                result.extend(group)
                continue

            if self.mode == "deterministic_sub_order":
                if sub_order_key:
                    sorted_group = sorted(group, key=sub_order_key)
                else:
                    sorted_group = sorted(
                        group, key=lambda op: type(op).__name__
                    )
                result.extend(sorted_group)

            elif self.mode == "sample_random":
                shuffled = group.copy()
                random.shuffle(shuffled)
                result.extend(shuffled)

            elif self.mode == "sample_cached":
                group_key = tuple(id(op) for op in group)

                if group_key in self.cache:
                    cached_indices = self.cache[group_key]
                    ordered = [group[i] for i in cached_indices]
                else:
                    indices = list(range(len(group)))
                    random.shuffle(indices)
                    self.cache[group_key] = indices
                    ordered = [group[i] for i in indices]

                result.extend(ordered)

        return result
