"""
Constraint Enforcer Transformer.

Reorders operations for paste patterns and fingerprint-based constraints.

Per-group block ordering (respects region boundaries):
- paste_full: For each similarity group:
    Source ops << PASTE << INVERSE << Target ops
- Fingerprint constraints: Greedy schedule respecting before-after constraints
    and immutable blocks.
"""

from collections import defaultdict
from typing import Dict, List

from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.generation.sequencing.base import (
    BaseTransformer,
    SequencingContext,
)


class ConstraintEnforcer(BaseTransformer):
    """
    Enforces operation ordering using per-group block constraints.
    Processes each similarity group independently to respect region boundaries.
    Preserves spatial ordering within each block.
    """

    DEFAULT_CONFIG = {
        "enabled": True,
    }

    def transform(self, context: SequencingContext) -> SequencingContext:
        """Apply constraint enforcement."""
        if self.can_skip(context):
            return context

        current_ops = context.operations

        # Apply paste_full constraints
        if (
            hasattr(context, 'paste_full_target_regions')
            and context.paste_full_target_regions
        ):
            current_ops = self._apply_paste_full_constraints(
                current_ops, context
            )

        # Apply fingerprint constraints (for paste_template and other
        # constraints from OperationMerger)
        current_ops = self._apply_fingerprint_constraints(
            current_ops, context
        )

        return context.copy_with_operations(current_ops)

    def _apply_paste_full_constraints(
        self,
        operations: List[Operation],
        context: SequencingContext
    ) -> List[Operation]:
        """
        Per-group 4-block ordering: For each similarity group independently:
        Source ops << PASTE ops << INVERSE ops << Target non-INVERSE ops
        """
        from next_action_pred_eval.core.operations import SetInput, PasteFrom

        if (
            not hasattr(context, 'paste_full_groups')
            or not context.paste_full_groups
        ):
            return operations

        result_ops = operations.copy()

        for group in context.paste_full_groups:
            source_range = group.get('source_range')
            target_ranges = group['target_ranges']
            paste_execution = group.get('paste_execution', 'block')

            # Skip reordering for interleaved mode
            if paste_execution == "interleaved":
                continue

            # Collect affected ranges for THIS group only
            affected_ranges = target_ranges.copy()
            if source_range:
                affected_ranges.append(source_range)

            # Extract source ranges from PASTE operations if not in group
            if not source_range:
                source_ranges_from_paste = []
                for op in result_ops:
                    if isinstance(op, PasteFrom):
                        op_range = op.cell_range.range
                        if any(
                            self._ranges_overlap_str(op_range, tr)
                            for tr in target_ranges
                        ):
                            if hasattr(op, 'source_range'):
                                source_ranges_from_paste.append(
                                    op.source_range
                                )
                affected_ranges.extend(source_ranges_from_paste)

            # Extract operations in this group
            group_ops = []
            other_ops = []

            for idx, op in enumerate(result_ops):
                op_range = op.cell_range.range
                is_affected = any(
                    self._ranges_overlap_str(op_range, r)
                    for r in affected_ranges
                )
                if is_affected:
                    group_ops.append((idx, op))
                else:
                    other_ops.append((idx, op))

            if not group_ops:
                continue

            # Classify operations within THIS group
            source_ops = []
            paste_ops = []
            inverse_ops = []
            target_ops = []

            for idx, op in group_ops:
                op_range = op.cell_range.range

                if isinstance(op, PasteFrom):
                    if any(
                        self._ranges_overlap_str(op_range, tr)
                        for tr in target_ranges
                    ):
                        paste_ops.append((idx, op))
                        if (
                            hasattr(op, 'source_range')
                            and source_range is None
                        ):
                            source_range = op.source_range
                        continue

                if isinstance(op, SetInput) and op.is_inverse:
                    if any(
                        self._ranges_overlap_str(op_range, tr)
                        for tr in target_ranges
                    ):
                        inverse_ops.append((idx, op))
                        continue

                if source_range and self._ranges_overlap_str(
                    op_range, source_range
                ):
                    source_ops.append((idx, op))
                    continue

                if any(
                    self._ranges_overlap_str(op_range, tr)
                    for tr in target_ranges
                ):
                    target_ops.append((idx, op))
                    continue

            # Reorder THIS group
            paste_execution = group.get('paste_execution', 'block')
            paste_full_ordering = group.get('paste_full_ordering', 'grouped')

            reordered_group = []
            reordered_group.extend(
                op for _, op in sorted(source_ops, key=lambda x: x[0])
            )

            if paste_execution == "interleaved":
                paste_list = sorted(paste_ops, key=lambda x: x[0])
                inverse_list = sorted(inverse_ops, key=lambda x: x[0])
                for i in range(max(len(paste_list), len(inverse_list))):
                    if i < len(paste_list):
                        reordered_group.append(paste_list[i][1])
                    if i < len(inverse_list):
                        reordered_group.append(inverse_list[i][1])

            elif paste_full_ordering == "alternating":
                paste_list = sorted(paste_ops, key=lambda x: x[0])
                inverse_list = sorted(inverse_ops, key=lambda x: x[0])
                for i in range(max(len(paste_list), len(inverse_list))):
                    if i < len(paste_list):
                        reordered_group.append(paste_list[i][1])
                    if i < len(inverse_list):
                        reordered_group.append(inverse_list[i][1])

            else:  # grouped (default)
                reordered_group.extend(
                    op for _, op in sorted(paste_ops, key=lambda x: x[0])
                )
                reordered_group.extend(
                    op for _, op in sorted(inverse_ops, key=lambda x: x[0])
                )

            reordered_group.extend(
                op for _, op in sorted(target_ops, key=lambda x: x[0])
            )

            # Rebuild result
            if group_ops and reordered_group:
                first_idx = min(idx for idx, _ in group_ops)
                result_ops = [op for idx, op in other_ops]
                insert_pos = sum(
                    1 for idx, _ in other_ops if idx < first_idx
                )
                result_ops = (
                    result_ops[:insert_pos]
                    + reordered_group
                    + result_ops[insert_pos:]
                )

            self.log(
                context,
                f"paste_full group {source_range or 'unknown'}: "
                f"{len(source_ops)} source, {len(paste_ops)} PASTE, "
                f"{len(inverse_ops)} INVERSE, {len(target_ops)} target"
            )

        return result_ops

    def _ranges_overlap_str(self, range1: str, range2: str) -> bool:
        """Check if two range strings overlap."""
        from openpyxl.utils import range_boundaries

        try:
            min_col1, min_row1, max_col1, max_row1 = range_boundaries(range1)
            min_col2, min_row2, max_col2, max_row2 = range_boundaries(range2)

            return not (
                max_col1 < min_col2 or min_col1 > max_col2 or
                max_row1 < min_row2 or min_row1 > max_row2
            )
        except Exception:
            return False

    def _apply_fingerprint_constraints(
        self,
        operations: List[Operation],
        context: SequencingContext
    ) -> List[Operation]:
        """
        Apply fingerprint-based constraints from context.

        Uses fingerprint_map to resolve operation references and schedule
        operations to satisfy all before-after constraints.
        """
        if not context.constraints:
            return operations

        # Build fingerprint → operation index mapping
        fp_to_idx = {}
        for idx, op in enumerate(operations):
            fp = context.fingerprint_map.get(op)
            if fp:
                fp_to_idx[fp] = idx

        # Build dependency graph
        blocked_by = defaultdict(set)
        blocks = defaultdict(set)

        skipped_constraints = 0
        applied_constraints = 0

        for constraint in context.constraints:
            before_idx = fp_to_idx.get(constraint.before_fingerprint)
            after_idx = fp_to_idx.get(constraint.after_fingerprint)

            if before_idx is None or after_idx is None:
                skipped_constraints += 1
                continue

            if before_idx != after_idx:
                blocked_by[after_idx].add(before_idx)
                blocks[before_idx].add(after_idx)
                applied_constraints += 1

        if applied_constraints == 0:
            self.log(
                context,
                f"No fingerprint constraints to apply "
                f"({skipped_constraints} skipped)"
            )
            return operations

        # Schedule operations using constraint scheduling
        scheduled_ops = self._constraint_schedule(
            operations, blocked_by, blocks, context
        )

        self.log(
            context,
            f"Applied {applied_constraints} constraints, "
            f"{skipped_constraints} skipped"
        )

        return scheduled_ops

    def _constraint_schedule(
        self,
        operations: List[Operation],
        blocked_by: dict,
        blocks: dict,
        context: 'SequencingContext' = None
    ) -> List[Operation]:
        """
        Schedule operations respecting dependencies using greedy algorithm.
        Maintains original ordering as much as possible while satisfying
        constraints. Respects immutable blocks.
        """
        # Build block membership: which indices belong to which block
        idx_to_block = {}
        block_to_indices = {}
        if (
            context
            and hasattr(context, 'immutable_blocks')
            and hasattr(context, 'operation_to_block')
        ):
            for idx, op in enumerate(operations):
                fp = context.fingerprint_map.get(op)
                if fp and fp in context.operation_to_block:
                    block_id = context.operation_to_block[fp]
                    idx_to_block[idx] = block_id
                    if block_id not in block_to_indices:
                        block_to_indices[block_id] = []
                    block_to_indices[block_id].append(idx)

        # For each block, determine when ALL operations in block are ready
        block_ready = {}
        for block_id, indices in block_to_indices.items():
            block_ready[block_id] = all(
                len(blocked_by[idx] - set(indices)) == 0
                for idx in indices
            )

        # Find initially ready operations/blocks
        ready = set()
        scheduled_blocks = set()
        for i in range(len(operations)):
            if i in idx_to_block:
                block_id = idx_to_block[i]
                if (
                    block_id not in scheduled_blocks
                    and block_ready.get(block_id, False)
                ):
                    block_indices = block_to_indices[block_id]
                    ready.add(min(block_indices))
            elif len(blocked_by[i]) == 0:
                ready.add(i)

        result = []
        scheduled = set()

        while ready:
            # Pick earliest ready operation (preserve order)
            next_idx = min(ready)
            ready.remove(next_idx)

            if next_idx in idx_to_block:
                block_id = idx_to_block[next_idx]
                if block_id not in scheduled_blocks:
                    # Schedule ALL operations in this block in order
                    block_indices = sorted(block_to_indices[block_id])
                    for idx in block_indices:
                        result.append(operations[idx])
                        scheduled.add(idx)
                    scheduled_blocks.add(block_id)

                    # Update ready set based on last operation in block
                    last_idx = block_indices[-1]
                    for blocked_idx in blocks[last_idx]:
                        if blocked_idx not in scheduled:
                            blocked_by[blocked_idx].discard(last_idx)
                            for idx in block_indices:
                                blocked_by[blocked_idx].discard(idx)

                            if blocked_idx in idx_to_block:
                                check_block_id = idx_to_block[blocked_idx]
                                if check_block_id not in scheduled_blocks:
                                    check_indices = block_to_indices[
                                        check_block_id
                                    ]
                                    if all(
                                        len(
                                            blocked_by[i]
                                            - set(check_indices)
                                        ) == 0
                                        for i in check_indices
                                    ):
                                        ready.add(min(check_indices))
                            elif len(blocked_by[blocked_idx]) == 0:
                                ready.add(blocked_idx)
            else:
                # Regular operation - schedule it
                result.append(operations[next_idx])
                scheduled.add(next_idx)

                for blocked_idx in blocks[next_idx]:
                    if blocked_idx not in scheduled:
                        blocked_by[blocked_idx].discard(next_idx)

                        if blocked_idx in idx_to_block:
                            check_block_id = idx_to_block[blocked_idx]
                            if check_block_id not in scheduled_blocks:
                                check_indices = block_to_indices[
                                    check_block_id
                                ]
                                if all(
                                    len(
                                        blocked_by[i] - set(check_indices)
                                    ) == 0
                                    for i in check_indices
                                ):
                                    ready.add(min(check_indices))
                        elif len(blocked_by[blocked_idx]) == 0:
                            ready.add(blocked_idx)

        # Check for cyclic dependencies
        if len(result) < len(operations):
            return operations

        return result

    def can_skip(self, context: SequencingContext) -> bool:
        """Skip if no constraints and no paste groups."""
        has_paste = (
            hasattr(context, 'paste_full_target_regions')
            and context.paste_full_target_regions
        )
        return not context.constraints and not has_paste
