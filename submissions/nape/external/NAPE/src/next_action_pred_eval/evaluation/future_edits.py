"""
Future Edits Manager Module (v3 - simulation_order_fix)
Handles ground truth updates when predictions are accepted in online evaluation.

Full port of the v3 algorithm from the reference implementation including:
- Fingerprint-based operation tracking with side-specific border precision
- PasteFrom destination expansion with paste mode awareness
- MergeCells handling (duplicate detection, conflict resolution, unmerge correction)
- Merge-adjusted state building (step 2.5)
- Two-pass locking algorithm with hierarchical last-writer tracking
- Two-pass simulation validation
- Smart operation placement (fingerprint, merge-range, range-based fallback)
- Operation synthesis from state differences
- All metadata tracking
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.cell_range import CellRange
from next_action_pred_eval.core.operations import (
    MergeCells,
    SetAlignment,
    SetBorder,
    SetFillColor,
    SetFontProperty,
    SetFormula,
    SetInput,
    SetNumberFormat,
    SetTextOrientation,
    SetValue,
    SetWrapText,
    PasteFrom,
    get_cells_in_range,
)
from next_action_pred_eval.core.state import StateBuilder
from next_action_pred_eval.evaluation.evaluator import EvaluationResult
from next_action_pred_eval.evaluation.state_comparator import (
    StateComparator,
    ComparisonResult,
    PropertyDifference,
)
from next_action_pred_eval.utils.cell_utils import (
    expand_range,
    get_cell_address,
    get_range_string,
)
from next_action_pred_eval.utils.workbook.excel_parser import ExcelParser
from next_action_pred_eval.core.operation_merger import RectangleMerger, BorderMerger
from next_action_pred_eval.generation.sequencing.transformers import OperationSequencer
from next_action_pred_eval.generation.sequencing.base import SequencingContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: set a nested dict value using a dot-separated path
# (ported from the source StateComparator._set_nested_value)
# ---------------------------------------------------------------------------

def _set_nested_value(obj: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set a nested dictionary value using a dot-separated path.

    Examples:
        - "value" -> obj["value"] = value
        - "Format.font.bold" -> obj["Format"]["font"]["bold"] = value
        - "Format.borders.left.lineStyle" -> obj["Format"]["borders"]["left"]["lineStyle"] = value
        - "Format.fill" with value {"fgColor": "#FF0000"} -> obj["Format"]["fill"]["fgColor"] = "#FF0000"

    Args:
        obj: Dictionary to set value in
        path: Dot-separated path (e.g., "Format.font.bold")
        value: Value to set (can be a dict for composite properties)
    """
    parts = path.split(".")

    # Navigate to the parent of the final key, creating dicts as needed
    current = obj
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        elif not isinstance(current[part], dict):
            logger.warning("Cannot set nested value: %s in path %s is not a dict", part, path)
            return
        current = current[part]

    final_key = parts[-1]

    # If value is a dict (composite property), merge into the nested structure
    if isinstance(value, dict):
        if final_key not in current:
            current[final_key] = {}
        elif not isinstance(current[final_key], dict):
            current[final_key] = {}
        for sub_key, sub_value in value.items():
            current[final_key][sub_key] = sub_value
    else:
        # Set the final value directly
        current[final_key] = value


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class FutureEditsConfig:
    """Configuration for the future edits manager."""

    max_inverse_ops: Optional[int] = 250
    """Safety cap for inverse operations appended after accepted predictions."""

    track_metadata: bool = True
    """When True the manager emits detailed metadata for debugging/visualization."""

    max_fingerprint_cells: int = 1000
    """Maximum number of (cell, property) pairs to track in fingerprint.
    If prediction touches more cells, will log a warning but still proceed."""

    min_replacement_ops_to_keep_original: int = 3
    """If synthesizing a replacement for a GT op requires >= this many ops, keep the original GT op."""


@dataclass
class FutureEditChanges:
    """Audit information describing how the ground truth was modified."""

    old_length: int
    new_length: int
    original_region: List[Operation]
    operations_added: List[Operation]
    operations_removed: List[Operation]
    inverse_ops_added: List[Operation]
    removed_from_after_region: List[Operation]
    dedup_window_range: Tuple[int, int]
    metadata: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> Dict[str, int]:
        final_state_gain = len(self.original_region)
        dedup_gain = len(self.operations_removed)
        inverse_cost = len(self.inverse_ops_added)
        # TRUE net_gain: actual sequence length reduction
        # Positive = sequence got shorter (good), Negative = sequence got longer (bad)
        net_gain = self.old_length - self.new_length
        return {
            "final_state_gain": final_state_gain,
            "dedup_gain": dedup_gain,
            "inverse_cost": inverse_cost,
            "net_gain": net_gain,
            "operations_added": len(self.operations_added),
            "old_length": self.old_length,
            "new_length": self.new_length,
        }

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["summary"] = self.summary()
        return data


# ============================================================================
# FutureEditsManager  (v3 - simulation order fix)
# ============================================================================


class FutureEditsManager:
    """Applies minimal edits to the ground truth after an acceptance."""

    def __init__(
        self,
        config: Optional[FutureEditsConfig] = None,
        region_metadata: Optional[Dict] = None,
    ):
        self.config = config or FutureEditsConfig()
        self.region_metadata = region_metadata
        self._state_comparator = StateComparator(ignore_defaults=True)
        self._state_builder = StateBuilder()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def apply_future_edits(
        self,
        current_gt: List[Operation],
        start_idx: int,
        end_idx: int,  # DEPRECATED: ignored, kept for backward compatibility
        predicted_ops: List[Operation],
        eval_result: EvaluationResult,
        initial_state: Optional[Dict[str, Any]] = None,
        final_target_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Operation], Dict[str, object]]:
        """Apply minimal edits to ground truth after accepting a prediction.

        Uses state-based fingerprint algorithm for intelligent replacement tracking.

        Semantics (pop-first approach):
        - User has already executed the action at (start_idx - 1), now in history
        - start_idx points to the first action in the remaining future
        - initial_state is the state AFTER the user action (before prediction)
        - history_before = GT[0:start_idx] includes the user's trigger action
        - future_before = GT[start_idx:] is the remaining sequence to be processed

        The algorithm processes ALL ops from start_idx onwards, checking each one
        against the prediction's fingerprint. Ops that overlap are processed
        (synthesized from FNs or removed if fully covered). Ops that don't overlap
        are kept as-is. No ops are blindly deleted based on region span.

        Algorithm:
        1. Build fingerprint {(cell, property)} from prediction operations
        2. Build prediction_state by applying predicted_ops to initial_state
        3. Build merge-adjusted state (apply GT merge/unmerge ops to prediction_state)
        4. Compare adjusted_state vs final_target_state to get ALL FNs and FPs
        5. FIRST PASS:
           a. Generate inverse_ops from FPs (before building future_after)
           b. For FP merged_cells: generate UNMERGE ops
           c. Generate unmerge_correction_ops by comparing (prediction_state + unmerge_ops)
              vs initial_state - this restores data LOST by the prediction's FP merge
        6. Process ALL future_before ops (GT[start_idx:]) sequentially:
           - For affected ops (overlap with fingerprint): synthesize from FNs or remove
           - For unaffected ops: keep as-is (locked)
        7. SECOND PASS: Simulate full sequence to catch remaining discrepancies:
           - Simulate: merge_unmerge_ops + merge_unmerge_correction_ops + inverse_ops + future_after
           - This matches final execution order
           - Generate remaining_inverse_ops and missing_ops from FPs/FNs
        8. Place remaining ops at correct positions in future_after
        9. Build final sequence:
           - final_future = merge_unmerge_ops + correction_ops + inverse_ops + future_after
           - new_gt = history_before + predicted_ops + final_future

        Args:
            current_gt: Current ground truth operations list
            start_idx: Index from which to start processing future ops
            end_idx: DEPRECATED - ignored, kept for backward compatibility
            predicted_ops: Operations predicted by the model
            eval_result: Evaluation result containing state comparisons
            initial_state: State after user action, before prediction (REQUIRED)
            final_target_state: Final target state computed at run start (REQUIRED)

        Returns:
            Tuple of (new_gt, changes_dict)

        Raises:
            ValueError: If initial_state or final_target_state is not provided
        """
        if initial_state is None or final_target_state is None:
            raise ValueError(
                "apply_future_edits requires both initial_state and final_target_state. "
                "These must be provided by the orchestrator."
            )
        logger.debug("Using state-based future edits algorithm (v3 - simulation order fix)")

        # Step 0: Build set of predicted merge operations to avoid duplicates
        # When a prediction includes a MERGE, we should remove that same MERGE from future
        predicted_merge_ranges: Set[Tuple[str, str, bool]] = set()  # (sheet, range, is_merge)
        for pred_op in predicted_ops:
            if type(pred_op).__name__ == "MergeCells":
                predicted_merge_ranges.add((
                    pred_op.cell_range.sheet,
                    pred_op.cell_range.range,
                    pred_op.value,  # True=merge, False=unmerge
                ))

        # Step 1: Build fingerprint from predictions
        fingerprint = self._build_fingerprint(predicted_ops)

        # Log warning if fingerprint is large (may be slow)
        if len(fingerprint) > self.config.max_fingerprint_cells:
            logger.warning(
                "Large fingerprint (%d > %d threshold), processing may be slow",
                len(fingerprint), self.config.max_fingerprint_cells
            )

        # Step 2: Build prediction_state (state after applying predictions to initial_state)
        prediction_state = self._state_builder.build_state(
            initial_state=deepcopy(initial_state),
            operations=predicted_ops,
        )

        # Step 2.5: Build merge-adjusted state for accurate FN comparison
        # When MERGE operations exist in future_before, they can clear borders that
        # the prediction set. To make accurate coverage decisions, we need to compute
        # FN diffs against a state that reflects these future merges.
        #
        # Example: Prediction sets BORDER_OUTSIDE J56, GT has BORDER_ALL J56:N56.
        # If GT also has MERGE J56:N56 AFTER the border op, the merge will clear
        # J56's borders except left edge. Without merge-adjustment, we'd think
        # BORDER_ALL K56:N56 covers the effect, but J56's borders get lost.

        # Extract GT merge/unmerge ops from future_before
        future_before_preview = current_gt[start_idx:]  # Preview for merge extraction
        gt_future_merge_ops = [
            op for op in future_before_preview
            if type(op).__name__ == "MergeCells"
        ]

        # Build merge-adjusted state by applying GT merges to prediction_state
        if gt_future_merge_ops:
            logger.debug(
                "Found %d GT merge ops in future - building merge-adjusted state",
                len(gt_future_merge_ops)
            )
            merge_adjusted_state = deepcopy(prediction_state)

            # For each GT merge, check if there's a conflicting prediction merge
            # (same cells but different range). If so, unmerge prediction's merge first.
            for gt_merge_op in gt_future_merge_ops:
                if not gt_merge_op.value:  # Skip UNMERGE ops for now
                    continue

                gt_range = (gt_merge_op.cell_range.sheet, gt_merge_op.cell_range.range)

                # Check for conflicting prediction merges (overlapping but not identical)
                for pred_merge in predicted_merge_ranges:
                    pred_sheet, pred_range, pred_is_merge = pred_merge
                    if not pred_is_merge:  # Skip prediction UNMERGEs
                        continue
                    if pred_sheet != gt_range[0]:  # Different sheets
                        continue
                    if pred_range == gt_range[1]:  # Exact match - no conflict
                        continue

                    # Check if ranges overlap
                    pred_cr = CellRange.from_string(f"{pred_sheet}!{pred_range}")
                    gt_cr = CellRange.from_string(f"{gt_range[0]}!{gt_range[1]}")

                    if self._ranges_overlap(pred_cr, gt_cr):
                        # Conflicting merge - unmerge prediction's merge first
                        logger.debug(
                            "Conflicting merge detected: pred=%s!%s vs gt=%s!%s - unmerging pred first",
                            pred_sheet, pred_range, gt_range[0], gt_range[1]
                        )
                        try:
                            unmerge_op = MergeCells(
                                cell_range=pred_cr,
                                value=False,  # UNMERGE
                            )
                            unmerge_op.apply_to_state(merge_adjusted_state)
                        except Exception as e:
                            logger.warning("Failed to apply unmerge for conflicting merge: %s", e)

                # Now apply the GT merge
                try:
                    gt_merge_op.apply_to_state(merge_adjusted_state)
                except Exception as e:
                    logger.warning("Failed to apply GT merge to adjusted state: %s", e)

            # Also apply GT UNMERGE ops
            for gt_merge_op in gt_future_merge_ops:
                if gt_merge_op.value:  # Skip MERGE ops (already applied)
                    continue
                try:
                    gt_merge_op.apply_to_state(merge_adjusted_state)
                except Exception as e:
                    logger.warning("Failed to apply GT unmerge to adjusted state: %s", e)
        else:
            # No GT merge ops - use prediction_state directly
            merge_adjusted_state = prediction_state

        # Step 3: Compare merge_adjusted_state vs final_target_state to get ALL differences
        # Using merge_adjusted_state ensures accurate FN detection that accounts for
        # how GT merges will affect the prediction's changes
        final_comparison = self._state_comparator.compare(
            predicted_state=merge_adjusted_state,
            true_state=final_target_state,
            skip_ops_diff=True,
        )

        # Get all FN differences - things merge-adjusted state is missing to reach final state
        all_fn_diffs = [d for d in final_comparison.differences if d.match_type in ("FN", "MISMATCH")]
        # Build a set of FN fingerprints for tracking: (sheet, cell, property_path)
        all_fn_fingerprints = {
            (d.sheet, d.cell, d.property_path) for d in all_fn_diffs
        }

        # Get all FP differences - things in prediction state that shouldn't be there
        # These are used to generate inverse ops in the FIRST PASS (before building future_after)
        all_fp_diffs = [d for d in final_comparison.differences if d.match_type in ("FP", "MISMATCH")]

        # FIRST PASS: Generate inverse ops from FPs detected in adjusted_state
        # This happens BEFORE processing future ops, ensuring inverse ops are available
        # for the second pass simulation which matches final execution order
        inverse_ops = self._extract_inverse_ops_from_comparison(final_comparison)

        logger.debug(
            "First pass (adjusted_state comparison): %d FPs -> %d inverse ops, %d FNs",
            len(all_fp_diffs), len(inverse_ops), len(all_fn_diffs)
        )

        # Step 5: Process future_before ops
        # history_before = ops before the current position (already executed)
        # future_before = ALL remaining ops from start_idx onwards that need to be processed
        # NOTE: We do NOT blindly delete any ops - each future op is processed individually
        # based on whether it overlaps with the prediction's fingerprint
        history_before = current_gt[:start_idx]
        future_before = current_gt[start_idx:]  # ALL remaining ops from trigger point

        # Step 5.0: Build locked ops set using TWO-PASS approach
        # Pass 1: Identify which ops WOULD be removed (last writers with no FN diffs + overlap)
        # Pass 2: Only lock ops whose last writer will be KEPT
        # This ensures if PASTE_FROM@110 is removed, INPUT@5 (shadowed by it) can also be removed.

        # First, build last_writer map
        last_writer: Dict[Tuple[str, str, str], int] = {}  # (sheet, cell, prop) -> last index
        for idx, op in enumerate(future_before):
            op_fp = self._build_fingerprint([op])
            for fp_tuple in op_fp:
                last_writer[fp_tuple] = idx

        # Secondary index: group properties by (sheet, cell) for fast hierarchical lookup
        cell_writers: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
        for (sheet, cell, prop), idx in last_writer.items():
            key = (sheet, cell)
            if key not in cell_writers:
                cell_writers[key] = []
            cell_writers[key].append((prop, idx))

        # Helper function to find the last writer for a fingerprint tuple, considering hierarchy
        # E.g., if we're checking (J40, Format.wrapText), we should also check if there's a
        # later writer for (J40, Format) since PASTE_FROM|formats writes to "Format" and
        # would overwrite all child properties like Format.wrapText
        def find_last_writer_hierarchical(fp_tuple: Tuple[str, str, str]) -> Optional[int]:
            sheet, cell, prop = fp_tuple
            best_idx = last_writer.get(fp_tuple)  # Exact match

            # Check for parent property writers (e.g., "Format" overwrites "Format.wrapText")
            for lw_prop, lw_idx in cell_writers.get((sheet, cell), []):
                # Check if lw_prop is a parent of prop (would overwrite it)
                if prop.startswith(lw_prop + "."):
                    # lw_prop is a parent (e.g., "Format" is parent of "Format.wrapText")
                    if best_idx is None or lw_idx > best_idx:
                        best_idx = lw_idx

            return best_idx

        # Pass 1: Identify which ops would be removed if not locked
        # An op is a candidate for removal if:
        #   - It overlaps with prediction fingerprint
        #   - It has no FN diffs (prediction covers it)
        #   - It's a last writer for at least one of its fingerprints
        ops_that_would_be_removed: Set[int] = set()
        for idx, op in enumerate(future_before):
            op_name = type(op).__name__
            if op_name == "MergeCells":
                continue  # MergeCells handled separately

            op_fp = self._build_fingerprint([op])
            is_affected = self._fingerprints_overlap(fingerprint, op_fp)
            if not is_affected:
                continue  # Not affected, won't be removed

            # Check for FN diffs
            op_cells = {(sheet, cell) for sheet, cell, _ in op_fp}
            op_property_types = {prop_type for _, _, prop_type in op_fp}
            fn_diffs = [
                d for d in all_fn_diffs
                if (d.sheet, d.cell) in op_cells
                and self._property_matches(d.property_path, op_property_types)
            ]

            if not fn_diffs:
                # This op would be removed (if not locked)
                # But only if it's a last writer for at least one property
                # Use hierarchical matching to check if this op is the last writer
                is_last_writer_for_any = any(
                    find_last_writer_hierarchical(fp_tuple) == idx
                    for fp_tuple in op_fp
                )
                if is_last_writer_for_any:
                    ops_that_would_be_removed.add(idx)

        # Pass 2: Only lock an op if AT LEAST ONE of its last writers will be KEPT
        # If ALL last writers that shadow this op are being removed, don't lock it
        locked_op_indices: Set[int] = set()
        for idx, op in enumerate(future_before):
            op_fp = self._build_fingerprint([op])
            # Use hierarchical matching to check if this op is the last writer
            is_last_writer_for_any = any(
                find_last_writer_hierarchical(fp_tuple) == idx
                for fp_tuple in op_fp
            )

            if is_last_writer_for_any:
                # This op is a last writer for something, don't lock it
                continue

            # This op is NOT a last writer - check if any of its last writers will be KEPT
            # For each fingerprint tuple, check if the last writer will be removed
            # Use hierarchical matching to find parent property writers
            any_last_writer_kept = False
            for fp_tuple in op_fp:
                last_writer_idx = find_last_writer_hierarchical(fp_tuple)
                if last_writer_idx is not None and last_writer_idx not in ops_that_would_be_removed:
                    # The last writer for this property will be KEPT
                    any_last_writer_kept = True
                    break

            if any_last_writer_kept:
                # Lock this op - it will be overwritten by a kept last writer
                locked_op_indices.add(idx)
            # else: all last writers are being removed, so this op can also be removed

        if locked_op_indices:
            logger.debug(
                "Locked %d ops that will be overwritten by later ops",
                len(locked_op_indices)
            )

        future_after: List[Operation] = []
        addressed_fn_fingerprints: Set[Tuple[str, str, str]] = set()
        replacement_details: List[Dict] = []
        removed_future_ops: List[Operation] = []  # Track ops removed from future

        # Step 5a: Handle FP merged_cells specially - add UNMERGE inverse ops
        # This must happen BEFORE processing future ops, as UNMERGE ops need to be
        # placed at the start of future_after
        fp_merge_diffs = [
            d for d in final_comparison.differences
            if d.match_type in ("FP", "MISMATCH") and d.property_path == "merged_cells"
        ]
        merge_unmerge_ops: List[Operation] = []
        merge_unmerge_correction_ops: List[Operation] = []

        if fp_merge_diffs:
            logger.debug("Found %d FP merged_cells - generating UNMERGE ops", len(fp_merge_diffs))
            merge_unmerge_ops = self._build_unmerge_ops_for_fp_merges(fp_merge_diffs)

            # After UNMERGE, check for data that was LOST by the prediction's merge.
            # We compare (prediction_state + unmerge) vs initial_state to find what was
            # in initial_state but is now missing due to the prediction's merge clearing it.
            merge_unmerge_correction_ops = self._generate_unmerge_correction_ops(
                fp_merge_diffs,
                merge_unmerge_ops,
                prediction_state,
                initial_state,
            )

            replacement_details.append({
                "action": "merge_fp_handled",
                "unmerge_count": len(merge_unmerge_ops),
                "correction_count": len(merge_unmerge_correction_ops),
                "reason": "FP merged_cells require UNMERGE inverse + value corrections",
            })

        for idx, gt_op in enumerate(future_before):
            # Special handling for MergeCells operations - skip generic processing
            # MergeCells ops should pass through intact; FP merges are handled via
            # UNMERGE inverse ops above, and GT merge ops remain unchanged
            # HOWEVER: If the exact same merge was already predicted, skip it to avoid duplicates
            op_name = type(gt_op).__name__
            if op_name == "MergeCells":
                # Check if this exact merge was already in the prediction
                gt_merge_key = (
                    gt_op.cell_range.sheet,
                    gt_op.cell_range.range,
                    gt_op.value,
                )
                if gt_merge_key in predicted_merge_ranges:
                    # This merge was already predicted - remove from future to avoid duplicate
                    removed_future_ops.append(gt_op)
                    replacement_details.append({
                        "original_op": gt_op.to_symbolic(),
                        "action": "removed",
                        "reason": "merge_already_in_prediction",
                    })
                    continue

                # Keep the merge in future
                future_after.append(gt_op)
                replacement_details.append({
                    "original_op": gt_op.to_symbolic(),
                    "action": "kept_intact",
                    "reason": "merge_special_handling",
                })
                continue

            # Check if this op is affected by the fingerprint
            op_fingerprint = self._build_fingerprint([gt_op])
            is_affected = self._fingerprints_overlap(fingerprint, op_fingerprint)

            if not is_affected:
                # Unaffected op - keep as-is
                future_after.append(gt_op)
                continue

            # This op is affected - need to analyze it
            logger.debug("GT op %d affected by fingerprint: %s", idx, gt_op)

            # Get cells and property types this GT op touches
            op_cells = {(sheet, cell) for sheet, cell, _ in op_fingerprint}
            op_property_types = {prop_type for _, _, prop_type in op_fingerprint}

            # Filter all_fn_diffs to only include FNs for cells this op touches
            # AND that match the property types this op affects
            # This ensures a BORDER op only considers border FNs, not value/font FNs
            fn_diffs = [
                d for d in all_fn_diffs
                if (d.sheet, d.cell) in op_cells
                and self._property_matches(d.property_path, op_property_types)
            ]

            if not fn_diffs:
                # Check if this op is LOCKED (will be overwritten by later ops)
                # Locked ops cannot be removed - they must stay in future
                if idx in locked_op_indices:
                    future_after.append(gt_op)
                    replacement_details.append({
                        "original_op": gt_op.to_symbolic(),
                        "action": "kept_locked",
                        "reason": "locked_op_will_be_overwritten_by_later_ops",
                    })
                    continue

                # Prediction already covers what this GT op would do - skip it
                removed_future_ops.append(gt_op)  # Track removed op
                replacement_details.append({
                    "original_op": gt_op.to_symbolic(),
                    "action": "removed",
                    "reason": "prediction_covers_effect",
                })
                continue

            # Synthesize replacement operations from FN differences
            replacement_ops = self._synthesize_operations_from_differences(fn_diffs, final_target_state)

            # Check if replacement is too complex
            if len(replacement_ops) >= self.config.min_replacement_ops_to_keep_original:
                logger.debug(
                    "Keeping original op: %d replacement ops >= threshold %d",
                    len(replacement_ops), self.config.min_replacement_ops_to_keep_original
                )
                future_after.append(gt_op)
                # Track which FNs we've addressed (by keeping the original op that handles them)
                for diff in fn_diffs:
                    addressed_fn_fingerprints.add((diff.sheet, diff.cell, diff.property_path))
                replacement_details.append({
                    "original_op": gt_op.to_symbolic(),
                    "action": "kept_original",
                    "reason": f"replacement_too_complex ({len(replacement_ops)} ops)",
                    "replacement_count": len(replacement_ops),
                })
            else:
                future_after.extend(replacement_ops)
                # Track which FNs we've addressed
                for diff in fn_diffs:
                    addressed_fn_fingerprints.add((diff.sheet, diff.cell, diff.property_path))
                replacement_details.append({
                    "original_op": gt_op.to_symbolic(),
                    "action": "replaced",
                    "replacement_ops": [op.to_symbolic() for op in replacement_ops],
                    "replacement_count": len(replacement_ops),
                })

        # Step 6: Second pass - verify with same execution order as final sequence
        # CRITICAL: We simulate `inverse_ops + future_after` which MATCHES the final execution order.
        # This catches any discrepancies that would occur at runtime (e.g., if a GT op in
        # future_after re-creates an FP that the inverse op was supposed to fix).
        #
        # Key insight: The first pass (step 3) generated inverse_ops from comparing adjusted_state
        # to final_target. Now we verify that inverse_ops + future_after actually reaches the target.

        remaining_inverse_ops: List[Operation] = []
        missing_ops: List[Operation] = []

        try:
            # Simulate the full final sequence: prediction_state + merge_unmerge + correction + inverse + future_after
            # This matches final execution order: predicted_ops + merge_unmerge_ops + merge_unmerge_correction_ops + inverse_ops + future_after
            second_pass_ops = merge_unmerge_ops + merge_unmerge_correction_ops + inverse_ops + future_after
            simulated_state = self._state_builder.build_state(
                initial_state=deepcopy(prediction_state),  # Start from prediction_state
                operations=second_pass_ops,
            )

            second_pass_comparison = self._state_comparator.compare(
                predicted_state=simulated_state,
                true_state=final_target_state,
                skip_ops_diff=True,
            )

            # Extract remaining inverse ops from second pass FPs
            remaining_inverse_ops = self._extract_inverse_ops_from_comparison(second_pass_comparison)

            # Synthesize ops for remaining FNs and INCORRECTs (mismatches)
            # FN = missing property, INCORRECT = wrong value
            remaining_fn_diffs = [
                d for d in second_pass_comparison.differences
                if d.match_type in ("FN", "MISMATCH")
            ]
            if remaining_fn_diffs:
                missing_ops = self._synthesize_operations_from_differences(
                    remaining_fn_diffs, final_target_state
                )

            logger.debug(
                "Second pass simulation (merge_unmerge + correction + inverse + future_after): %d remaining inverse ops, %d missing ops",
                len(remaining_inverse_ops), len(missing_ops)
            )
        except Exception as e:
            logger.warning("Second pass simulation failed: %s", e)

        # Step 7: Place remaining ops into future_after at correct positions
        # This ensures proper execution order (e.g., inverse ops run AFTER the ops that caused FPs)
        # Note: We don't filter out MergeCells - UNMERGE ops from second pass are needed,
        # and duplicate MERGEs are harmless (idempotent)
        if remaining_inverse_ops or missing_ops:
            future_after = self._place_remaining_ops_in_future(
                future_after,
                remaining_inverse_ops + missing_ops,
            )

            replacement_details.append({
                "action": "second_pass_corrections",
                "remaining_inverse_count": len(remaining_inverse_ops),
                "missing_ops_count": len(missing_ops),
                "reason": "Two-pass simulation caught additional FPs/FNs",
            })

        # Step 8: Build final sequences
        # history_after = history_before + predicted_ops (handled by orchestrator)
        # future_after includes:
        #   - merge_unmerge_ops: UNMERGE ops for FP merges (at start)
        #   - merge_unmerge_correction_ops: Corrections for data lost in unmerge
        #   - inverse_ops: Inverse ops from first pass (clean up prediction FPs)
        #   - future_after: Processed GT ops with remaining corrections inserted at correct positions
        final_future = (
            merge_unmerge_ops
            + merge_unmerge_correction_ops
            + inverse_ops
            + future_after
        )

        # Build the complete new GT sequence
        new_gt = history_before + predicted_ops + final_future

        # Build changes for auditing
        # NOTE: original_region is empty since we don't blindly delete ops anymore
        # All ops are processed individually based on fingerprint overlap
        changes = FutureEditChanges(
            old_length=len(current_gt),
            new_length=len(new_gt),
            original_region=[],  # No blind deletion - each op is processed individually
            operations_added=predicted_ops + missing_ops + merge_unmerge_correction_ops,
            operations_removed=removed_future_ops,  # Ops removed from future_before
            inverse_ops_added=inverse_ops + remaining_inverse_ops + merge_unmerge_ops,
            removed_from_after_region=removed_future_ops,  # Alias for clarity
            dedup_window_range=(start_idx, len(current_gt)),
            metadata={
                "algorithm": "v3_simulation_order_fix",
                "fingerprint_size": len(fingerprint),
                "all_fn_count": len(all_fn_diffs),
                "all_fp_count": len(all_fp_diffs),
                "addressed_fn_count": len(addressed_fn_fingerprints),
                "first_pass_inverse_count": len(inverse_ops),
                "second_pass_inverse_count": len(remaining_inverse_ops),
                "missing_ops_count": len(missing_ops),
                "future_after_count": len(future_after),
                "inverse_ops_count": len(inverse_ops) + len(remaining_inverse_ops),
                "merge_unmerge_ops_count": len(merge_unmerge_ops),
                "merge_unmerge_correction_ops_count": len(merge_unmerge_correction_ops),
                "replacement_details": replacement_details,
                "future_before_count": len(future_before),
                "final_future_count": len(final_future),
                "gt_future_merge_ops_count": len(gt_future_merge_ops),
                "merge_adjusted": len(gt_future_merge_ops) > 0,
            },
        )

        if self.config.track_metadata:
            changes.metadata.update({
                "predicted_count": len(predicted_ops),
                "inverse_count": len(inverse_ops) + len(remaining_inverse_ops),
            })

        logger.debug(
            "FutureEditsManager: predicted=%d, inverse=%d, missing=%d",
            len(predicted_ops), len(inverse_ops) + len(remaining_inverse_ops),
            len(missing_ops),
        )

        return new_gt, changes.to_dict()

    # ------------------------------------------------------------------
    # Fingerprint building
    # ------------------------------------------------------------------

    def _build_fingerprint(self, operations: List[Operation]) -> Set[Tuple[str, str, str]]:
        """
        Build a fingerprint of (sheet, cell, property) tuples touched by operations.

        For most operations, this is simply the cells in the operation's range.
        Special handling:
        - PasteFrom: calculates actual affected cells based on paste mode and source dimensions
        - SetBorder: only includes cells that are actually affected based on the border side
          (e.g., BORDER_TOP only affects the top row of the range)
        - MergeCells: affects all cells in the range for the merged_cells property

        Args:
            operations: List of operations

        Returns:
            Set of (sheet, cell, property_type) tuples
        """
        fingerprint: Set[Tuple[str, str, str]] = set()

        for op in operations:
            op_name = type(op).__name__
            sheet = op.cell_range.sheet or "Sheet1"
            property_type = self._get_property_type(op)

            # Special handling for PasteFrom - calculate actual affected cells
            if op_name == "PasteFrom":
                source_range_str = getattr(op, 'source_range', '')
                if not source_range_str:
                    # Fallback: try the value field for source_range
                    # In some implementations source_range is stored differently
                    source_range_str = ''
                if source_range_str:
                    # Parse source range
                    source_cell_range = CellRange.from_string(source_range_str)
                    src_rows, src_cols = source_cell_range.get_dimensions()
                    dst_rows, dst_cols = op.cell_range.get_dimensions()
                    dst_start_row, dst_start_col, _, _ = op.cell_range.get_coordinates()

                    paste_mode = (op.value or 'all').lower() if isinstance(op.value, str) else 'all'
                    is_single_cell_dest = (dst_rows == 1 and dst_cols == 1)

                    # Determine iteration dimensions based on paste mode (matches apply_to_state)
                    if paste_mode == 'formats' and not is_single_cell_dest:
                        # Format-only paste with multi-cell destination: tile source pattern
                        iterate_rows, iterate_cols = dst_rows, dst_cols
                    else:
                        # All other cases: use source dimensions
                        iterate_rows, iterate_cols = src_rows, src_cols

                    # Add actual destination cells (WRITES only)
                    # Note: Source cells are READ, not written - don't include them
                    # in fingerprint since we use fingerprint to track "last writer"
                    for r in range(iterate_rows):
                        for c in range(iterate_cols):
                            cell_addr = get_cell_address(dst_start_row + r, dst_start_col + c)
                            if isinstance(property_type, list):
                                for pt in property_type:
                                    fingerprint.add((sheet, cell_addr, pt))
                            else:
                                fingerprint.add((sheet, cell_addr, property_type))

                    continue  # Skip the generic processing below

            # Special handling for AutoFill - calculate actual filled cells (excludes source)
            if op_name == "AutoFill":
                try:
                    fill_range = op._get_fill_range()
                    fill_cells = list(expand_range(fill_range.range))
                    for row, col in fill_cells:
                        cell_addr = get_cell_address(row, col)
                        if isinstance(property_type, list):
                            for pt in property_type:
                                fingerprint.add((sheet, cell_addr, pt))
                        else:
                            fingerprint.add((sheet, cell_addr, property_type))
                except Exception:
                    # Fallback to generic handling if geometry fails
                    pass
                else:
                    continue  # Skip the generic processing below

            # Special handling for SetBorder - only include cells that are actually affected
            # based on the border side (matches apply_to_state behavior)
            # For ranged operations like BORDER_OUTSIDE, only the edge cells get specific borders
            if op_name == "SetBorder":
                side = getattr(op, 'side', '').lower()
                start_row, start_col, end_row, end_col = op.cell_range.get_coordinates()
                all_cells = list(expand_range(op.cell_range.range))

                for row, col in all_cells:
                    cell_addr = get_cell_address(row, col)

                    if side == "all":
                        # All 4 borders for every cell
                        for border_side in ["left", "right", "top", "bottom"]:
                            fingerprint.add((sheet, cell_addr, f"Format.borders.{border_side}"))

                    elif side == "outside":
                        # Only cells on the perimeter get their outward-facing borders
                        if row == start_row:  # Top edge
                            fingerprint.add((sheet, cell_addr, "Format.borders.top"))
                        if row == end_row:  # Bottom edge
                            fingerprint.add((sheet, cell_addr, "Format.borders.bottom"))
                        if col == start_col:  # Left edge
                            fingerprint.add((sheet, cell_addr, "Format.borders.left"))
                        if col == end_col:  # Right edge
                            fingerprint.add((sheet, cell_addr, "Format.borders.right"))

                    elif side == "left":
                        # Only leftmost column gets left border
                        if col == start_col:
                            fingerprint.add((sheet, cell_addr, "Format.borders.left"))

                    elif side == "right":
                        # Only rightmost column gets right border
                        if col == end_col:
                            fingerprint.add((sheet, cell_addr, "Format.borders.right"))

                    elif side == "top":
                        # Only topmost row gets top border
                        if row == start_row:
                            fingerprint.add((sheet, cell_addr, "Format.borders.top"))

                    elif side == "bottom":
                        # Only bottommost row gets bottom border
                        if row == end_row:
                            fingerprint.add((sheet, cell_addr, "Format.borders.bottom"))

                    elif side == "inside_horizontal":
                        # Interior horizontal borders (bottom of each row except last)
                        if row < end_row:
                            fingerprint.add((sheet, cell_addr, "Format.borders.bottom"))
                        # And top of each row except first
                        if row > start_row:
                            fingerprint.add((sheet, cell_addr, "Format.borders.top"))

                    elif side == "inside_vertical":
                        # Interior vertical borders (right of each col except last)
                        if col < end_col:
                            fingerprint.add((sheet, cell_addr, "Format.borders.right"))
                        # And left of each col except first
                        if col > start_col:
                            fingerprint.add((sheet, cell_addr, "Format.borders.left"))

                    elif side == "inside":
                        # Both inside_horizontal and inside_vertical
                        if row < end_row:
                            fingerprint.add((sheet, cell_addr, "Format.borders.bottom"))
                        if row > start_row:
                            fingerprint.add((sheet, cell_addr, "Format.borders.top"))
                        if col < end_col:
                            fingerprint.add((sheet, cell_addr, "Format.borders.right"))
                        if col > start_col:
                            fingerprint.add((sheet, cell_addr, "Format.borders.left"))

                    else:
                        # Unknown side, include all borders for all cells
                        for border_side in ["left", "right", "top", "bottom"]:
                            fingerprint.add((sheet, cell_addr, f"Format.borders.{border_side}"))

                continue  # Skip the generic processing below

            # Generic handling for all other operations
            cells = list(expand_range(op.cell_range.range))  # Returns List[Tuple[int, int]]

            for row, col in cells:
                cell_addr = get_cell_address(row, col)
                # PasteFrom returns a list, others return a string
                if isinstance(property_type, list):
                    for pt in property_type:
                        fingerprint.add((sheet, cell_addr, pt))
                else:
                    fingerprint.add((sheet, cell_addr, property_type))

        return fingerprint

    # ------------------------------------------------------------------
    # Property type mapping
    # ------------------------------------------------------------------

    def _get_property_type(self, op: Operation) -> Union[str, List[str]]:
        """
        Get the property type string(s) for an operation.

        Maps operation class names to property paths used in state comparison.
        Most operations affect a single property (returns str), but PasteFrom
        affects multiple properties based on paste mode (returns List[str]).

        Returns:
            Property type string, or list of property types for PasteFrom
        """
        op_name = type(op).__name__

        # Handle PasteFrom specially - it affects multiple properties based on paste mode
        if op_name == "PasteFrom":
            paste_mode = getattr(op, 'value', 'all')  # Default to 'all' if not set
            if paste_mode == 'all':
                # Affects value, formula, all formatting, and merged cells
                return ["value", "formula", "Format", "merged_cells"]
            elif paste_mode == 'values':
                return "value"
            elif paste_mode == 'formats':
                # Formats paste also copies merged cells
                return ["Format", "merged_cells"]
            elif paste_mode == 'formulas':
                return "formula"
            else:
                # Unknown paste mode, assume all
                return ["value", "formula", "Format", "merged_cells"]

        # Handle SetFontProperty - uses 'property' attribute to specify which font property
        if op_name == "SetFontProperty":
            font_prop = getattr(op, 'property', '').lower()
            font_property_map = {
                'bold': 'Format.font.bold',
                'italic': 'Format.font.italic',
                'underline': 'Format.font.underline',
                'name': 'Format.font.name',
                'size': 'Format.font.size',
                'color': 'Format.font.color',
            }
            return font_property_map.get(font_prop, 'Format.font')

        # Handle SetAlignment - uses 'alignment_type' attribute
        if op_name == "SetAlignment":
            align_type = getattr(op, 'alignment_type', '').lower()
            if align_type == 'horizontal':
                return 'Format.horizontalAlignment'
            elif align_type == 'vertical':
                return 'Format.verticalAlignment'
            return 'Format'  # Fallback

        # Handle SetBorder - uses 'side' attribute
        if op_name == "SetBorder":
            side = getattr(op, 'side', '').lower()
            side_map = {
                'left': 'Format.borders.left',
                'right': 'Format.borders.right',
                'top': 'Format.borders.top',
                'bottom': 'Format.borders.bottom',
                'all': 'Format.borders',
                'outside': 'Format.borders',
                'inside': 'Format.borders',
                'inside_horizontal': 'Format.borders',
                'inside_vertical': 'Format.borders',
            }
            return side_map.get(side, 'Format.borders')

        # Handle AutoFill - affects values, formulas, and formatting
        if op_name == "AutoFill":
            return ["value", "formula", "Format"]

        # Map operation class names to property categories
        property_map = {
            "SetValue": "value",
            "SetFormula": "formula",
            "SetInput": "value",  # SetInput sets values
            "SetNumberFormat": "number_format",
            "SetFillColor": "Format.fill.fgColor",
            "SetTextOrientation": "Format.textOrientation",
            "SetWrapText": "Format.wrapText",
            "MergeCells": "merged_cells",
        }

        return property_map.get(op_name, op_name)

    # ------------------------------------------------------------------
    # Fingerprint overlap
    # ------------------------------------------------------------------

    def _fingerprints_overlap(
        self,
        fp1: Set[Tuple[str, str, str]],
        fp2: Set[Tuple[str, str, str]]
    ) -> bool:
        """
        Check if two fingerprints have any overlapping entries.

        Handles hierarchical property matching:
        - "Format" matches any "Format.*" property
        - "Format.borders" matches "Format.borders.left", "Format.borders.right", etc.
        - "Format.font" matches "Format.font.bold", "Format.font.color", etc.

        Args:
            fp1: First fingerprint set of (sheet, cell, property) tuples
            fp2: Second fingerprint set of (sheet, cell, property) tuples

        Returns:
            True if there's any overlap between the fingerprints
        """
        # Handle empty inputs
        if not fp1 or not fp2:
            return False

        # First try simple set intersection (handles exact matches)
        if fp1 & fp2:
            return True

        # Check for hierarchical property matching
        for sheet1, cell1, prop1 in fp1:
            for sheet2, cell2, prop2 in fp2:
                if sheet1 != sheet2 or cell1 != cell2:
                    continue
                # Same sheet and cell - check property matching
                if self._properties_are_related(prop1, prop2):
                    return True

        return False

    def _properties_are_related(self, prop1: str, prop2: str) -> bool:
        """
        Check if two property paths are related (one is prefix of the other).

        Examples:
        - "Format" and "Format.font.bold" -> True
        - "Format.borders" and "Format.borders.left" -> True
        - "Format.font" and "Format.borders" -> False
        - "value" and "formula" -> False
        """
        if prop1 == prop2:
            return True
        # prop1 is a prefix of prop2
        if prop2.startswith(prop1 + "."):
            return True
        # prop2 is a prefix of prop1
        if prop1.startswith(prop2 + "."):
            return True
        return False

    def _is_format_property(self, prop_type: Union[str, List[str]]) -> bool:
        """
        Check if a property type represents a format property.

        Format properties are affected by MERGE operations (which clear borders/fill
        on non-anchor cells). Used in positioning to prioritize placing format ops
        after MERGE operations.

        Args:
            prop_type: Property type string or list of property types

        Returns:
            True if the property type is a format property
        """
        format_prefixes = ("Format", "Format.")

        if isinstance(prop_type, list):
            # For list of properties, check if any is format-related
            return any(
                p.startswith(format_prefixes) if isinstance(p, str) else False
                for p in prop_type
            )

        if isinstance(prop_type, str):
            return prop_type.startswith(format_prefixes)

        return False

    # ------------------------------------------------------------------
    # Smart operation placement
    # ------------------------------------------------------------------

    def _place_remaining_ops_in_future(
        self,
        future_after: List[Operation],
        remaining_ops: List[Operation],
    ) -> List[Operation]:
        """
        Place remaining ops (inverse ops and missing ops from second pass) into future_after
        at the correct positions based on fingerprint matching.

        Algorithm for each remaining op:
        1. Match by fingerprint: find last op in future_after with overlapping fingerprint
        2. MERGE/UNMERGE match: for format ops, find last MergeCells op with overlapping range
           (MERGE clears borders/fill on non-anchor cells, so format FNs caused by MERGE
           should be placed after that MERGE)
        3. Fallback - match by range intersection: find last op with intersecting range
        4. Fallback: append at end

        Each op is placed independently (they have different fingerprints).

        Args:
            future_after: Current list of future operations
            remaining_ops: Remaining inverse ops and missing ops to place

        Returns:
            Updated future_after with remaining_ops inserted at correct positions
        """
        if not remaining_ops:
            return future_after

        # Pre-build fingerprint map for future_after ops (optimization)
        # Maps (sheet, cell, property) -> list of indices in future_after
        fp_to_indices: Dict[Tuple[str, str, str], List[int]] = {}
        future_fingerprints: List[Set[Tuple[str, str, str]]] = []

        for idx, op in enumerate(future_after):
            op_fp = self._build_fingerprint([op])
            future_fingerprints.append(op_fp)
            for fp_tuple in op_fp:
                if fp_tuple not in fp_to_indices:
                    fp_to_indices[fp_tuple] = []
                fp_to_indices[fp_tuple].append(idx)

        # Track insertions: (original_index, op) - we'll sort and insert later
        # Using original_index + 0.5 means "insert after index"
        insertions: List[Tuple[float, Operation]] = []

        for remaining_op in remaining_ops:
            remaining_fp = self._build_fingerprint([remaining_op])
            insert_after_idx: Optional[int] = None

            # Strategy 1: Find last matching fingerprint in future_after
            best_fp_match_idx = -1
            for fp_tuple in remaining_fp:
                # Check exact match
                if fp_tuple in fp_to_indices:
                    for idx in fp_to_indices[fp_tuple]:
                        if idx > best_fp_match_idx:
                            best_fp_match_idx = idx

                # Check hierarchical match (e.g., Format.borders vs Format.borders.left)
                fp_sheet, fp_cell, fp_prop = fp_tuple
                for (lk_sheet, lk_cell, lk_prop), indices in fp_to_indices.items():
                    if lk_sheet != fp_sheet or lk_cell != fp_cell:
                        continue
                    if self._properties_are_related(fp_prop, lk_prop):
                        for idx in indices:
                            if idx > best_fp_match_idx:
                                best_fp_match_idx = idx

            if best_fp_match_idx >= 0:
                insert_after_idx = best_fp_match_idx
            else:
                # Strategy 2: For format ops, find last MergeCells op with overlapping range
                # MERGE operations clear borders/fill on non-anchor cells, so format FNs
                # caused by MERGE should be placed after that MERGE operation
                remaining_prop_type = self._get_property_type(remaining_op)
                is_format_op = self._is_format_property(remaining_prop_type)

                if is_format_op:
                    try:
                        remaining_range = remaining_op.cell_range
                        best_merge_match_idx = -1

                        for idx, op in enumerate(future_after):
                            if isinstance(op, MergeCells):
                                try:
                                    if self._ranges_overlap(remaining_range, op.cell_range):
                                        best_merge_match_idx = idx
                                except Exception:
                                    continue

                        if best_merge_match_idx >= 0:
                            insert_after_idx = best_merge_match_idx
                    except Exception:
                        pass

                # Strategy 3: Find last op with intersecting range (fallback)
                if insert_after_idx is None:
                    try:
                        remaining_range = remaining_op.cell_range
                        best_range_match_idx = -1

                        for idx, op in enumerate(future_after):
                            try:
                                if self._ranges_overlap(remaining_range, op.cell_range):
                                    best_range_match_idx = idx
                            except Exception:
                                continue

                        if best_range_match_idx >= 0:
                            insert_after_idx = best_range_match_idx
                    except Exception:
                        pass

            if insert_after_idx is not None:
                # Insert after this index
                insertions.append((insert_after_idx + 0.5, remaining_op))
            else:
                # Append at end (use a large number)
                insertions.append((len(future_after) + len(insertions), remaining_op))

        # Sort insertions by position (stable sort maintains relative order for same position)
        insertions.sort(key=lambda x: x[0])

        # Build result by inserting at correct positions
        result: List[Operation] = []
        insertion_idx = 0

        for i, op in enumerate(future_after):
            result.append(op)
            # Insert any ops that should come after index i
            while insertion_idx < len(insertions):
                pos, insert_op = insertions[insertion_idx]
                if pos < i + 1:
                    result.append(insert_op)
                    insertion_idx += 1
                else:
                    break

        # Append any remaining insertions (those at the end)
        while insertion_idx < len(insertions):
            _, insert_op = insertions[insertion_idx]
            result.append(insert_op)
            insertion_idx += 1

        logger.debug(
            "Placed %d remaining ops in future_after (size %d -> %d)",
            len(remaining_ops), len(future_after), len(result)
        )

        return result

    # ------------------------------------------------------------------
    # Range overlap
    # ------------------------------------------------------------------

    def _ranges_overlap(self, range1: CellRange, range2: CellRange) -> bool:
        """
        Check if two cell ranges overlap (share any cells).

        Args:
            range1: First CellRange
            range2: Second CellRange

        Returns:
            True if the ranges share at least one cell
        """
        # Must be on same sheet
        if range1.sheet != range2.sheet:
            return False

        # Get coordinates for both ranges
        r1_start_row, r1_start_col, r1_end_row, r1_end_col = range1.get_coordinates()
        r2_start_row, r2_start_col, r2_end_row, r2_end_col = range2.get_coordinates()

        # Check for overlap: ranges overlap if one doesn't end before the other starts
        # No overlap if: r1 ends before r2 starts, or r2 ends before r1 starts
        # (in either dimension)
        no_overlap = (
            r1_end_row < r2_start_row or  # r1 above r2
            r2_end_row < r1_start_row or  # r2 above r1
            r1_end_col < r2_start_col or  # r1 left of r2
            r2_end_col < r1_start_col     # r2 left of r1
        )
        return not no_overlap

    # ------------------------------------------------------------------
    # Property matching
    # ------------------------------------------------------------------

    def _property_matches(
        self,
        diff_property_path: str,
        op_property_types: Set[str]
    ) -> bool:
        """
        Check if a difference's property path matches any of the operation's property types.

        This ensures that when processing a GT operation (e.g., BORDER_ALL), we only
        consider FN differences that are relevant to that operation's effect (e.g.,
        Format.borders.*), not unrelated differences (e.g., value, Format.font.*).

        Args:
            diff_property_path: Property path from the FN difference (e.g., "Format.borders.left")
            op_property_types: Set of property types the GT operation affects (e.g., {"Format.borders"})

        Returns:
            True if the difference's property matches any of the operation's property types
        """
        # Handle None or empty inputs gracefully
        if not diff_property_path or not op_property_types:
            return False

        for prop_type in op_property_types:
            if prop_type is None:
                continue
            # Exact match
            if diff_property_path == prop_type:
                return True

            # Prefix matching: "Format" matches "Format.borders.left", "Format.font.bold", etc.
            if diff_property_path.startswith(prop_type + "."):
                return True

            # Reverse prefix: "Format.borders" should match if op has "Format.borders.left"
            if prop_type.startswith(diff_property_path + "."):
                return True

            # Special case: "merged_cells" is its own property type
            if diff_property_path == "merged_cells" and prop_type == "merged_cells":
                return True

        return False

    # ------------------------------------------------------------------
    # Border consolidation
    # ------------------------------------------------------------------

    def _consolidate_border_differences(
        self,
        differences: List[PropertyDifference],
        target_state: Optional[Dict[str, Any]],
    ) -> List[PropertyDifference]:
        """
        Consolidate border FN differences for BORDER_ALL optimization.

        If a cell has some border sides as FN but the target state shows all 4 sides
        with the same style, add the missing sides so parse_excel can generate BORDER_ALL.

        Args:
            differences: Original FN differences
            target_state: The target state to check full border info

        Returns:
            Updated differences list with consolidated border entries
        """
        if not target_state:
            return differences

        # Collect border FNs by (sheet, cell)
        border_fns: Dict[Tuple[str, str], Dict[str, PropertyDifference]] = {}
        non_border_diffs = []

        for diff in differences:
            if diff.property_path.startswith("Format.borders."):
                side = diff.property_path.split(".")[-1]  # left, right, top, bottom
                key = (diff.sheet, diff.cell)
                if key not in border_fns:
                    border_fns[key] = {}
                border_fns[key][side] = diff
            else:
                non_border_diffs.append(diff)

        if not border_fns:
            return differences

        # For each cell with border FNs, check target state
        consolidated_diffs = list(non_border_diffs)
        all_sides = ["left", "right", "top", "bottom"]

        for (sheet, cell), sides_dict in border_fns.items():
            # Get target borders for this cell
            target_borders = self._get_cell_borders_from_state(target_state, sheet, cell)

            if not target_borders:
                # No target borders, keep original FNs
                consolidated_diffs.extend(sides_dict.values())
                continue

            # Check if all 4 sides in target have same non-null value
            target_values = [target_borders.get(s) for s in all_sides]
            non_null = [v for v in target_values if v is not None]

            if len(non_null) == 4 and len(set(str(v) for v in non_null)) == 1:
                # All 4 sides same - add any missing sides to FNs
                for side in all_sides:
                    if side in sides_dict:
                        consolidated_diffs.append(sides_dict[side])
                    else:
                        # Create a new diff for the missing side (copy from target)
                        new_diff = PropertyDifference(
                            sheet=sheet,
                            cell=cell,
                            property_path=f"Format.borders.{side}",
                            predicted_value=None,  # We're adding it
                            true_value=target_borders[side],
                            match_type="FN",
                        )
                        consolidated_diffs.append(new_diff)
            else:
                # Not all same, keep original FNs
                consolidated_diffs.extend(sides_dict.values())

        return consolidated_diffs

    def _get_cell_borders_from_state(
        self,
        state: Dict[str, Any],
        sheet: str,
        cell: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract border dict {left, right, top, bottom} from state for a cell."""
        try:
            sheet_data = state.get("worksheets", {}).get(sheet, {})
            cells = sheet_data.get("cells", {})
            cell_data = cells.get(cell, {})
            return cell_data.get("Format", {}).get("borders", {})
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Operation synthesis from state differences
    # ------------------------------------------------------------------

    def _synthesize_operations_from_differences(
        self,
        differences: List[PropertyDifference],
        target_state: Optional[Dict[str, Any]] = None,
    ) -> List[Operation]:
        """
        Convert FN PropertyDifferences to operations.

        Uses ExcelParser to convert a synthetic state dict (built from differences)
        into operations, then merges borders into BORDER_ALL/BORDER_OUTSIDE where
        possible.

        For borders: If target_state is provided and all 4 border sides of a cell
        have the same style, includes all 4 to enable BORDER_ALL consolidation.
        """
        # Pre-process: consolidate border FNs using target state
        differences = self._consolidate_border_differences(differences, target_state)

        # Build a state dict from the FN differences
        state: Dict[str, Any] = {"worksheets": {}}

        for diff in differences:
            sheet_name = diff.sheet
            cell_addr = diff.cell
            prop_path = diff.property_path
            value = diff.true_value  # Use the true value (what we need to achieve)

            if value is None:
                continue

            # Ensure worksheet exists
            if sheet_name not in state["worksheets"]:
                state["worksheets"][sheet_name] = {
                    "cells": {},
                    "worksheetProperties": {"merged_cells": []}
                }

            # Handle merged_cells specially
            if prop_path == "merged_cells":
                if isinstance(value, tuple) and len(value) == 4:
                    merge_dict = {
                        "start_row": value[0],
                        "start_col": value[1],
                        "end_row": value[2],
                        "end_col": value[3]
                    }
                    state["worksheets"][sheet_name]["worksheetProperties"]["merged_cells"].append(merge_dict)
                continue

            # Ensure cell exists
            if cell_addr not in state["worksheets"][sheet_name]["cells"]:
                state["worksheets"][sheet_name]["cells"][cell_addr] = {}

            cell_data = state["worksheets"][sheet_name]["cells"][cell_addr]

            # Set nested property value
            _set_nested_value(cell_data, prop_path, value)

            # Special handling for fill colors: ExcelParser requires patternType
            # to be set for fill colors to be parsed correctly
            if prop_path == "Format.fill.fgColor":
                _set_nested_value(
                    cell_data, "Format.fill.patternType", "solid"
                )

        # Convert state to operations using ExcelParser
        parser = ExcelParser()
        operations = parser.parse(state=state)

        # Merge same-signature formatting ops into rectangles (e.g. FILL C6 + FILL D6 -> FILL C6:D6)
        # merge_inputs=False: value/formula corrections stay individual (only formatting merges)
        border_ops = [op for op in operations if isinstance(op, SetBorder)]
        non_border_ops = [op for op in operations if not isinstance(op, SetBorder)]

        if non_border_ops:
            rectangle_merger = RectangleMerger()
            non_border_ops = rectangle_merger.merge(non_border_ops, merge_inputs=False)

        if border_ops:
            border_merger = BorderMerger()
            border_ops = border_merger.merge(border_ops)

        operations = non_border_ops + border_ops

        # Apply deterministic ordering: priority groups (merges first, then values,
        # then formatting), spatial sort within each group
        if operations:
            sequencer = OperationSequencer({"scope": "global", "ordering_strategy": "hybrid"})
            context = SequencingContext(operations=operations)
            operations = sequencer.transform(context).operations

        return operations

    # ------------------------------------------------------------------
    # Simulation (read-only)
    # ------------------------------------------------------------------

    def simulate_future_edits(
        self,
        current_gt: List[Operation],
        start_idx: int,
        end_idx: int,
        predicted_ops: List[Operation],
        eval_result: EvaluationResult,
        initial_state: Optional[Dict[str, Any]] = None,
        final_target_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Operation], Dict[str, object]]:
        """Run future-edits logic on a copy of the ground truth without mutating input.

        Returns:
            Tuple of (new_gt, changes_dict). The new_gt is the result of applying
            future edits to a snapshot — can be adopted directly for accepted
            predictions to avoid re-running apply_future_edits.

        Note: If states are not provided, returns (original GT copy, empty changes).
        """
        if initial_state is None or final_target_state is None:
            # Return minimal changes dict when states not available
            return list(current_gt), {
                "old_length": len(current_gt),
                "new_length": len(current_gt),
                "original_region": [],
                "operations_added": [],
                "operations_removed": [],
                "inverse_ops_added": [],
                "removed_from_after_region": [],
                "dedup_window_range": (start_idx, len(current_gt)),
                "metadata": {
                    "algorithm": "preview_no_state",
                    "note": "Full analysis requires initial_state and final_target_state",
                },
                "summary": {
                    "final_state_gain": 0,
                    "dedup_gain": 0,
                    "inverse_cost": 0,
                    "net_gain": 0,
                    "operations_added": 0,
                    "old_length": len(current_gt),
                    "new_length": len(current_gt),
                },
            }

        snapshot = list(current_gt)
        new_gt, changes = self.apply_future_edits(
            current_gt=snapshot,
            start_idx=start_idx,
            end_idx=end_idx,
            predicted_ops=predicted_ops,
            eval_result=eval_result,
            initial_state=initial_state,
            final_target_state=final_target_state,
        )
        return new_gt, changes

    # ------------------------------------------------------------------
    # Inverse operation extraction
    # ------------------------------------------------------------------

    def _extract_inverse_ops(self, eval_result: EvaluationResult) -> List[Operation]:
        if not eval_result:
            return []

        comparison = getattr(eval_result, "final_state_comparison", None)
        if not comparison:
            return []

        inverse_ops = list(getattr(comparison, "inverse_ops_merged", []) or [])
        if self.config.max_inverse_ops is not None:
            inverse_ops = inverse_ops[: self.config.max_inverse_ops]
        return inverse_ops

    def _extract_inverse_ops_from_comparison(self, comparison: Any) -> List[Operation]:
        """
        Extract inverse ops directly from a ComparisonResult object.

        This is used when we need to calculate inverse ops from a fresh comparison
        (e.g., after simulating the full trajectory including PASTE_FROM ops).

        Args:
            comparison: ComparisonResult object from state_comparator.compare()

        Returns:
            List of inverse operations to cancel pure FP properties
        """
        if not comparison:
            return []

        inverse_ops = list(getattr(comparison, "inverse_ops_merged", []) or [])
        if self.config.max_inverse_ops is not None:
            inverse_ops = inverse_ops[: self.config.max_inverse_ops]
        return inverse_ops

    # ------------------------------------------------------------------
    # Merge special handling
    # ------------------------------------------------------------------

    def _build_unmerge_ops_for_fp_merges(
        self,
        fp_merge_diffs: List[PropertyDifference]
    ) -> List[Operation]:
        """
        Build UNMERGE operations for FP merged_cells.

        When prediction applies a merge that's different from the GT merge,
        we need to UNMERGE the wrong prediction merge so the GT merge can be
        applied correctly later.

        Args:
            fp_merge_diffs: List of FP PropertyDifference for merged_cells

        Returns:
            List of UNMERGE (MergeCells with value=False) operations
        """
        unmerge_ops = []

        for diff in fp_merge_diffs:
            if diff.property_path != "merged_cells" or diff.predicted_value is None:
                continue

            # predicted_value is a tuple (start_row, start_col, end_row, end_col)
            merge_tuple = diff.predicted_value
            if isinstance(merge_tuple, tuple) and len(merge_tuple) == 4:
                start_row, start_col, end_row, end_col = merge_tuple
                # Build the range string with sheet
                range_str = f"{diff.sheet}!{get_range_string(start_row, start_col, end_row, end_col)}"
                cell_range = CellRange.from_string(range_str)
                # Create UNMERGE operation (MergeCells with value=False, is_inverse=True)
                unmerge_op = MergeCells(cell_range=cell_range, value=False, is_inverse=True)
                unmerge_ops.append(unmerge_op)
                logger.debug("Created UNMERGE for FP merge: %s", range_str)

        # NOTE: No sequencing for unmerge ops - they're individual UNMERGE operations
        # that can't be consolidated or benefit from reordering

        return unmerge_ops

    def _generate_unmerge_correction_ops(
        self,
        fp_merge_diffs: List[PropertyDifference],
        unmerge_ops: List[Operation],
        prediction_state: Dict[str, Any],
        initial_state: Dict[str, Any],
    ) -> List[Operation]:
        """
        Generate correction operations for data lost when unmerging cells.

        When the prediction creates an FP merge, data from initial_state is LOST:
        - Excel's merge clears values/formats from non-anchor cells
        - After UNMERGE, the cells are still empty
        - We need to restore what was there BEFORE the prediction

        Algorithm:
        1. Apply unmerge_ops to prediction_state to get unmerged_state
        2. Compare unmerged_state vs initial_state
        3. FN diffs = what initial_state had but unmerged_state is missing
        4. Filter to only cells within the FP merge ranges
        5. Synthesize correction ops from those FN diffs

        Args:
            fp_merge_diffs: List of FP PropertyDifference for merged_cells
            unmerge_ops: UNMERGE operations that will be applied
            prediction_state: State after applying prediction (has the FP merges)
            initial_state: State before prediction (has the original values)

        Returns:
            List of correction operations (SetValue, SetFont, etc.)
        """
        if not fp_merge_diffs or not unmerge_ops:
            return []

        # Build set of cells within FP merge ranges
        merge_cells: Set[Tuple[str, str]] = set()
        for fp_diff in fp_merge_diffs:
            if fp_diff.property_path != "merged_cells" or fp_diff.predicted_value is None:
                continue

            merge_tuple = fp_diff.predicted_value
            if not isinstance(merge_tuple, tuple) or len(merge_tuple) != 4:
                continue

            start_row, start_col, end_row, end_col = merge_tuple
            sheet_name = fp_diff.sheet

            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    cell_addr = get_cell_address(row, col)
                    merge_cells.add((sheet_name, cell_addr))

        if not merge_cells:
            return []

        # Apply unmerge_ops to prediction_state to get unmerged state
        unmerged_state = self._state_builder.build_state(
            initial_state=deepcopy(prediction_state),
            operations=unmerge_ops,
        )

        # Compare unmerged_state vs initial_state to find what was LOST
        # FN diffs = what initial_state has but unmerged_state is missing
        comparison = self._state_comparator.compare(
            predicted_state=unmerged_state,
            true_state=initial_state,
            skip_ops_diff=True,
        )

        # Filter FN diffs to only cells within the FP merge ranges
        # Exclude merged_cells property itself - we only want value/format FNs
        correction_diffs: List[PropertyDifference] = []
        for diff in comparison.differences:
            if diff.match_type not in ("FN", "MISMATCH"):
                continue
            if diff.property_path == "merged_cells":
                continue
            if (diff.sheet, diff.cell) in merge_cells:
                correction_diffs.append(diff)
                logger.debug(
                    "Found lost data in unmerge range: %s!%s.%s (was: %s)",
                    diff.sheet, diff.cell, diff.property_path, diff.true_value
                )

        if not correction_diffs:
            return []

        # Synthesize correction operations from the FN diffs
        # Use initial_state as target since that's what we're restoring to
        correction_ops = self._synthesize_operations_from_differences(correction_diffs, initial_state)

        logger.debug(
            "Generated %d correction ops for %d FN diffs (data lost by FP merges)",
            len(correction_ops), len(correction_diffs)
        )

        return correction_ops


__all__ = [
    "FutureEditsManager",
    "FutureEditsConfig",
    "FutureEditChanges",
]
