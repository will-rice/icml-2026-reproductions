import copy
import json
import json
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Set, Tuple, Union
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
)
from next_action_pred_eval.utils.cell_utils import (
    expand_range,
    cells_to_range,
    get_range_string,
)

class RectangleMerger:
    def merge(self, operations: List[Operation], row_first=True, sort_input_by_type=True, merge_inputs=False, merge_formulas=False) -> List[Operation]:
        """
        Merge non-border operations into rectangles. Borders are ignored.

        Args:
            operations: List of operations to merge
            row_first: Whether to grow rectangles row-first or column-first
            sort_input_by_type: Whether to group SetInput operations by value type for merging by inferring from values
            merge_inputs: Whether to merge SetValue/SetFormula/SetInput operations into rectangles
        """
        if not operations:
            return []

        self.sort_input_by_type = sort_input_by_type

        merge_ops: List[MergeCells] = [
            op for op in operations if isinstance(op, MergeCells)
        ]
        value_formula_ops: List[Operation] = [
            op
            for op in operations
            if isinstance(op, (SetValue, SetFormula, SetInput))
        ]
        other_ops: List[Operation] = [
            op
            for op in operations
            if not isinstance(op, (SetBorder, SetValue, SetFormula, SetInput, MergeCells))
        ]

        merged: List[Operation] = []
        merged.extend(merge_ops)

        # Handle input operations based on merge_inputs and merge_formulas flags
        if merge_inputs:
            if merge_formulas:
                # Merge all input operations together (including formulas)
                merged_input_ops = self._merge_rectangles(value_formula_ops, row_first=row_first)
                for merged_op in merged_input_ops:
                    # Convert merged rectangles to SetInput if they cover multiple cells
                    if not self._is_single_cell(merged_op.cell_range.range):
                        input_op = self._convert_to_set_input_with_values(merged_op, value_formula_ops)
                        if input_op:
                            merged.append(input_op)
                        else:
                            merged.append(merged_op)
                    else:
                        merged.append(merged_op)
            else:
                # Merge only non-formula operations (SetValue and SetInput)
                formula_ops = [op for op in value_formula_ops if isinstance(op, SetFormula)]
                non_formula_ops = [op for op in value_formula_ops if not isinstance(op, SetFormula)]

                # Add formulas without merging
                merged.extend(formula_ops)

                # Merge non-formula operations
                merged_input_ops = self._merge_rectangles(non_formula_ops, row_first=row_first)
                for merged_op in merged_input_ops:
                    # Convert merged rectangles to SetInput if they cover multiple cells
                    if not self._is_single_cell(merged_op.cell_range.range):
                        input_op = self._convert_to_set_input_with_values(merged_op, non_formula_ops)
                        if input_op:
                            merged.append(input_op)
                        else:
                            merged.append(merged_op)
                    else:
                        merged.append(merged_op)
        else:
            # Backward compatibility: don't merge input operations
            merged.extend(value_formula_ops)

        groups: Dict[Tuple[str, str], List[Operation]] = {}
        for op in other_ops:
            key = (op.cell_range.sheet, self._op_signature(op))
            groups.setdefault(key, []).append(op)

        for (_, _), ops in groups.items():
            merged.extend(self._merge_rectangles(ops, row_first=row_first))

        return merged

    def _op_signature(self, op: Operation) -> str:
        """Create a signature so that ops with the same effect are coalesced."""
        # Get base signature first
        base_signature = self._get_base_signature(op)

        # Check if operation is inverse (clearing operation) and prepend CLEAR prefix
        if op.is_inverse:
            return f"CLEAR-{base_signature}"

        return base_signature

    def _get_base_signature(self, op: Operation) -> str:
        """Get the base signature for an operation without inverse prefix."""
        if isinstance(op, SetInput):
            if not self.sort_input_by_type:
                return f"INPUT"

            # Determine actual value structure
            op_value = op.value
            if not isinstance(op_value, list):
                op_value = [[op_value]]
            elif op_value and not isinstance(op_value[0], list):
                # 1D array - treat as single row
                op_value = [op_value]

            # Flatten and analyze (keep None for empty detection)
            flat_values = [item for sublist in op_value for item in sublist]
            # Filter None for type checking
            non_none_values = [v for v in flat_values if v is not None]

            # Check empty first
            if not non_none_values or all(v == "" or v is None for v in flat_values):
                return f"INPUT::empty"

            # Check boolean BEFORE numeric (bool is subclass of int)
            if all(isinstance(v, bool) for v in non_none_values):
                return f"INPUT::boolean"

            # Check numeric types
            if all(isinstance(v, (int, float)) for v in non_none_values):
                if all(isinstance(v, int) for v in non_none_values):
                    return f"INPUT::numeric_integer"
                elif all(isinstance(v, float) for v in non_none_values):
                    return f"INPUT::numeric_float"
                else:
                    return f"INPUT::numeric_mixed"

            # Check for datetime/date/time types (stored as ISO format strings)
            if all(isinstance(v, str) for v in non_none_values):
                from datetime import datetime, time

                types = set()
                for v in non_none_values:
                    try:
                        dt = datetime.fromisoformat(str(v))
                        has_time = dt.hour or dt.minute or dt.second or dt.microsecond
                        types.add('datetime' if has_time else 'date')
                    except:
                        try:
                            # Require colons for time classification to avoid
                            # compact HHMMSS format matching 6-digit ID strings
                            # (e.g., "123456" would parse as 12:34:56 without this guard)
                            if ':' not in str(v):
                                raise ValueError("Not a time string")
                            time.fromisoformat(str(v))
                            types.add('time')
                        except:
                            types.add('text')
                            break  # Found non-temporal, no need to continue

                if types == {'text'}:
                    return f"INPUT::text"
                elif len(types) == 1:
                    return f"INPUT::{types.pop()}"
                elif 'text' in types:
                    return f"INPUT::mixed"
                else:
                    return f"INPUT::temporal_mixed"

            return f"INPUT::mixed"

        # SetValue and SetFormula are separate from SetInput
        if isinstance(op, SetValue):  # SPECIAL CASE for INPUT
            return f"INPUT_VALUE::{op.datatype or 'none'}"
        if isinstance(op, SetFormula):
            return f"INPUT_FORMULA"  # SPECIAL CASE for INPUT
        if isinstance(op, SetNumberFormat):
            return f"NUMBER_FORMAT::{op.value}"
        if isinstance(op, SetFontProperty):
            return f"FONT::{op.property}::{json.dumps(op.value) if isinstance(op.value, (dict, list, str)) else op.value}"
        if isinstance(op, SetWrapText):
            return f"wrap_{op.value}"
        if isinstance(op, SetFillColor):
            return f"FILL::{op.value}"
        if isinstance(op, SetAlignment):
            return f"ALIGN::{op.alignment_type}::{op.value}"
        if isinstance(op, SetTextOrientation):
            return f"TEXT_ORIENTATION::{op.value}"
        # For unknown types, avoid merging
        return f"UNKNOWN::{type(op).__name__}"

    def _merge_rectangles(
        self, ops: List[Operation], *, row_first: bool
    ) -> List[Operation]:
        """Merge a homogeneous list of operations into rectangles."""
        # Expand ranges into individual cells
        cell_to_op: Dict[Tuple[str, int, int], Operation] = {}
        for op in ops:
            for r, c in expand_range(op.cell_range.range):
                cell_to_op[(op.cell_range.sheet, r, c)] = op

        processed: set[Tuple[str, int, int]] = set()
        merged_ops: List[Operation] = []

        for (sheet, r, c), base_op in sorted(cell_to_op.items()):
            key = (sheet, r, c)
            if key in processed:
                continue

            cells = self._grow_rectangle(
                cell_to_op, sheet, r, c, processed, row_first=row_first
            )
            range_str = cells_to_range(cells)
            new_range = CellRange(sheet=sheet, range=range_str)
            merged_ops.append(self._rebuild_op(base_op, new_range))

        return merged_ops
    def _grow_rectangle(
        self,
        cell_to_op: Dict[Tuple[str, int, int], Operation],
        sheet: str,
        r0: int,
        c0: int,
        processed: set[Tuple[str, int, int]],
        *,
        row_first: bool,
    ) -> List[Tuple[int, int]]:
        base = cell_to_op[(sheet, r0, c0)]

        # First dimension growth
        r1, r2, c1, c2 = r0, r0, c0, c0

        def same(r: int, c: int) -> bool:
            op = cell_to_op.get((sheet, r, c))
            return (
                op is not None
                and self._op_signature(op) == self._op_signature(base)
                and (sheet, r, c) not in processed
            )

        if row_first:
            # grow downward first
            rr = r0
            while same(rr + 1, c0):
                rr += 1
            r2 = rr
            # then grow right, ensuring all rows match
            cc = c0
            while True:
                nxt = cc + 1
                ok = True
                for r in range(r1, r2 + 1):
                    if not same(r, nxt):
                        ok = False
                        break
                if ok:
                    cc = nxt
                else:
                    break
            c2 = cc
        else:
            # grow right first
            cc = c0
            while same(r0, cc + 1):
                cc += 1
            c2 = cc
            # then grow down, ensuring all cols match
            rr = r0
            while True:
                nxt = rr + 1
                ok = True
                for c in range(c1, c2 + 1):
                    if not same(nxt, c):
                        ok = False
                        break
                if ok:
                    rr = nxt
                else:
                    break
            r2 = rr

        # Mark processed and return cells
        cells: List[Tuple[int, int]] = []
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                processed.add((sheet, r, c))
                cells.append((r, c))
        return cells

    def _rebuild_op(self, base: Operation, cell_range: CellRange) -> Operation:
        is_inverse = getattr(base, 'is_inverse', False)
        if isinstance(base, SetInput):
            return SetInput(cell_range=cell_range, value=base.value, is_inverse=is_inverse)
        if isinstance(base, SetValue):
            return SetValue(cell_range=cell_range, value=base.value, datatype=base.datatype, is_inverse=is_inverse)
        if isinstance(base, SetFormula):
            return SetFormula(cell_range=cell_range, value=base.value, is_inverse=is_inverse)
        if isinstance(base, SetNumberFormat):
            return SetNumberFormat(cell_range=cell_range, value=base.value, is_inverse=is_inverse)
        if isinstance(base, SetFontProperty):
            return SetFontProperty(cell_range=cell_range, property=base.property, value=base.value, is_inverse=is_inverse)
        if isinstance(base, SetWrapText):
            return SetWrapText(cell_range=cell_range, value=base.value, is_inverse=is_inverse)
        if isinstance(base, SetFillColor):
            return SetFillColor(cell_range=cell_range, value=base.value, is_inverse=is_inverse)
        if isinstance(base, SetAlignment):
            return SetAlignment(cell_range=cell_range, alignment_type=base.alignment_type, value=base.value, is_inverse=is_inverse)
        if isinstance(base, SetTextOrientation):
            return SetTextOrientation(cell_range=cell_range, value=base.value, is_inverse=is_inverse)
        # Borders handled separately
        return base

    def _is_single_cell(self, range_str: str) -> bool:
        """Check if a range string represents a single cell."""
        return ':' not in range_str

    def _convert_to_set_input_with_values(self, merged_op: Union[SetValue, SetFormula, SetInput], original_ops: List[Operation]) -> Optional[SetInput]:
        """Convert a merged operation into a SetInput by collecting values from original operations."""
        try:
            # Parse the range to get dimensions
            cells = list(expand_range(merged_op.cell_range.range))
            if len(cells) <= 1:
                return None

            # Build a map of cell coordinates to values from original operations
            cell_value_map = {}
            for op in original_ops:
                if op.cell_range.sheet == merged_op.cell_range.sheet:
                    for r, c in expand_range(op.cell_range.range):
                        cell_value_map[(r, c)] = op.value

            # Organize cells into a 2D array structure
            rows = sorted(set(r for r, c in cells))
            cols = sorted(set(c for r, c in cells))

            # Create 2D array with actual values from original operations
            values_2d = []
            for row in rows:
                row_values = []
                for col in cols:
                    if (row, col) in cells:
                        # Get the actual value from the original operation for this cell
                        value = cell_value_map.get((row, col))
                        if value is not None:
                            row_values.append(value)
                        else:
                            # Fallback to merged_op value if no original found
                            row_values.append(merged_op.value)
                    else:
                        # This shouldn't happen with rectangular ranges, but handle it
                        row_values.append(None)
                values_2d.append(row_values)

            # Create SetInput operation with 2D array of actual values
            return SetInput(cell_range=merged_op.cell_range, value=values_2d)

        except Exception:
            # If conversion fails, return None to use original operation
            return None

class BorderMerger:
    def merge(self, border_ops: List[SetBorder]) -> List[SetBorder]:
        """Optimized border merging with reduced N"""
        # Separate inverse and non-inverse operations
        inverse_ops = [op for op in border_ops if op.is_inverse]
        non_inverse_ops = [op for op in border_ops if not op.is_inverse]

        merged_operations = []

        # Process inverse and non-inverse operations separately
        merged_operations.extend(self._merge_border_group(inverse_ops, is_inverse=True))
        merged_operations.extend(self._merge_border_group(non_inverse_ops, is_inverse=False))

        return merged_operations

    def _merge_border_group(self, border_ops: List[SetBorder], is_inverse: bool) -> List[SetBorder]:
        """Merge a group of border operations with the same is_inverse status"""
        # Group operations by sheet
        sheet_ops = defaultdict(list)
        for op in border_ops:
            sheet_ops[op.cell_range.sheet].append(op)

        merged_operations = []

        for sheet_name, ops in sheet_ops.items():
            border_map = defaultdict(lambda: {
                'left': None, 'right': None, 'top': None, 'bottom': None
            })
            active_cells = set()

            # First pass: collect all explicitly defined borders
            for border_op in ops:
                cells = list(expand_range(border_op.cell_range.range))
                weight = border_op.value.get('weight', 'Thin') if border_op.value else 'Thin'
                style = border_op.value.get('style', 'Continuous') if border_op.value else 'Continuous'
                color = border_op.value.get('color') if border_op.value else None
                border_style = (weight, style, color)

                # Calculate range boundaries for edge-only border semantics
                rows = [c[0] for c in cells]
                cols = [c[1] for c in cells]
                min_row, max_row = min(rows), max(rows)
                min_col, max_col = min(cols), max(cols)

                for cell in cells:
                    # Single-side borders only apply to the edge of the range
                    if border_op.side == 'left':
                        # Only leftmost column gets left border
                        if cell[1] == min_col:
                            active_cells.add(cell)
                            border_map[cell]['left'] = border_style
                    elif border_op.side == 'right':
                        # Only rightmost column gets right border
                        if cell[1] == max_col:
                            active_cells.add(cell)
                            border_map[cell]['right'] = border_style
                    elif border_op.side == 'top':
                        # Only topmost row gets top border
                        if cell[0] == min_row:
                            active_cells.add(cell)
                            border_map[cell]['top'] = border_style
                    elif border_op.side == 'bottom':
                        # Only bottommost row gets bottom border
                        if cell[0] == max_row:
                            active_cells.add(cell)
                            border_map[cell]['bottom'] = border_style
                    elif border_op.side == 'outside':
                        active_cells.add(cell)
                        if cell[0] == min_row:
                            border_map[cell]['top'] = border_style
                        if cell[0] == max_row:
                            border_map[cell]['bottom'] = border_style
                        if cell[1] == min_col:
                            border_map[cell]['left'] = border_style
                        if cell[1] == max_col:
                            border_map[cell]['right'] = border_style
                    elif border_op.side == 'all':
                        active_cells.add(cell)
                        for side in ['left', 'right', 'top', 'bottom']:
                            border_map[cell][side] = border_style
                    elif border_op.side == 'insideHorizontal':
                        if cell[0] > min_row:
                            active_cells.add(cell)
                            border_map[cell]['top'] = border_style
                        if cell[0] < max_row:
                            active_cells.add(cell)
                            border_map[cell]['bottom'] = border_style
                    elif border_op.side == 'insideVertical':
                        if cell[1] > min_col:
                            active_cells.add(cell)
                            border_map[cell]['left'] = border_style
                        if cell[1] < max_col:
                            active_cells.add(cell)
                            border_map[cell]['right'] = border_style

            # Only process active_cells (cells with borders)
            # Connected component analysis
            regions = self._find_border_regions(active_cells)

            for region in regions:
                region_ops = self._merge_borders_in_region(region, border_map, sheet_name, is_inverse)
                merged_operations.extend(region_ops)

        return merged_operations

    def _merge_borders_in_region(self, region_cells: set, border_map: dict, sheet_name: str, is_inverse: bool) -> List[SetBorder]:
        """Merge borders only within a connected region - O(R²) where R = region size"""
        merged_ops = []

        # Build complete border map for this region only
        complete_border_map = defaultdict(lambda: {
            'left': None, 'right': None, 'top': None, 'bottom': None
        })

        for (r, c) in region_cells:
            borders = border_map[(r, c)]
            # Left border: check this cell's left OR right neighbor's right
            if borders['left']:
                complete_border_map[(r, c)]['left'] = borders['left']
            elif (r, c-1) in border_map and border_map[(r, c-1)]['right']:
                complete_border_map[(r, c)]['left'] = border_map[(r, c-1)]['right']
            # Right border: check this cell's right OR left neighbor's left
            if borders['right']:
                complete_border_map[(r, c)]['right'] = borders['right']
            elif (r, c+1) in border_map and border_map[(r, c+1)]['left']:
                complete_border_map[(r, c)]['right'] = border_map[(r, c+1)]['left']
            # Top border: check this cell's top OR top neighbor's bottom
            if borders['top']:
                complete_border_map[(r, c)]['top'] = borders['top']
            elif (r-1, c) in border_map and border_map[(r-1, c)]['bottom']:
                complete_border_map[(r, c)]['top'] = border_map[(r-1, c)]['bottom']
            # Bottom border: check this cell's bottom OR bottom neighbor's top
            if borders['bottom']:
                complete_border_map[(r, c)]['bottom'] = borders['bottom']
            elif (r+1, c) in border_map and border_map[(r+1, c)]['top']:
                complete_border_map[(r, c)]['bottom'] = border_map[(r+1, c)]['top']

        processed_cells = set()

        # Process only cells in this region
        region_border_map = {cell: complete_border_map[cell] for cell in region_cells}

        # First, try to find complete grids (BORDER_ALL) within the region
        all_border_regions = self._find_complete_grids(region_border_map, region_cells, processed_cells)
        for region_cells_list, border_style in all_border_regions:
            if region_cells_list:
                range_str = cells_to_range(region_cells_list)
                merged_ops.append(
                    SetBorder(
                        cell_range=CellRange(sheet=sheet_name, range=range_str),
                        side='all',
                        weight=border_style[0],
                        lineStyle=border_style[1],
                        color=border_style[2],
                        value=None,
                        is_inverse=is_inverse
                    )
                )

        # Then find boxes (BORDER_OUTSIDE) within the region
        outside_border_regions = self._find_boxes(region_border_map, processed_cells)
        for region_cells_list, border_style in outside_border_regions:
            if region_cells_list:
                range_str = cells_to_range(region_cells_list)
                merged_ops.append(
                    SetBorder(
                        cell_range=CellRange(sheet=sheet_name, range=range_str),
                        side='outside',
                        weight=border_style[0],
                        lineStyle=border_style[1],
                        color=border_style[2],
                        value=None,
                        is_inverse=is_inverse
                    )
                )

        # Handle remaining individual borders within the region
        style_groups = defaultdict(lambda: defaultdict(list))
        for cell in region_cells:
            if cell in processed_cells:
                continue
            # USE complete_border_map HERE, not border_map
            borders = border_map[cell]
            for side, style in borders.items():
                if style:
                    style_groups[style][side].append(cell)

        for style, sides in style_groups.items():
            for side, cells in sides.items():
                processed_side = set()
                for cell in sorted(cells):
                    if cell in processed_side:
                        continue
                    # Only use cells from this region
                    cell_dict = {c: True for c in cells if c not in processed_side and c in region_cells}

                    # For single-side borders, we can only merge in specific directions
                    # because applying a border to a range only affects the edge:
                    # - BORDER_TOP/BOTTOM: can only merge horizontally (same row)
                    # - BORDER_LEFT/RIGHT: can only merge vertically (same column)
                    if side in ['top', 'bottom']:
                        # Only merge horizontally (same row)
                        rect = self._find_horizontal_strip(cell_dict, cell[0], cell[1], processed_side)
                    elif side in ['left', 'right']:
                        # Only merge vertically (same column)
                        rect = self._find_vertical_strip(cell_dict, cell[0], cell[1], processed_side)
                    else:
                        # For other sides (shouldn't happen normally), use rectangle
                        rect = self._find_matching_rectangle(cell_dict, cell[0], cell[1], processed_side)

                    if rect:
                        range_str = cells_to_range(rect)
                        merged_ops.append(
                            SetBorder(
                                cell_range=CellRange(sheet=sheet_name, range=range_str),
                                side=side,
                                weight=style[0],
                                lineStyle=style[1],
                                color=style[2],
                                value=None,
                                is_inverse=is_inverse
                            )
                        )

        return merged_ops

    def _find_horizontal_strip(
        self,
        cells: Dict[Tuple[int, int], Any],
        r0: int,
        c0: int,
        processed: set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Find a horizontal strip of contiguous cells in the same row.
        Used for BORDER_TOP and BORDER_BOTTOM which can only merge horizontally.
        """
        if (r0, c0) not in cells or (r0, c0) in processed:
            return []

        # Only grow right (same row)
        c2 = c0
        while (r0, c2 + 1) in cells and (r0, c2 + 1) not in processed:
            c2 += 1

        strip: List[Tuple[int, int]] = []
        for c in range(c0, c2 + 1):
            processed.add((r0, c))
            strip.append((r0, c))
        return strip

    def _find_vertical_strip(
        self,
        cells: Dict[Tuple[int, int], Any],
        r0: int,
        c0: int,
        processed: set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Find a vertical strip of contiguous cells in the same column.
        Used for BORDER_LEFT and BORDER_RIGHT which can only merge vertically.
        """
        if (r0, c0) not in cells or (r0, c0) in processed:
            return []

        # Only grow down (same column)
        r2 = r0
        while (r2 + 1, c0) in cells and (r2 + 1, c0) not in processed:
            r2 += 1

        strip: List[Tuple[int, int]] = []
        for r in range(r0, r2 + 1):
            processed.add((r, c0))
            strip.append((r, c0))
        return strip

    def _find_border_regions(self, active_cells: set) -> List[set]:
        """Find connected regions of cells with borders - O(N) where N = cells with borders"""
        regions = []
        unvisited = active_cells.copy()

        while unvisited:
            # Start a new region
            start = unvisited.pop()
            region = set()
            stack = [start]

            while stack:
                cell = stack.pop()
                if cell in region:
                    continue
                region.add(cell)
                unvisited.discard(cell)

                # Add neighbors that have borders
                r, c = cell
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                for neighbor in neighbors:
                    if neighbor in unvisited:
                        stack.append(neighbor)

            regions.append(region)

        return regions

    def _find_complete_grids(self, border_map, active_cells, processed_cells):
        """Find complete grids with early termination"""
        regions = []

        # Pre-compute cells with all 4 borders (much smaller set)
        complete_border_cells = set()
        for cell in active_cells:
            if cell in processed_cells:
                continue
            borders = border_map.get(cell, {})
            if all(borders.get(side) for side in ['top', 'bottom', 'left', 'right']):
                complete_border_cells.add(cell)

        # Now only process complete_border_cells (much smaller N)
        # Group by style
        style_groups = {}
        for cell in complete_border_cells:
            borders = border_map[cell]
            styles = tuple(borders[side] for side in ['top', 'bottom', 'left', 'right'])
            if len(set(styles)) == 1:  # All sides same style
                style_groups.setdefault(styles[0], set()).add(cell)

        # Process each style group
        for style, cells in style_groups.items():
            # Don't use spatial index - just use the set directly
            available = cells.copy()  # Keep it as a set of tuples
            while available:
                # Start with the top-leftmost cell
                start_cell = min(available)
                r0, c0 = start_cell
                # Grow right
                c2 = c0
                while (r0, c2 + 1) in available:
                    c2 += 1
                # Grow down (full-width rows only)
                r2 = r0
                while True:
                    nr = r2 + 1
                    ok = True
                    for c in range(c0, c2 + 1):
                        if (nr, c) not in available:
                            ok = False
                            break
                    if ok:
                        r2 = nr
                    else:
                        break
                # Collect all cells in the rectangle
                region = []
                for r in range(r0, r2 + 1):
                    for c in range(c0, c2 + 1):
                        region.append((r, c))
                        processed_cells.add((r, c))
                        available.discard((r, c))
                regions.append((region, style))
        return regions

    def _find_boxes(self, border_map, processed_cells):
        """Find rectangular regions where only the outer edges have borders (O(N²) optimized)."""
        regions = []

        # Sort all unprocessed cells for deterministic processing
        unprocessed_list = sorted([cell for cell in border_map if cell not in processed_cells])

        for start_cell in unprocessed_list:
            if start_cell in processed_cells:
                continue

            r1, c1 = start_cell

            # Check if this cell has at least one perimeter border
            cell_borders = border_map.get(start_cell, {})
            if not any(cell_borders.get(side) for side in ['top', 'bottom', 'left', 'right']):
                continue

            # Try to grow the largest valid outside border rectangle from this cell
            best_rect = self._grow_outside_rectangle(border_map, r1, c1, processed_cells)

            if best_rect:
                box_cells, box_style = best_rect
                for cell in box_cells:
                    processed_cells.add(cell)
                regions.append((box_cells, box_style))

        return regions

    def _grow_outside_rectangle(self, border_map, r1, c1, processed_cells):
        """Grow the largest valid outside border rectangle from (r1, c1) - O(N) per cell."""

        # First, determine the potential border style from this starting cell
        initial_borders = border_map.get((r1, c1), {})
        potential_style = None
        for side in ['top', 'left', 'bottom', 'right']:
            if initial_borders.get(side):
                potential_style = initial_borders[side]
                break

        if not potential_style:
            return None

        # Find maximum extent we can grow while maintaining outside border pattern
        # Start with just this cell
        max_valid_r2, max_valid_c2 = r1, c1

        # Try to expand right and down to find the largest valid rectangle
        # First, find how far we can go right on the top edge
        c2 = c1
        while True:
            next_c = c2 + 1
            if (r1, next_c) in processed_cells:
                break
            cell_borders = border_map.get((r1, next_c), {})
            # Top edge cell must have top border with matching style
            if cell_borders.get('top') != potential_style:
                break
            c2 = next_c

        # Now find how far we can go down on the left edge
        r2 = r1
        while True:
            next_r = r2 + 1
            if (next_r, c1) in processed_cells:
                break
            cell_borders = border_map.get((next_r, c1), {})
            # Left edge cell must have left border with matching style
            if cell_borders.get('left') != potential_style:
                break
            r2 = next_r

        # Now verify and potentially shrink the rectangle to ensure it's valid
        for test_r2 in range(r2, r1 - 1, -1):
            for test_c2 in range(c2, c1 - 1, -1):
                if self._is_valid_outside_rectangle(border_map, r1, c1, test_r2, test_c2, potential_style, processed_cells):
                    # Found largest valid rectangle
                    box_cells = []
                    for r in range(r1, test_r2 + 1):
                        for c in range(c1, test_c2 + 1):
                            box_cells.append((r, c))
                    return (box_cells, potential_style)

        return None

    def _is_valid_outside_rectangle(self, border_map, r1, c1, r2, c2, style, processed_cells):
        """Check if rectangle has valid outside border pattern - O(perimeter)."""

        # Check no cells are already processed
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if (r, c) in processed_cells:
                    return False

        # Check perimeter cells have correct borders
        # Top edge
        for c in range(c1, c2 + 1):
            borders = border_map.get((r1, c), {})
            if borders.get('top') != style:
                return False

        # Bottom edge
        for c in range(c1, c2 + 1):
            borders = border_map.get((r2, c), {})
            if borders.get('bottom') != style:
                return False

        # Left edge
        for r in range(r1, r2 + 1):
            borders = border_map.get((r, c1), {})
            if borders.get('left') != style:
                return False

        # Right edge
        for r in range(r1, r2 + 1):
            borders = border_map.get((r, c2), {})
            if borders.get('right') != style:
                return False

        # Check interior cells don't have all 4 borders (would be BORDER_ALL)
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                borders = border_map.get((r, c), {})
                if all(borders.get(side) == style for side in ['top', 'bottom', 'left', 'right']):
                    return False

        return True

    def _find_matching_rectangle(
        self,
        cells: Dict[Tuple[int, int], Any],
        r0: int,
        c0: int,
        processed: set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Find the largest rectangle of contiguous cells contained in `cells`,
        starting from (r0, c0), growing right first then down (ensuring full rows).
        """
        if (r0, c0) not in cells or (r0, c0) in processed:
            return []

        # Grow right
        c2 = c0
        while (r0, c2 + 1) in cells and (r0, c2 + 1) not in processed:
            c2 += 1

        # Grow down (full-width rows only)
        r2 = r0
        while True:
            nr = r2 + 1
            ok = True
            for c in range(c0, c2 + 1):
                if (nr, c) not in cells or (nr, c) in processed:
                    ok = False
                    break
            if ok:
                r2 = nr
            else:
                break

        rect: List[Tuple[int, int]] = []
        for r in range(r0, r2 + 1):
            for c in range(c0, c2 + 1):
                processed.add((r, c))
                rect.append((r, c))
        return rect

