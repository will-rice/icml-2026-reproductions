from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from next_action_pred_eval.utils.cell_utils import get_cell_address, parse_cell, expand_range
from openpyxl.utils import range_boundaries

from next_action_pred_eval.evaluation.state_comparator import ComparisonResult, PropertyDifference


# Mapping from operation names to property paths used in state comparison
OPERATION_TO_PROPERTY = {
    "FONT_BOLD": "Format.font.bold",
    "FONT_ITALIC": "Format.font.italic",
    "FONT_SIZE": "Format.font.size",
    "FONT_NAME": "Format.font.name",
    "FONT_COLOR": "Format.font.color",
    "FONT_UNDERLINE": "Format.font.underline",
    "FILL_COLOR": "Format.fill.fgColor",
    "NUMBER_FORMAT": "number_format",
    "ALIGN_HORIZONTAL": "Format.horizontalAlignment",
    "ALIGN_VERTICAL": "Format.verticalAlignment",
    "WRAP_TEXT": "Format.wrapText",
    "TEXT_ORIENTATION": "Format.textOrientation",
    "BORDER_LEFT": "Format.borders.left",
    "BORDER_RIGHT": "Format.borders.right",
    "BORDER_TOP": "Format.borders.top",
    "BORDER_BOTTOM": "Format.borders.bottom",
    "BORDER_ALL": "Format.borders",
    "BORDER_OUTSIDE": "Format.borders",
    "MERGE": "merged_cells",
    "INPUT": "value",
    "VALUE": "value",
}

# Property paths that correspond to PRIME_VISIBLE_OPS (SetValue, SetFormula, SetInput, SetBorder, SetFillColor)
# These are the only properties that should affect validation pass/fail
PRIME_VISIBLE_PROPERTY_PREFIXES = (
    "value",  # SetValue, SetFormula, SetInput
    "Format.borders",  # SetBorder (all border variants)
    "Format.fill.fgColor",  # SetFillColor
)


def is_prime_visible_property(property_path: str) -> bool:
    """Check if a property path corresponds to a PRIME_VISIBLE_OPS operation."""
    return property_path.startswith(PRIME_VISIBLE_PROPERTY_PREFIXES)


def filter_to_prime_visible(differences: List[PropertyDifference]) -> List[PropertyDifference]:
    """Filter differences to only include PRIME_VISIBLE_OPS properties."""
    return [d for d in differences if is_prime_visible_property(d.property_path)]


def _cell_in_range(cell: str, range_str: str) -> bool:
    """Check if a cell address is within a range."""
    try:
        min_col, min_row, max_col, max_row = range_boundaries(range_str)
        cell_row, cell_col = parse_cell(cell)
        return min_row <= cell_row <= max_row and min_col <= cell_col <= max_col
    except Exception:
        return cell == range_str  # Single cell comparison


@dataclass
class IgnoreDeclarations:
    """
    Represents the IGNORE section from LLM output.
    Maps operation names to lists of ranges that should be ignored.

    Example:
        IGNORE:
        FONT_SIZE: [B1:E1, M16:M17]
        NUMBER_FORMAT: [C9, E14]
    """
    declarations: Dict[str, List[str]] = field(default_factory=dict)

    def covers(self, sheet: str, cell: str, property_path: str) -> bool:
        """Check if this ignore declaration covers a given cell+property."""
        for op_name, ranges in self.declarations.items():
            # Get the property path for this operation
            mapped_prop = OPERATION_TO_PROPERTY.get(op_name.upper(), op_name)

            # Check if property matches
            if not (property_path.startswith(mapped_prop) or property_path == mapped_prop):
                continue

            # Check if cell is in any of the ranges
            for range_str in ranges:
                if _cell_in_range(cell, range_str):
                    return True

        return False

    def is_empty(self) -> bool:
        return not self.declarations or all(len(v) == 0 for v in self.declarations.values())


@dataclass
class CorrectionDeclaration:
    """
    Represents an intentional content correction from LLM output.

    Example:
        CORRECTIONS:
        - Sheet1!A5: "Totla" -> "Total" (spelling fix)
        - Sheet1!B3: "Janury" -> "January" (typo)
    """
    cell_ref: str  # e.g., "Sheet1!A5"
    original_value: str  # e.g., "Totla"
    corrected_value: str  # e.g., "Total"
    reason: str  # e.g., "spelling fix"

    def covers(self, sheet: str, cell: str, property_path: str) -> bool:
        """
        Check if this correction covers a given cell+property.
        Corrections apply to value/content properties.
        """
        # Parse the cell reference
        if "!" in self.cell_ref:
            corr_sheet, corr_cell = self.cell_ref.split("!", 1)
        else:
            corr_sheet = sheet
            corr_cell = self.cell_ref

        if corr_sheet != sheet:
            return False

        # Check if cell matches (corrections are typically single cells)
        if corr_cell != cell:
            return False

        # Corrections typically apply to value properties, but can also cover formatting
        # Allow any property to be covered - the correction description indicates what changed
        return True


@dataclass
class CorrectionDeclarations:
    """Collection of correction declarations."""
    corrections: List[CorrectionDeclaration] = field(default_factory=list)

    def covers(self, sheet: str, cell: str, property_path: str) -> bool:
        """Check if any correction covers this cell+property."""
        return any(c.covers(sheet, cell, property_path) for c in self.corrections)

    def is_empty(self) -> bool:
        return not self.corrections


# Keep SkipDeclaration for backward compatibility
@dataclass
class SkipDeclaration:
    """Represents a #SKIP directive from the LLM (legacy format)."""
    range_str: str  # e.g., "Sheet1!A1:B2"
    property_name: str  # e.g., "FONT_BOLD", "NUMBER_FORMAT", or "*" for all
    reason: str

    def covers(self, sheet: str, cell: str, property_path: str) -> bool:
        """Check if this skip declaration covers a given cell+property."""
        # Parse the skip range
        if "!" in self.range_str:
            skip_sheet, skip_range = self.range_str.split("!", 1)
        else:
            skip_sheet = sheet  # assume same sheet if not specified
            skip_range = self.range_str

        if skip_sheet != sheet:
            return False

        # Check if cell is within the skip range
        if not _cell_in_range(cell, skip_range):
            return False

        # Check property match
        if self.property_name == "*":
            return True

        mapped_prop = OPERATION_TO_PROPERTY.get(self.property_name.upper(), self.property_name)
        return property_path.startswith(mapped_prop) or property_path == mapped_prop


@dataclass
class ParsedLLMOutput:
    operations: List[str]
    skip_declarations: List[SkipDeclaration] = field(default_factory=list)
    ignore_declarations: IgnoreDeclarations = field(default_factory=IgnoreDeclarations)
    correction_declarations: CorrectionDeclarations = field(default_factory=CorrectionDeclarations)
    human_enough: bool = False
    rationale: str = ""
    raw_operations_block: str = ""
    errors: List[str] = field(default_factory=list)


@dataclass
class JudgeVerdict:
    is_human: bool
    rationale: str
    raw_text: str


def parse_llm_response(text: str, keyword: str) -> ParsedLLMOutput:
    errors: List[str] = []
    human = _extract_boolean(text, keyword, default=True)
    rationale = _extract_rationale(text)
    block = _extract_operations_block(text)
    ignore_declarations = _extract_ignore_declarations(text)
    correction_declarations = _extract_correction_declarations(text)

    operations: List[str] = []
    skip_declarations: List[SkipDeclaration] = []

    if block:
        for line in block.splitlines():
            clean = line.strip()
            if not clean:
                continue
            # Parse #SKIP lines: #SKIP | range | property | reason (legacy format)
            if clean.upper().startswith("#SKIP"):
                skip = _parse_skip_line(clean)
                if skip:
                    skip_declarations.append(skip)
                continue
            # Skip other comments
            if clean.startswith("#"):
                continue
            if " | " not in clean:
                continue
            operations.append(clean)
    else:
        errors.append("Missing operations code block")

    if not operations:
        errors.append("No operations detected in response")

    return ParsedLLMOutput(
        operations=operations,
        skip_declarations=skip_declarations,
        ignore_declarations=ignore_declarations,
        correction_declarations=correction_declarations,
        human_enough=human,
        rationale=rationale,
        raw_operations_block=block or "",
        errors=errors,
    )


def _extract_ignore_declarations(text: str) -> IgnoreDeclarations:
    """
    Extract the IGNORE section from LLM response.

    Format:
        IGNORE:
        FONT_SIZE: [B1:E1, M16:M17, H53:H54]
        NUMBER_FORMAT: [C9, E14, E21, C28]

    Returns:
        IgnoreDeclarations with parsed operation->ranges mapping
    """
    declarations: Dict[str, List[str]] = {}

    # Find the IGNORE: section
    ignore_match = re.search(r'IGNORE\s*:', text, re.IGNORECASE)
    if not ignore_match:
        return IgnoreDeclarations(declarations=declarations)

    # Get text after IGNORE:
    remainder = text[ignore_match.end():]

    # Parse each line until we hit something that's not an operation: [ranges] pattern
    for line in remainder.splitlines():
        line = line.strip()
        if not line:
            continue

        # Match pattern: OPERATION_NAME: [range1, range2, ...]
        match = re.match(r'^([A-Z_]+)\s*:\s*\[(.*?)\]', line, re.IGNORECASE)
        if match:
            op_name = match.group(1).upper()
            ranges_str = match.group(2)
            # Parse ranges - split by comma and strip whitespace
            ranges = [r.strip() for r in ranges_str.split(',') if r.strip()]
            if ranges:
                if op_name not in declarations:
                    declarations[op_name] = []
                declarations[op_name].extend(ranges)
        else:
            # Stop parsing if we hit a line that doesn't match the pattern
            # (unless it's empty or a comment)
            if line and not line.startswith('#'):
                break

    return IgnoreDeclarations(declarations=declarations)


def _extract_correction_declarations(text: str) -> CorrectionDeclarations:
    """
    Extract the CORRECTIONS section from LLM response.

    Format:
        CORRECTIONS:
        - Sheet1!A5: "Totla" -> "Total" (spelling fix)
        - Sheet1!B3: "Janury" -> "January" (typo)
        - Sheet1!C1: fill #FF0000 -> #CC0000 (color coordination)

    Returns:
        CorrectionDeclarations with parsed corrections
    """
    corrections: List[CorrectionDeclaration] = []

    # Find the CORRECTIONS: section
    corrections_match = re.search(r'CORRECTIONS\s*:', text, re.IGNORECASE)
    if not corrections_match:
        return CorrectionDeclarations(corrections=corrections)

    # Get text after CORRECTIONS:
    remainder = text[corrections_match.end():]

    # Parse each line that starts with - (bullet point)
    for line in remainder.splitlines():
        line = line.strip()
        if not line:
            continue

        # Stop if we hit another section header (e.g., IGNORE:, RATIONALE:, etc.)
        if re.match(r'^[A-Z]+\s*:', line):
            break

        # Skip non-bullet lines
        if not line.startswith('-'):
            continue

        # Remove the leading dash
        line = line[1:].strip()

        # Parse format: Sheet1!A5: "old" -> "new" (reason)
        # Also handle: Sheet1!A5: old -> new (reason) without quotes
        # And: Sheet1!A5: fill #XXX -> #YYY (reason) for formatting

        # First, extract the cell reference (everything before the first colon after sheet!cell)
        cell_match = re.match(r'^([A-Za-z0-9_]+![A-Z]+\d+)\s*:', line)
        if not cell_match:
            # Try without sheet prefix
            cell_match = re.match(r'^([A-Z]+\d+)\s*:', line)

        if not cell_match:
            continue

        cell_ref = cell_match.group(1)
        rest = line[cell_match.end():].strip()

        # Parse the "old" -> "new" (reason) part
        # Support both quoted and unquoted values
        arrow_match = re.search(r'(.+?)\s*->\s*(.+?)(?:\s*\((.+?)\)\s*)?$', rest)
        if arrow_match:
            original_value = arrow_match.group(1).strip().strip('"\'')
            corrected_value = arrow_match.group(2).strip().strip('"\'')
            # Remove trailing parenthetical from corrected_value if reason wasn't captured
            if '(' in corrected_value and arrow_match.group(3) is None:
                paren_idx = corrected_value.rfind('(')
                reason = corrected_value[paren_idx+1:].rstrip(')')
                corrected_value = corrected_value[:paren_idx].strip().strip('"\'')
            else:
                reason = arrow_match.group(3) or ""

            corrections.append(CorrectionDeclaration(
                cell_ref=cell_ref,
                original_value=original_value,
                corrected_value=corrected_value,
                reason=reason.strip(),
            ))

    return CorrectionDeclarations(corrections=corrections)


def _parse_skip_line(line: str) -> SkipDeclaration | None:
    """
    Parse a #SKIP directive line.

    Format: #SKIP | range | property | reason
    Examples:
        #SKIP | Sheet1!M16:M17 | FONT_BOLD | stray formatting on empty column
        #SKIP | Sheet1!C9 | NUMBER_FORMAT | text cell, not a date
        #SKIP | Sheet1!A1:B5 | * | entire range intentionally removed
    """
    # Remove the #SKIP prefix
    remainder = line[5:].strip()  # Remove "#SKIP"
    if remainder.startswith("|"):
        remainder = remainder[1:].strip()

    parts = [p.strip() for p in remainder.split("|")]
    if len(parts) < 2:
        return None

    range_str = parts[0]
    property_name = parts[1] if len(parts) > 1 else "*"
    reason = parts[2] if len(parts) > 2 else ""

    return SkipDeclaration(range_str=range_str, property_name=property_name, reason=reason)


def parse_judge_response(text: str, keyword: str) -> JudgeVerdict:
    verdict = _extract_boolean(text, keyword, default=False)
    rationale = _extract_rationale(text)
    return JudgeVerdict(is_human=verdict, rationale=rationale, raw_text=text)


def _extract_boolean(text: str, keyword: str, default: bool = False) -> bool:
    if not keyword:
        return default
    pattern = re.compile(rf"{re.escape(keyword)}\s*:\s*(yes|true)", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return default
    return match.group(1).lower() in {"yes", "true"}


def _extract_rationale(text: str) -> str:
    match = re.search(r"RATIONALE\s*:(.*)", text, re.IGNORECASE)
    if not match:
        return ""
    remainder = match.group(1).strip()
    return remainder.split("SEQUENCE", 1)[0].strip()


def _extract_operations_block(text: str) -> str | None:
    # First try: fenced code block with triple backticks
    pattern = re.compile(r"```(?:ops|text)?\n(.*?)```", re.DOTALL)
    for block in pattern.findall(text):
        if " | " in block:
            return block.strip()

    # Fallback: bare "ops" keyword followed by operations (no triple backticks)
    # Matches "ops\n<content>" where content has operations (lines with " | ")
    bare_pattern = re.compile(r"^ops\n((?:.*\|.*\n?)+)", re.MULTILINE)
    match = bare_pattern.search(text)
    if match:
        block = match.group(1)
        if " | " in block:
            return block.strip()

    return None


def filter_differences_by_skips(
    differences: List[PropertyDifference],
    skip_declarations: List[SkipDeclaration],
    ignore_declarations: IgnoreDeclarations | None = None,
    correction_declarations: CorrectionDeclarations | None = None,
) -> Tuple[List[PropertyDifference], List[PropertyDifference]]:
    """
    Filter comparison differences based on skip/ignore/correction declarations.

    Args:
        differences: List of PropertyDifference from state comparison
        skip_declarations: List of SkipDeclaration from parsed LLM output (legacy format)
        ignore_declarations: IgnoreDeclarations from parsed LLM output (new format)
        correction_declarations: CorrectionDeclarations for intentional content changes

    Returns:
        Tuple of (remaining_differences, skipped_differences)
        - remaining_differences: Differences NOT covered by any declaration (real mismatches)
        - skipped_differences: Differences that were intentionally skipped/ignored/corrected
    """
    has_declarations = (
        skip_declarations or
        (ignore_declarations and not ignore_declarations.is_empty()) or
        (correction_declarations and not correction_declarations.is_empty())
    )
    if not has_declarations:
        return differences, []

    remaining: List[PropertyDifference] = []
    skipped: List[PropertyDifference] = []

    for diff in differences:
        # TP (true positives) always pass through unchanged
        if diff.match_type == "TP":
            remaining.append(diff)
            continue

        # Check if any skip declaration covers this difference (legacy format)
        is_skipped = any(
            skip.covers(diff.sheet, diff.cell, diff.property_path)
            for skip in skip_declarations
        )

        # Check if ignore declarations cover this difference (new format)
        if not is_skipped and ignore_declarations:
            is_skipped = ignore_declarations.covers(diff.sheet, diff.cell, diff.property_path)

        # Check if correction declarations cover this difference (intentional content changes)
        if not is_skipped and correction_declarations:
            is_skipped = correction_declarations.covers(diff.sheet, diff.cell, diff.property_path)

        if is_skipped:
            skipped.append(diff)
        else:
            remaining.append(diff)

    return remaining, skipped


def count_unskipped_mismatches(
    differences: List[PropertyDifference],
    skip_declarations: List[SkipDeclaration],
    ignore_declarations: IgnoreDeclarations | None = None,
    correction_declarations: CorrectionDeclarations | None = None,
) -> int:
    """
    Count mismatches that are NOT covered by skip/ignore/correction declarations.

    This is used to determine if the sequence should be considered valid
    (only skipped mismatches) or invalid (has real mismatches).
    """
    remaining, _ = filter_differences_by_skips(
        differences, skip_declarations, ignore_declarations, correction_declarations
    )
    return sum(1 for d in remaining if d.match_type != "TP")


def summarize_sequence_diff(candidate: Sequence[str], reference: Sequence[str], limit: int | None = None) -> str:
    cand_set = set(candidate)
    ref_set = set(reference)
    added = sorted(cand_set - ref_set)
    removed = sorted(ref_set - cand_set)
    def _format(values: List[str]) -> str:
        if not values:
            return "none"
        if limit is None or limit >= len(values):
            return ", ".join(values)
        preview = ", ".join(values[:limit])
        remainder = len(values) - limit
        if remainder > 0:
            preview += f" (+{remainder} more)"
        return preview

    return f"added {len(added)} ({_format(added)}); removed {len(removed)} ({_format(removed)})"


def summarize_comparison(result: ComparisonResult | None, cell_limit: int | None = None) -> str:
    if result is None:
        return "Comparison skipped"
    cells = _collect_mismatched_cells(result.differences)
    if not cells:
        return "All compared cells match the workbook"
    if cell_limit is None or cell_limit >= len(cells):
        preview = ", ".join(cells)
    else:
        preview = ", ".join(cells[:cell_limit])
    return f"{len(cells)} mismatched cells: {preview}"


def _collect_mismatched_cells(differences: Iterable[PropertyDifference]) -> List[str]:
    seen: List[str] = []
    for diff in differences:
        if diff.match_type == "TP":
            continue
        cell_label = f"{diff.sheet}!{diff.cell}"
        if cell_label not in seen:
            seen.append(cell_label)
    return seen


def build_retry_hint(result: ComparisonResult | None) -> str:
    if result is None:
        return "Provide a parseable operations block."
    cells = _collect_mismatched_cells(result.differences)
    if cells:
        sample = ", ".join(cells[:5])
        return f"Cells still differ ({len(cells)}): {sample}. Align them with the workbook."
    return "Focus on a clearer, human narrative ordering (group related ranges and respect reading order)."


def list_mismatched_cells(result: ComparisonResult | None) -> List[str]:
    if result is None:
        return []
    return _collect_mismatched_cells(result.differences)


def build_mismatch_report(
    differences: List[PropertyDifference] | None,
    result: ComparisonResult | None = None,
) -> str | None:
    """
    Build a compressed mismatch report showing only unique cells and property names.

    Args:
        differences: List of PropertyDifference (already filtered by ignore/skip)
        result: Optional ComparisonResult for backward compatibility
    """
    # Backward compatibility: if differences is a ComparisonResult, extract differences
    if result is not None and differences is None:
        differences = [d for d in result.differences if d.match_type != "TP"]

    if differences is None:
        return None

    # Filter out TPs
    mismatches = [d for d in differences if d.match_type != "TP"]
    if not mismatches:
        return "All cells match the workbook."

    # Collect unique cells and properties
    cells: set[str] = set()
    properties: set[str] = set()
    for diff in mismatches:
        cells.add(f"{diff.sheet}!{diff.cell}")
        properties.add(diff.property_path)

    lines: List[str] = [
        f"Mismatched cells: {{{', '.join(sorted(cells))}}}",
        f"Mismatched properties: {{{', '.join(sorted(properties))}}}",
    ]
    return "\n".join(lines)


def build_completion_moves(differences: List[PropertyDifference], limit: int | None = None) -> List[str]:
    """
    Build list of possible completion moves from FN differences (missing operations).

    These are operations that need to be added to reach parity.
    Groups by operation type and ranges.
    """
    fns = [d for d in differences if d.match_type in ("FN", "MISMATCH")]
    if not fns:
        return []

    # Group by property path and true value (what should be there)
    grouped: Dict[Tuple[str, str], List[str]] = {}
    for diff in fns:
        op_name = _property_path_to_op_name(diff.property_path)
        val_str = _format_value(diff.true_value)
        key = (op_name, val_str)
        grouped.setdefault(key, []).append(diff.cell)

    # Build operation suggestions
    suggestions: List[str] = []
    for (op_name, val_str), cells in sorted(grouped.items()):
        compressed = _compress_cells(cells)
        for range_str in compressed:
            suggestions.append(f"{op_name} | Sheet1!{range_str} | {val_str}")

    if limit and len(suggestions) > limit:
        return suggestions[:limit]
    return suggestions


def build_mismatch_operations(differences: List[PropertyDifference], limit: int | None = None) -> List[str]:
    """
    Build list of possibly incorrect operations from FP differences (extra operations).

    These are operations that were applied but shouldn't have been.
    Groups by operation type and ranges.
    """
    fps = [d for d in differences if d.match_type in ("FP", "MISMATCH")]
    if not fps:
        return []

    # Group by property path and predicted value
    grouped: Dict[Tuple[str, str], List[str]] = {}
    for diff in fps:
        op_name = _property_path_to_op_name(diff.property_path)
        val_str = _format_value(diff.predicted_value)
        key = (op_name, val_str)
        grouped.setdefault(key, []).append(diff.cell)

    # Build operation suggestions for removal
    suggestions: List[str] = []
    for (op_name, val_str), cells in sorted(grouped.items()):
        compressed = _compress_cells(cells)
        for range_str in compressed:
            suggestions.append(f"{op_name} | Sheet1!{range_str} | {val_str}")

    if limit and len(suggestions) > limit:
        return suggestions[:limit]
    return suggestions


def _property_path_to_op_name(property_path: str) -> str:
    """Convert a property path to an operation name."""
    # Reverse mapping from OPERATION_TO_PROPERTY
    reverse_map = {
        "Format.font.bold": "FONT_BOLD",
        "Format.font.italic": "FONT_ITALIC",
        "Format.font.size": "FONT_SIZE",
        "Format.font.name": "FONT_NAME",
        "Format.font.color": "FONT_COLOR",
        "Format.font.underline": "FONT_UNDERLINE",
        "Format.fill.fgColor": "FILL_COLOR",
        "number_format": "NUMBER_FORMAT",
        "Format.horizontalAlignment": "ALIGN_HORIZONTAL",
        "Format.verticalAlignment": "ALIGN_VERTICAL",
        "Format.wrapText": "WRAP_TEXT",
        "Format.textOrientation": "TEXT_ORIENTATION",
        "Format.borders.left": "BORDER_LEFT",
        "Format.borders.right": "BORDER_RIGHT",
        "Format.borders.top": "BORDER_TOP",
        "Format.borders.bottom": "BORDER_BOTTOM",
        "Format.borders": "BORDER",
        "merged_cells": "MERGE",
        "value": "INPUT",
    }
    # Check for exact match first
    if property_path in reverse_map:
        return reverse_map[property_path]
    # Check for prefix match
    for path, op in reverse_map.items():
        if property_path.startswith(path):
            return op
    return property_path


def _format_value(value: Any) -> str:
    """Format a value for display in operation suggestions."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return str(value)


def summarize_extra_properties(result: ComparisonResult | None, limit: int = 5) -> str:
    if result is None:
        return "Comparison unavailable"
    extras = [d for d in result.differences if d.match_type in ("FP", "MISMATCH")]
    if not extras:
        return "No extra cells detected"

    grouped: dict[tuple[str, str, str], List[str]] = {}
    for diff in extras:
        key = (diff.sheet, diff.property_path, str(diff.predicted_value))
        grouped.setdefault(key, []).append(diff.cell)

    summaries: List[str] = []
    for (sheet, prop, pred_val), cells in grouped.items():
        compressed = _compress_cells(cells)
        sample = ", ".join(compressed[:2])
        if len(compressed) > 2:
            sample += ", ..."
        summaries.append(f"{sheet}!{sample} — {prop} predicted {pred_val}")

    if not summaries:
        return "No extra cells detected"
    if len(summaries) > limit:
        return "; ".join(summaries[:limit]) + "; ..."
    return "; ".join(summaries)


def _compress_cells(cells: Iterable[str]) -> List[str]:
    """Compress a list of cells (or ranges) into minimal range notation.

    Handles both single cell addresses (e.g., 'A1') and ranges (e.g., 'A1:B2').
    """
    coords = []
    for cell in cells:
        # Handle both single cells and ranges
        if ":" in cell:
            # It's a range - expand it to individual cells
            for row, col in expand_range(cell):
                coords.append((row, col))
        else:
            row, col = parse_cell(cell)
            coords.append((row, col))
    coords.sort()
    if not coords:
        return []

    ranges: List[str] = []
    start_row, start_col = coords[0]
    prev_row, prev_col = coords[0]

    def _close_range(sr: int, sc: int, er: int, ec: int) -> None:
        if sr == er and sc == ec:
            ranges.append(get_cell_address(sr, sc))
        else:
            ranges.append(f"{get_cell_address(sr, sc)}:{get_cell_address(er, ec)}")

    for row, col in coords[1:]:
        same_row = row == prev_row and col == prev_col + 1
        same_col = col == prev_col and row == prev_row + 1
        if same_row or same_col:
            prev_row, prev_col = row, col
            continue
        _close_range(start_row, start_col, prev_row, prev_col)
        start_row, start_col = row, col
        prev_row, prev_col = row, col

    _close_range(start_row, start_col, prev_row, prev_col)
    return ranges
