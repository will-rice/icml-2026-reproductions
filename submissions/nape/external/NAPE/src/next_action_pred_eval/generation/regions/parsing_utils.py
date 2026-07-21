"""
Region analysis parsing and validation utilities.

Provides functions for:
- Extracting JSON from LLM responses
- Parsing and validating structured region output
- Formatting validation errors for retry prompts
- Coverage validation against sheet operations
- Formula range extraction from operations
"""

import json
import re
from collections import deque
from typing import Dict, Any, Optional, List, Tuple, Set

from pydantic import ValidationError
from openpyxl.utils import range_boundaries

from next_action_pred_eval.generation.regions.models import (
    StructuredRegionOutput,
    RegionModel,
    PastedRange,
    SimilarlyFormattedRegions,
)
from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.operations import SetValue, SetFormula, SetInput, SetBorder, SetFillColor
from next_action_pred_eval.utils.cell_utils import expand_range, get_range_string


def get_formula_ranges(sheet_operations: List[Operation], max_sheet_limit: int = 100) -> List[str]:
    """
    Extract cells containing formulas and merge them into rectangular ranges.

    Args:
        sheet_operations: List of operations for the sheet
        max_sheet_limit: Maximum sheet dimension to consider

    Returns:
        List of range strings representing formula regions
    """
    formula_cells: Set[Tuple[int, int]] = set()

    # Collect all cells with formulas
    for op in sheet_operations:
        if isinstance(op, SetInput):
            # Check if the value is a formula (starts with =)
            value = op.value

            # Handle single formula value
            if isinstance(value, str) and value.startswith('='):
                bounds = range_boundaries(op.cell_range.range)
                if all(dim <= max_sheet_limit for dim in bounds):
                    cells = expand_range(op.cell_range.range)
                    formula_cells.update(cells)

            # Handle 2D array with formulas
            elif isinstance(value, list) and all(isinstance(row, list) for row in value):
                bounds = range_boundaries(op.cell_range.range)
                if all(dim <= max_sheet_limit for dim in bounds):
                    min_col, min_row, max_col, max_row = bounds
                    for row_offset, row_data in enumerate(value):
                        for col_offset, cell_value in enumerate(row_data):
                            if isinstance(cell_value, str) and cell_value.startswith('='):
                                row = min_row + row_offset
                                col = min_col + col_offset
                                if row <= max_row and col <= max_col:
                                    formula_cells.add((row, col))

            # Handle 1D array with formulas
            elif isinstance(value, list):
                bounds = range_boundaries(op.cell_range.range)
                if all(dim <= max_sheet_limit for dim in bounds):
                    min_col, min_row, max_col, max_row = bounds
                    if max_row == min_row:  # Row vector
                        for col_offset, cell_value in enumerate(value):
                            if isinstance(cell_value, str) and cell_value.startswith('='):
                                col = min_col + col_offset
                                if col <= max_col:
                                    formula_cells.add((min_row, col))
                    else:  # Column vector
                        for row_offset, cell_value in enumerate(value):
                            if isinstance(cell_value, str) and cell_value.startswith('='):
                                row = min_row + row_offset
                                if row <= max_row:
                                    formula_cells.add((row, min_col))

    if not formula_cells:
        return []

    # Merge adjacent formula cells into rectangular regions
    return _create_bounding_boxes_from_cells(formula_cells)


def extract_json_from_response(response_text: str) -> Optional[str]:
    """
    Extract JSON content from LLM response.
    Handles cases where JSON is wrapped in markdown code blocks or has extra text.

    Args:
        response_text: Raw text from LLM response

    Returns:
        Extracted JSON string or None if not found
    """
    if not response_text:
        return None

    # Try to find JSON in markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    if matches:
        return matches[0]

    # Try to find JSON object directly (look for outermost braces)
    # Find the first { and last }
    first_brace = response_text.find('{')
    last_brace = response_text.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return response_text[first_brace:last_brace + 1]

    return None


def parse_structured_output(response_text: str) -> Tuple[Optional[StructuredRegionOutput], List[str]]:
    """
    Parse and validate the structured region output from LLM response.

    Args:
        response_text: Raw text from LLM response

    Returns:
        Tuple of (parsed_output, validation_errors)
        - parsed_output: StructuredRegionOutput object if valid, None otherwise
        - validation_errors: List of error messages
    """
    errors = []

    # Extract JSON
    json_str = extract_json_from_response(response_text)
    if not json_str:
        errors.append("No valid JSON found in response")
        return None, errors

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        errors.append(f"JSON parsing error: {str(e)}")
        return None, errors

    # Validate with Pydantic
    try:
        # Convert string keys to integers for region_dependencies
        if 'region_dependencies' in data and isinstance(data['region_dependencies'], dict):
            data['region_dependencies'] = {
                int(k): int(v) for k, v in data['region_dependencies'].items()
            }

        output = StructuredRegionOutput(**data)
        return output, []
    except ValidationError as e:
        for error in e.errors():
            field = " -> ".join(str(x) for x in error['loc'])
            errors.append(f"Validation error in {field}: {error['msg']}")
        return None, errors
    except Exception as e:
        errors.append(f"Unexpected error during validation: {str(e)}")
        return None, errors


def format_validation_errors(errors: List[str], previous_errors: List[List[str]] = None) -> str:
    """
    Format validation errors for inclusion in retry prompt.

    Args:
        errors: List of current validation errors
        previous_errors: List of error lists from previous attempts

    Returns:
        Formatted error message string
    """
    message = "\n**VALIDATION ERRORS FROM PREVIOUS ATTEMPT:**\n"
    message += "\n".join(f"- {error}" for error in errors)

    if previous_errors:
        message += "\n\n**Errors from earlier attempts:**\n"
        for i, prev_errs in enumerate(previous_errors[-2:], 1):  # Show last 2 attempts
            message += f"\nAttempt -{len(previous_errors) - i + 1}:\n"
            message += "\n".join(f"  - {error}" for error in prev_errs)

    message += "\n\nPlease fix these issues and provide a corrected JSON output."
    return message


def create_retry_prompt(
    original_prompt: str,
    response_text: str,
    errors: List[str],
    previous_errors: List[List[str]] = None
) -> str:
    """
    Create a retry prompt that includes the original prompt and validation errors.

    Args:
        original_prompt: The original prompt sent to the model
        response_text: The response that failed validation
        errors: List of validation errors
        previous_errors: List of error lists from previous attempts

    Returns:
        Retry prompt string
    """
    retry_prompt = original_prompt
    retry_prompt += "\n\n" + "="*80 + "\n"
    retry_prompt += format_validation_errors(errors, previous_errors)

    if response_text:
        # Show a snippet of the previous response for context
        json_str = extract_json_from_response(response_text)
        if json_str:
            try:
                parsed = json.loads(json_str)
                formatted = json.dumps(parsed, indent=2)
                retry_prompt += "\n\n**Your previous response:**\n```json\n"
                # Truncate if too long
                if len(formatted) > 1000:
                    retry_prompt += formatted[:500] + "\n...\n" + formatted[-500:]
                else:
                    retry_prompt += formatted
                retry_prompt += "\n```"
            except Exception:
                pass

    return retry_prompt


def validate_output_structure(output: StructuredRegionOutput) -> List[str]:
    """
    Additional semantic validation beyond Pydantic model validation.

    Args:
        output: The parsed StructuredRegionOutput

    Returns:
        List of warning/error messages (empty if all valid)
    """
    warnings = []

    # Check if regions list is empty when it probably shouldn't be
    if len(output.regions) == 0:
        warnings.append("No regions identified - is the sheet truly empty?")

    # Check for suspicious region sizes (very small or very large)
    for region in output.regions:
        min_col, min_row, max_col, max_row = range_boundaries(region.range)
        num_cells = (max_row - min_row + 1) * (max_col - min_col + 1)

        if num_cells > 10000:
            warnings.append(
                f"Region {region.id} is very large ({num_cells} cells) - "
                f"consider if it should be split"
            )

    # Check for orphaned dependencies
    region_ids = {r.id for r in output.regions}
    for dep_id, depends_on in output.region_dependencies.items():
        # Check if dependency makes spatial sense
        dep_region = next(r for r in output.regions if r.id == dep_id)
        depends_region = next(r for r in output.regions if r.id == depends_on)

        dep_bounds = dep_region.get_boundaries()
        depends_bounds = depends_region.get_boundaries()

        # If dependent region is spatially before the dependency, that's suspicious
        if dep_bounds[1] < depends_bounds[1]:  # dep starts before dependency
            warnings.append(
                f"Suspicious dependency: Region {dep_id} starts before "
                f"region {depends_on} but depends on it"
            )

    # No additional validation needed for pasted_ranges and similarly_formatted_regions
    # as their validation is already handled by the Pydantic models

    return warnings


def get_region_summary(output: StructuredRegionOutput) -> str:
    """
    Create a human-readable summary of the parsed output.

    Args:
        output: The parsed StructuredRegionOutput

    Returns:
        Summary string
    """
    summary = []
    summary.append(f"Found {len(output.regions)} region(s)")

    if output.regions:
        summary.append("\nRegions:")
        for region in output.regions:
            summary.append(f"  - Region {region.id}: {region.type} at {region.range}")
            if region.closing_operations:
                summary.append(f"    Closing operations: {len(region.closing_operations)}")

    if output.region_dependencies:
        summary.append(f"\nFound {len(output.region_dependencies)} dependency(ies)")
        for dep_id, depends_on in output.region_dependencies.items():
            summary.append(f"  - Region {dep_id} depends on Region {depends_on}")

    if output.pasted_ranges:
        summary.append(f"\nFound {len(output.pasted_ranges)} pasted range(s)")
        for pasted in output.pasted_ranges:
            summary.append(f"  - {pasted.paste_nature} paste at {pasted.range}")

    if output.similarly_formatted_regions:
        summary.append(f"\nFound {len(output.similarly_formatted_regions)} similarly formatted region group(s)")
        for group in output.similarly_formatted_regions:
            summary.append(f"  - {group.format_paste_type}: {', '.join(group.similar_regions)}")

    return "\n".join(summary)


def validate_coverage(
    output: StructuredRegionOutput,
    sheet_operations: List[Operation],
    max_sheet_limit: int = 100
) -> Tuple[bool, List[str], Set[Tuple[int, int]]]:
    """
    Validate that all cells from operations are covered by at least one region.

    Args:
        output: The parsed StructuredRegionOutput
        sheet_operations: List of operations for the sheet
        max_sheet_limit: Maximum sheet dimension to consider

    Returns:
        Tuple of:
        - is_covered: True if all cells are covered, False otherwise
        - uncovered_ranges: List of range strings for uncovered regions (bounding boxes)
        - uncovered_cells: Set of (row, col) tuples for uncovered cells
    """
    # Get all cells from operations (same filter as find_bounding_boxes)
    operation_cells: Set[Tuple[int, int]] = set()

    for op in sheet_operations:
        if isinstance(op, (SetValue, SetFormula, SetInput, SetBorder, SetFillColor)):
            # Check if within limits
            bounds = range_boundaries(op.cell_range.range)
            if all(dim <= max_sheet_limit for dim in bounds):
                cells = expand_range(op.cell_range.range)
                operation_cells.update(cells)

    if not operation_cells:
        # No cells to cover
        return True, [], set()

    # Get all cells covered by regions
    covered_cells: Set[Tuple[int, int]] = set()

    for region in output.regions:
        region_cells = expand_range(region.range)
        covered_cells.update(region_cells)

    # Find uncovered cells
    uncovered_cells = operation_cells - covered_cells

    if not uncovered_cells:
        return True, [], set()

    # Create bounding boxes for uncovered cells using flood fill approach
    uncovered_ranges = _create_bounding_boxes_from_cells(uncovered_cells)

    return False, uncovered_ranges, uncovered_cells


def _create_bounding_boxes_from_cells(cells: Set[Tuple[int, int]]) -> List[str]:
    """
    Create bounding boxes for a set of cells by finding contiguous regions.
    Similar to find_bounding_boxes but works directly with cell set.

    Args:
        cells: Set of (row, col) tuples

    Returns:
        List of range strings representing bounding boxes
    """
    if not cells:
        return []

    visited: Set[Tuple[int, int]] = set()
    bounding_boxes: List[str] = []

    for cell in cells:
        if cell in visited:
            continue

        # Flood fill to find contiguous region
        region_cells = _flood_fill_cells(cell, cells, visited)

        if region_cells:
            # Create bounding box
            rows = [r for r, c in region_cells]
            cols = [c for r, c in region_cells]
            min_row, max_row = min(rows), max(rows)
            min_col, max_col = min(cols), max(cols)

            range_str = get_range_string(min_row, min_col, max_row, max_col)
            bounding_boxes.append(range_str)

    return bounding_boxes


def _flood_fill_cells(
    start_cell: Tuple[int, int],
    all_cells: Set[Tuple[int, int]],
    visited: Set[Tuple[int, int]]
) -> Set[Tuple[int, int]]:
    """
    Perform flood fill to find contiguous cells (4-connected).

    Args:
        start_cell: Starting (row, col) position
        all_cells: Set of all valid cells
        visited: Set of already visited cells (will be modified)

    Returns:
        Set of cells in the contiguous region
    """
    if start_cell in visited or start_cell not in all_cells:
        return set()

    region_cells: Set[Tuple[int, int]] = set()
    queue: deque[Tuple[int, int]] = deque([start_cell])

    while queue:
        current_row, current_col = queue.popleft()

        if (current_row, current_col) in visited or (current_row, current_col) not in all_cells:
            continue

        visited.add((current_row, current_col))
        region_cells.add((current_row, current_col))

        # Check 4-connected neighbors
        neighbors = [
            (current_row - 1, current_col),
            (current_row + 1, current_col),
            (current_row, current_col - 1),
            (current_row, current_col + 1),
        ]

        for neighbor in neighbors:
            if neighbor in all_cells and neighbor not in visited:
                queue.append(neighbor)

    return region_cells
