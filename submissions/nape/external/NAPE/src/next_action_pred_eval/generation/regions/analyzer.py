"""
Region analysis with LLM-based structured output and retry logic.

This module provides the core analysis functions that use an LLMAdapter
to analyze spreadsheet images and extract structured region information.
All LLM calls go through the LLMAdapter interface.

Key functions:
- retry_structured_analysis: Low-level retry loop with validation feedback
- analyze_sheet_regions: High-level convenience wrapper with preprocessing
"""

import time
import logging
from typing import Optional, Tuple, List

from next_action_pred_eval.utils.llm.base import LLMAdapter
from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.generation.regions.models import StructuredRegionOutput
from next_action_pred_eval.generation.regions.prompts import render_structured_prompt
from next_action_pred_eval.generation.regions.parsing_utils import (
    parse_structured_output,
    format_validation_errors,
    validate_output_structure,
    validate_coverage,
    get_region_summary,
    get_formula_ranges,
)

logger = logging.getLogger(__name__)


def retry_structured_analysis(
    llm: LLMAdapter,
    sheet_image_path: str,
    sheet_ranges: str,
    regions_info: str = "",
    merged_cells_info: str = "",
    formula_ranges: str = "",
    sheet_operations: Optional[List[Operation]] = None,
    max_sheet_limit: int = 100,
    max_retries: int = 3,
    **template_vars
) -> Tuple[Optional[StructuredRegionOutput], Optional[str], float, str, int]:
    """
    Perform structured region analysis with retry logic for validation failures.

    All LLM calls are routed through the provided ``LLMAdapter`` instance.
    The adapter's ``complete`` method is called with a list of message dicts
    (each having ``role`` and ``content`` keys).  When an image is provided,
    the first user message includes an ``image`` key with the file path so
    that adapters supporting vision can forward it to the model.

    Args:
        llm: An LLMAdapter instance used for all LLM calls.
        sheet_image_path: Path to the sheet image.
        sheet_ranges: String describing sheet bounds.
        regions_info: String describing detected contiguous regions.
        merged_cells_info: String listing merged cell ranges.
        formula_ranges: String describing ranges containing formulas.
        sheet_operations: List of operations for coverage validation.
        max_sheet_limit: Maximum sheet dimension.
        max_retries: Maximum number of retries.
        **template_vars: Additional template variables.

    Returns:
        Tuple of:
        - parsed_output: StructuredRegionOutput if successful, None otherwise
        - final_response_text: The final raw LLM response text
        - total_time: Total time taken across all attempts
        - final_prompt: The original prompt content
        - attempt_count: Number of attempts made
    """
    total_time = 0.0
    previous_errors: List[List[str]] = []
    last_response_text: Optional[str] = None
    final_response_text: Optional[str] = None

    # Generate the initial prompt (store for return)
    original_content = render_structured_prompt(
        sheet_ranges=sheet_ranges,
        regions_info=regions_info,
        merged_cells_info=merged_cells_info,
        formula_ranges=formula_ranges,
        **template_vars
    )

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1} for structured region analysis")

            # Build messages for this attempt
            if attempt == 0:
                # Initial attempt - just the original prompt with image
                messages = [
                    {
                        "role": "user",
                        "content": original_content,
                        "image": sheet_image_path,
                    }
                ]
            else:
                # Retry attempt - original prompt + last response + feedback
                messages = [
                    {
                        "role": "user",
                        "content": original_content,
                        "image": sheet_image_path,
                    }
                ]

                # Add ONLY the last incorrect response
                if last_response_text:
                    messages.append({
                        "role": "assistant",
                        "content": last_response_text,
                    })

                # Generate feedback ONLY for the last attempt
                retry_feedback = format_validation_errors(
                    previous_errors[-1],
                    None  # Don't include earlier errors to reduce tokens
                )

                messages.append({
                    "role": "user",
                    "content": retry_feedback,
                })

            # Call the LLM through the adapter
            start_time = time.time()

            # Adjust temperature for retries
            temperature = 0.0 if attempt == 0 else 0.1 + attempt * 0.05

            response_text = llm.complete(
                messages=messages,
                temperature=temperature,
            )

            elapsed = time.time() - start_time
            total_time += elapsed

            final_response_text = response_text
            last_response_text = response_text

            logger.info(f"Received response in {elapsed:.2f}s")

            # Parse and validate the response
            parsed_output, validation_errors = parse_structured_output(response_text)

            if parsed_output is not None:
                # Additional semantic validation
                warnings = validate_output_structure(parsed_output)
                if warnings:
                    logger.warning("Validation warnings:")
                    for warning in warnings:
                        logger.warning(f"  - {warning}")

                # Coverage validation (if sheet_operations provided)
                if sheet_operations:
                    is_covered, uncovered_ranges, uncovered_cells = validate_coverage(
                        parsed_output, sheet_operations, max_sheet_limit
                    )

                    if not is_covered:
                        logger.warning(
                            f"Coverage validation failed: {len(uncovered_ranges)} uncovered region(s)"
                        )
                        logger.warning(f"Uncovered ranges: {', '.join(uncovered_ranges)}")

                        # Create coverage error message
                        coverage_errors = [
                            f"Coverage validation failed: {len(uncovered_cells)} cells are not covered by any region",
                            f"Found {len(uncovered_ranges)} uncovered region(s):",
                        ]
                        for i, rng in enumerate(uncovered_ranges, 1):
                            coverage_errors.append(f"  {i}. Range {rng}")

                        coverage_errors.append(
                            "\nPlease update your regions to cover ALL cells in the sheet."
                        )
                        coverage_errors.append(
                            "You can either expand existing regions or add new regions to cover these areas."
                        )

                        previous_errors.append(coverage_errors)

                        # If we have retries left, continue with coverage feedback
                        if attempt < max_retries:
                            logger.info("Retrying with coverage feedback...")
                            continue
                        else:
                            # Max retries reached
                            logger.error("Max retries exceeded with coverage issues")
                            return None, final_response_text, total_time, original_content, attempt + 1

                # Success!
                logger.info("Successfully parsed and validated structured output")
                logger.info(get_region_summary(parsed_output))
                return parsed_output, final_response_text, total_time, original_content, attempt + 1

            # Validation failed (parsing or model validation)
            logger.warning(f"Validation failed on attempt {attempt + 1}:")
            for error in validation_errors:
                logger.warning(f"  - {error}")

            previous_errors.append(validation_errors)

            # If we have retries left, continue to next iteration
            if attempt < max_retries:
                logger.info("Retrying with validation feedback...")

        except Exception as e:
            logger.error(f"Error during attempt {attempt + 1}: {str(e)}")
            if attempt >= max_retries:
                logger.error("Max retries exceeded")
                break
            logger.info("Retrying...")

    # All attempts failed
    logger.error(f"Failed to get valid output after {max_retries + 1} attempts")
    return None, final_response_text, total_time, original_content, max_retries + 1


def analyze_sheet_regions(
    llm: LLMAdapter,
    sheet_image_path: str,
    sheet_name: str,
    sheet_range: str,
    regions_dict: dict,
    merged_cells_list: List[str],
    sheet_operations: List[Operation],
    max_sheet_limit: int = 100,
    max_retries: int = 3,
    regions_info_formatter=None,
) -> Tuple[Optional[StructuredRegionOutput], Optional[str], float, str, int]:
    """
    Analyze regions in a sheet with all necessary preprocessing.

    This is a convenience wrapper around ``retry_structured_analysis`` that
    handles formatting of regions_info, merged_cells, and formula ranges
    before invoking the LLM.

    Args:
        llm: An LLMAdapter instance used for all LLM calls.
        sheet_image_path: Path to the sheet image.
        sheet_name: Name of the sheet.
        sheet_range: Range string for the sheet (e.g., "A1:Z100").
        regions_dict: Dictionary of detected bounding boxes.
        merged_cells_list: List of merged cell ranges.
        sheet_operations: List of operations for coverage validation.
        max_sheet_limit: Maximum sheet dimension.
        max_retries: Maximum number of retries.
        regions_info_formatter: Optional callable to format regions_dict into
            a string.  If ``None``, a simple default formatter is used.

    Returns:
        Tuple of (parsed_output, response_text, time_taken, prompt, attempts)
    """
    # Format regions info
    if regions_info_formatter is not None:
        regions_info = regions_info_formatter(regions_dict)
    else:
        regions_info = _default_regions_info_formatter(regions_dict)

    # Format merged cells info
    merged_cells_info = ", ".join(merged_cells_list) if merged_cells_list else "No merged cells"

    # Extract and format formula ranges
    formula_range_list = get_formula_ranges(sheet_operations, max_sheet_limit)
    formula_ranges = ", ".join(formula_range_list) if formula_range_list else "No formula cells detected"

    # Format sheet ranges
    sheet_ranges = f"{sheet_name}: {sheet_range}"

    return retry_structured_analysis(
        llm=llm,
        sheet_image_path=sheet_image_path,
        sheet_ranges=sheet_ranges,
        regions_info=regions_info,
        merged_cells_info=merged_cells_info,
        formula_ranges=formula_ranges,
        sheet_operations=sheet_operations,
        max_sheet_limit=max_sheet_limit,
        max_retries=max_retries,
    )


def _default_regions_info_formatter(regions_dict: dict) -> str:
    """
    Simple default formatter for regions_dict.

    Converts a dict of ``{region_label: range_string}`` (or similar)
    into a human-readable string for inclusion in the prompt.
    """
    if not regions_dict:
        return ""
    lines = []
    for key, value in regions_dict.items():
        lines.append(f"  - {key}: {value}")
    return "\n".join(lines)
