"""
Date Noise Filter - Removes unnecessary datetime noise from operations

Handles:
- Removing 00:00:00 times from date values (e.g., "2023-01-15 00:00:00" -> "2023-01-15")
- Filtering out epoch dates (1900-01-00, 1900-01-01, 1920-01-01) which are Excel artifacts
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
import re

from next_action_pred_eval.generation.sequencing.base import BaseTransformer, SequencingContext
from next_action_pred_eval.core.operation import Operation


# Common Excel epoch dates that are usually artifacts
EPOCH_DATES = {
    "1900-01-00",
    "1900-01-01",
    "1899-12-30",  # Excel's actual day 0
    "1899-12-31",
    "1920-01-01",
}

# Pattern to match datetime with midnight time (00:00:00)
MIDNIGHT_PATTERN = re.compile(r'^(\d{4}-\d{2}-\d{2})\s+00:00:00$')

# Pattern to match ISO datetime format
ISO_DATETIME_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}(T|\s)\d{2}:\d{2}:\d{2}')


def is_epoch_date(value: Any) -> bool:
    """Check if a value is a known Excel epoch date."""
    if value is None:
        return False

    value_str = str(value).strip()

    # Direct match
    if value_str in EPOCH_DATES:
        return True

    # Match with time component
    for epoch in EPOCH_DATES:
        if value_str.startswith(epoch):
            return True

    return False


def clean_midnight_time(value: Any) -> Any:
    """
    Remove 00:00:00 time from a date value if present.

    Args:
        value: The value to clean

    Returns:
        Cleaned value with midnight time removed, or original value if not applicable
    """
    if value is None:
        return value

    value_str = str(value).strip()

    # Check for midnight pattern
    match = MIDNIGHT_PATTERN.match(value_str)
    if match:
        return match.group(1)

    # Also handle "T00:00:00" format
    if value_str.endswith("T00:00:00"):
        return value_str[:-9]  # Remove "T00:00:00"

    return value


def clean_datetime_value(value: Any, remove_midnight: bool = True, filter_epoch: bool = True) -> Optional[Any]:
    """
    Clean a datetime value by removing noise.

    Args:
        value: The value to clean
        remove_midnight: If True, remove 00:00:00 times
        filter_epoch: If True, return None for epoch dates

    Returns:
        Cleaned value, or None if it should be filtered out
    """
    if value is None:
        return None

    # Check for epoch dates
    if filter_epoch and is_epoch_date(value):
        return None

    # Remove midnight time
    if remove_midnight:
        return clean_midnight_time(value)

    return value


def process_value_recursive(
    value: Any,
    remove_midnight: bool = True,
    filter_epoch: bool = True
) -> Any:
    """
    Recursively process a value (including 2D arrays) to clean datetime values.

    Args:
        value: The value to process (can be scalar, list, or 2D list)
        remove_midnight: If True, remove 00:00:00 times
        filter_epoch: If True, filter out epoch dates

    Returns:
        Processed value with datetime noise removed
    """
    if isinstance(value, list):
        return [process_value_recursive(item, remove_midnight, filter_epoch) for item in value]

    # Check if value looks like a datetime
    value_str = str(value).strip() if value is not None else ""

    # Only process if it looks like a date/datetime
    if ISO_DATETIME_PATTERN.match(value_str) or value_str in EPOCH_DATES:
        cleaned = clean_datetime_value(value, remove_midnight, filter_epoch)
        # If filtered out (epoch), replace with empty string instead of None
        # to preserve array dimensions
        return "" if cleaned is None else cleaned

    return value


class DateNoiseFilter(BaseTransformer):
    """
    Filters noise from date/datetime values in operations.

    This transformer cleans up common datetime noise:
    1. Removes midnight (00:00:00) times from dates
    2. Filters out Excel epoch dates (1900-01-00, 1900-01-01, etc.)

    Config:
        enabled: bool - Whether the filter is active
        remove_midnight: bool - Remove 00:00:00 times from dates
        filter_epoch_dates: bool - Filter out Excel epoch dates
    """

    DEFAULT_CONFIG = {
        "enabled": True,
        "remove_midnight": True,
        "filter_epoch_dates": True,
    }

    def transform(self, context: SequencingContext) -> SequencingContext:
        if self.can_skip(context):
            return context

        from next_action_pred_eval.core.operations import SetInput, SetValue

        remove_midnight = self.config.get("remove_midnight", True)
        filter_epoch = self.config.get("filter_epoch_dates", True)

        cleaned_ops = []
        ops_modified = 0
        ops_removed = 0

        for op in context.operations:
            # Only process SetInput and SetValue operations
            if not isinstance(op, (SetInput, SetValue)):
                cleaned_ops.append(op)
                continue

            # Skip inverse operations
            if op.is_inverse:
                cleaned_ops.append(op)
                continue

            # Process the value
            original_value = op.value
            cleaned_value = process_value_recursive(
                original_value,
                remove_midnight=remove_midnight,
                filter_epoch=filter_epoch
            )

            # Check if value was modified
            if cleaned_value != original_value:
                # Check if entire operation should be filtered (single epoch date)
                if cleaned_value == "" and not isinstance(original_value, list):
                    ops_removed += 1
                    continue  # Skip this operation

                # Create modified operation
                modified_op = op.model_copy(update={'value': cleaned_value})
                cleaned_ops.append(modified_op)
                ops_modified += 1
            else:
                cleaned_ops.append(op)

        self.log(
            context,
            f"Date filter: {ops_modified} ops modified, {ops_removed} ops removed"
        )

        return context.copy_with_operations(cleaned_ops)
