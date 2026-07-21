"""
Regions module - Region analysis models and LLM-based analyzer.

This module provides models and utilities for analyzing spreadsheet regions.
Regions are contiguous areas of a spreadsheet with similar formatting or purpose.

Key components:
- Region: Basic dataclass for representing regions (legacy, coordinate-based)
- RegionModel: Rich Pydantic model for LLM-produced region analysis output
- ClosingOperation: Formatting/data operations deferred to end of region building
- PastedRange: Range of raw data pasted from external sources
- SimilarlyFormattedRegions: Groups of regions sharing similar formatting
- StructuredRegionOutput: Complete output structure for region analysis
- retry_structured_analysis: LLM-based region analysis with retry logic
- analyze_sheet_regions: High-level convenience wrapper
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Region:
    """
    Represents a contiguous region in a spreadsheet.

    Attributes:
        id: Unique identifier for the region
        min_row: Starting row (1-indexed)
        max_row: Ending row (1-indexed)
        min_col: Starting column (1-indexed)
        max_col: Ending column (1-indexed)
        region_type: Type of region (header, data, totals, etc.)
        metadata: Additional metadata
    """
    id: str
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    region_type: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def contains_cell(self, row: int, col: int) -> bool:
        """Check if a cell is within this region."""
        return (
            self.min_row <= row <= self.max_row and
            self.min_col <= col <= self.max_col
        )

    def overlaps(self, other: 'Region') -> bool:
        """Check if this region overlaps with another."""
        return not (
            self.max_col < other.min_col or
            other.max_col < self.min_col or
            self.max_row < other.min_row or
            other.max_row < self.min_row
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "min_row": self.min_row,
            "max_row": self.max_row,
            "min_col": self.min_col,
            "max_col": self.max_col,
            "region_type": self.region_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Region':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            min_row=data["min_row"],
            max_row=data["max_row"],
            min_col=data["min_col"],
            max_col=data["max_col"],
            region_type=data.get("region_type", "unknown"),
            metadata=data.get("metadata", {}),
        )


# Pydantic models for structured LLM region analysis
from next_action_pred_eval.generation.regions.models import (  # noqa: E402
    ClosingOperation,
    RegionModel,
    PastedRange,
    SimilarlyFormattedRegions,
    StructuredRegionOutput,
    is_subset_range,
    is_subset_range_bounds,
)

# Prompt utilities
from next_action_pred_eval.generation.regions.prompts import (  # noqa: E402
    REGION_CLASSES,
    render_structured_prompt,
    get_structured_prompt,
)

# Parsing and validation utilities
from next_action_pred_eval.generation.regions.parsing_utils import (  # noqa: E402
    extract_json_from_response,
    parse_structured_output,
    format_validation_errors,
    create_retry_prompt,
    validate_output_structure,
    validate_coverage,
    get_region_summary,
    get_formula_ranges,
)

# Analyzer (LLM-based region analysis via LLMAdapter)
from next_action_pred_eval.generation.regions.analyzer import (  # noqa: E402
    retry_structured_analysis,
    analyze_sheet_regions,
)


__all__ = [
    # Legacy dataclass
    "Region",
    # Pydantic models
    "ClosingOperation",
    "RegionModel",
    "PastedRange",
    "SimilarlyFormattedRegions",
    "StructuredRegionOutput",
    "is_subset_range",
    "is_subset_range_bounds",
    # Prompt utilities
    "REGION_CLASSES",
    "render_structured_prompt",
    "get_structured_prompt",
    # Parsing utilities
    "extract_json_from_response",
    "parse_structured_output",
    "format_validation_errors",
    "create_retry_prompt",
    "validate_output_structure",
    "validate_coverage",
    "get_region_summary",
    "get_formula_ranges",
    # Analyzer
    "retry_structured_analysis",
    "analyze_sheet_regions",
]
