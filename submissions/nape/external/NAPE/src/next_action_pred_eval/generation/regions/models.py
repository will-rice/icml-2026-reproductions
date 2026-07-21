"""
Region analysis Pydantic models.

Rich Pydantic models for structured region analysis output, including:
- ClosingOperation: Formatting/data operations deferred to end of region building
- RegionModel: A single identified region with optional closing operations
- PastedRange: A range of raw data pasted from external sources
- SimilarlyFormattedRegions: Groups of regions sharing similar formatting
- StructuredRegionOutput: Complete output structure for region analysis
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
from openpyxl.utils import range_boundaries

from next_action_pred_eval.utils.cell_utils import ranges_intersect


class ClosingOperation(BaseModel):
    """Represents an operation that should be performed at the end of building a region."""
    operation_type: str = Field(..., description="Type of operation to perform")
    range: str = Field(..., description="Excel range where this operation applies")

    @field_validator('operation_type')
    @classmethod
    def validate_operation_type(cls, v: str) -> str:
        """Validate that operation_type is one of the allowed types."""
        allowed_types = {
            'INPUT', 'FONT_NAME', 'FONT_SIZE', 'FONT_BOLD',
            'FONT_ITALIC', 'FONT_UNDERLINE', 'FONT_COLOR', 'FILL_COLOR'
        }
        if v not in allowed_types:
            raise ValueError(f"operation_type must be one of {allowed_types}, got: {v}")
        return v

    @field_validator('range')
    @classmethod
    def validate_range(cls, v: str) -> str:
        """Validate that range is a valid Excel range."""
        if not v or not isinstance(v, str):
            raise ValueError("Range must be a non-empty string")

        if ',' in v or ' and ' in v.lower():
            raise ValueError(f"Range must be a single cell or continuous range, got: {v}")

        try:
            range_boundaries(v)
        except Exception as e:
            raise ValueError(f"Invalid Excel range '{v}': {str(e)}")

        return v


class RegionModel(BaseModel):
    """Represents a single region in the Excel sheet (Pydantic version)."""
    id: int = Field(..., description="Unique identifier for the region")
    range: str = Field(..., description="Excel range string (e.g., 'A1:D10' or 'E11')")
    type: str = Field(..., description="Classification of the region")
    closing_operations: List[ClosingOperation] = Field(
        default_factory=list,
        description="Operations to perform at the end of building this region"
    )

    @field_validator('range')
    @classmethod
    def validate_range(cls, v: str) -> str:
        """Validate that range is a valid Excel range."""
        if not v or not isinstance(v, str):
            raise ValueError("Range must be a non-empty string")

        # Check for invalid patterns (multiple ranges)
        if ',' in v or ' and ' in v.lower():
            raise ValueError(f"Range must be a single cell or continuous range, got: {v}")

        try:
            # Use openpyxl to validate the range
            range_boundaries(v)
        except Exception as e:
            raise ValueError(f"Invalid Excel range '{v}': {str(e)}")

        return v

    @model_validator(mode='after')
    def validate_closing_operations(self):
        """Validate that all closing operation ranges are within the region range."""
        region_bounds = range_boundaries(self.range)

        for op in self.closing_operations:
            op_bounds = range_boundaries(op.range)
            if not is_subset_range_bounds(op_bounds, region_bounds):
                raise ValueError(
                    f"Closing operation range {op.range} is not within region range {self.range}"
                )

        return self

    def get_boundaries(self) -> Tuple[int, int, int, int]:
        """Get the boundaries of this region as (min_col, min_row, max_col, max_row)."""
        return range_boundaries(self.range)


class PastedRange(BaseModel):
    """Represents a range of raw data pasted from an external source."""
    range: str = Field(..., description="Excel range of the pasted data")
    paste_nature: str = Field(..., description="How the data was pasted: 'full', 'column_wise', 'row_wise', or 'single_entry'")

    @field_validator('paste_nature')
    @classmethod
    def validate_paste_nature(cls, v: str) -> str:
        """Validate that paste_nature is one of the allowed values."""
        allowed_values = {'full', 'column_wise', 'row_wise', 'single_entry'}
        if v not in allowed_values:
            raise ValueError(f"paste_nature must be one of {allowed_values}, got: {v}")
        return v

    @field_validator('range')
    @classmethod
    def validate_range(cls, v: str) -> str:
        """Validate that range is a valid Excel range."""
        if not v or not isinstance(v, str):
            raise ValueError("Range must be a non-empty string")

        if ',' in v or ' and ' in v.lower():
            raise ValueError(f"Range must be a single cell or continuous range, got: {v}")

        try:
            range_boundaries(v)
        except Exception as e:
            raise ValueError(f"Invalid Excel range '{v}': {str(e)}")

        return v

    def get_boundaries(self) -> Tuple[int, int, int, int]:
        """Get the boundaries of this range as (min_col, min_row, max_col, max_row)."""
        return range_boundaries(self.range)


class SimilarlyFormattedRegions(BaseModel):
    """Represents a group of regions with similar formatting."""
    similar_regions: List[str] = Field(..., description="List of Excel ranges that share similar formatting")
    format_paste_type: str = Field(..., description="How the formatting was replicated: 'paste_format', 'paste_full', or 'paste_template'")

    @field_validator('format_paste_type')
    @classmethod
    def validate_format_paste_type(cls, v: str) -> str:
        """Validate that format_paste_type is one of the allowed values."""
        allowed_values = {'paste_format', 'paste_full', 'paste_template'}
        if v not in allowed_values:
            raise ValueError(f"format_paste_type must be one of {allowed_values}, got: {v}")
        return v

    @field_validator('similar_regions')
    @classmethod
    def validate_similar_regions(cls, v: List[str]) -> List[str]:
        """Validate that all ranges are valid Excel ranges."""
        if not v or len(v) < 2:
            raise ValueError("similar_regions must contain at least 2 ranges")

        for range_str in v:
            if not range_str or not isinstance(range_str, str):
                raise ValueError("Each range must be a non-empty string")

            if ',' in range_str or ' and ' in range_str.lower():
                raise ValueError(f"Range must be a single cell or continuous range, got: {range_str}")

            try:
                range_boundaries(range_str)
            except Exception as e:
                raise ValueError(f"Invalid Excel range '{range_str}': {str(e)}")

        return v


class StructuredRegionOutput(BaseModel):
    """The complete output structure for region analysis."""
    regions: List[RegionModel] = Field(default_factory=list, description="List of identified regions")
    region_dependencies: Dict[int, int] = Field(
        default_factory=dict,
        description="Mapping of dependent region ID to the region it depends on"
    )
    pasted_ranges: List[PastedRange] = Field(
        default_factory=list,
        description="List of ranges containing pasted raw data from external sources"
    )
    similarly_formatted_regions: List[SimilarlyFormattedRegions] = Field(
        default_factory=list,
        description="Groups of regions with similar formatting patterns"
    )

    @model_validator(mode='after')
    def validate_output(self):
        """Perform cross-field validation."""
        region_ids = {r.id for r in self.regions}

        # Validate region IDs are unique
        if len(region_ids) != len(self.regions):
            raise ValueError("Region IDs must be unique")

        # Validate no overlapping regions
        for i, region1 in enumerate(self.regions):
            for region2 in self.regions[i+1:]:
                if ranges_intersect(region1.range, region2.range):
                    raise ValueError(
                        f"Regions overlap: {region1.id} ({region1.range}) and "
                        f"{region2.id} ({region2.range})"
                    )

        # Validate region dependencies
        for dependent_id, dependency_id in self.region_dependencies.items():
            if dependent_id not in region_ids:
                raise ValueError(
                    f"Dependent region ID {dependent_id} not found in regions list"
                )
            if dependency_id not in region_ids:
                raise ValueError(
                    f"Dependency region ID {dependency_id} not found in regions list"
                )
            if dependent_id == dependency_id:
                raise ValueError(
                    f"Region {dependent_id} cannot depend on itself"
                )

        # Check for circular dependencies
        self._check_circular_dependencies()

        # Validate pasted ranges are subsets of regions
        for pasted_range in self.pasted_ranges:
            is_subset = False
            for region in self.regions:
                if is_subset_range(pasted_range.range, region.range):
                    is_subset = True
                    break

            if not is_subset:
                raise ValueError(
                    f"Pasted range {pasted_range.range} is not contained within any region"
                )

        return self

    def _check_circular_dependencies(self):
        """Check for circular dependencies in region_dependencies."""
        visited = set()
        rec_stack = set()

        def has_cycle(node: int) -> bool:
            visited.add(node)
            rec_stack.add(node)

            # Check if this node has a dependency
            if node in self.region_dependencies:
                neighbor = self.region_dependencies[node]
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for region_id in [r.id for r in self.regions]:
            if region_id not in visited:
                if has_cycle(region_id):
                    raise ValueError("Circular dependency detected in region_dependencies")


def is_subset_range(inner_range: str, outer_range: str) -> bool:
    """Check if inner_range is completely contained within outer_range."""
    inner_bounds = range_boundaries(inner_range)
    outer_bounds = range_boundaries(outer_range)
    return is_subset_range_bounds(inner_bounds, outer_bounds)


def is_subset_range_bounds(inner_bounds: Tuple[int, int, int, int], outer_bounds: Tuple[int, int, int, int]) -> bool:
    """Check if inner_bounds is completely contained within outer_bounds."""
    inner_min_col, inner_min_row, inner_max_col, inner_max_row = inner_bounds
    outer_min_col, outer_min_row, outer_max_col, outer_max_row = outer_bounds

    return (
        inner_min_col >= outer_min_col and
        inner_max_col <= outer_max_col and
        inner_min_row >= outer_min_row and
        inner_max_row <= outer_max_row
    )
