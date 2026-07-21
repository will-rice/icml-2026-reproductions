"""
Region analysis LLM prompt templates.

Contains the Jinja2 template for structured region analysis and helper
functions for rendering it with context variables.
"""

from jinja2 import Template

# Region classification categories
REGION_CLASSES = {
    "data_table": "A structured table with headers containing organized rows and columns of data. Includes the entire table with its headers, footers, and any associated labels or titles that are visually part of the table structure. This is the most common region type.",

    "sheet_title_header": "Main title or primary heading of the sheet or a major section. Typically large, bold, or prominently formatted text at the top. Use this only for standalone titles/headers that are clearly separate from data tables.",

    "text_description": "Explanatory text, notes, instructions, or descriptive content that provides context or information. Not part of a data structure. Use for legend boxes, instruction blocks, or standalone notes.",

    "formula_calculation": "Area containing formulas, calculations, or computed results that aren't part of a larger table. May include labels and the calculated values. Use for summary boxes, standalone KPIs, or calculation areas separate from main tables.",

    "miscellaneous": "Any other content that doesn't fit the above categories, including one-off elements, isolated formatting, charts, images, or unique visual components. Use sparingly - most content fits one of the above types."
}


structured_region_analysis_prompt = Template("""\
You are an expert Excel analyst. Your task is to analyze the provided Excel sheet image and extract structured information about its regions, dependencies, and data patterns.

The screenshot includes row and column headings (letters and numbers) - use these to determine cell ranges accurately.

# 📊 Your Analysis Task

Analyze the sheet and provide a structured JSON output containing:
1. **Regions**: Distinct visual/logical areas of the sheet with optional closing operations
2. **Region Dependencies**: Workflow ordering relationships between regions
3. **Pasted Ranges**: Raw data ranges pasted from external sources
4. **Similarly Formatted Regions**: Groups of regions sharing similar formatting patterns

# 🎯 Core Principles

**MINIMIZE REGIONS**: Keep regions as large and few as possible. Only separate when areas serve distinctly different purposes or would be built at completely different times.

**COMPLETE UNITS**: Each region should be a complete, meaningful unit of work (e.g., entire table with headers and footers, complete section with title).

**SPATIAL + CONTEXTUAL**: Consider both visual layout and functional relationships. Content within the same visual boundary (borders, background colors) should typically be one region.

**WORKFLOW-ORIENTED**: Think about how a human would recreate this sheet from scratch - what would they build as separate tasks?

# 📦 Output Format

Return a valid JSON object matching this structure:

```json
{
  "regions": [
    {
      "id": 0,
      "range": "A1:D10",
      "type": "data_table",
      "closing_operations": [
        {
          "operation_type": "FILL_COLOR",
          "range": "B2:C10"
        },
        {
          "operation_type": "FONT_BOLD",
          "range": "D2:D10"
        },
        {
          "operation_type": "INPUT",
          "range": "A1"
        }
      ]
    },
    {
      "id": 1,
      "range": "F1:F5",
      "type": "text_description",
      "closing_operations": []
    }
  ],
  "region_dependencies": {
    1: 0
  },
  "pasted_ranges": [
    {
      "range": "A2:D10",
      "paste_nature": "full"
    },
    {
      "range": "F2:I10",
      "paste_nature": "row_wise"
    }
  ],
  "similarly_formatted_regions": [
    {
      "similar_regions": ["A1:D10", "F1:I10"],
      "format_paste_type": "paste_format"
    }
  ]
}
```

# 🔍 Detailed Guidelines

## Regions

**What to identify:**
- Look for distinct visual/logical areas based on:
  - Borders, background colors, or spacing that create visual separation
  - Functional purpose (table, title, notes, calculations)
  - Content type (data vs. text vs. formulas)
  - When they would likely be created during sheet building

**Region types (choose from):**
{% for class_name, description in region_classes.items() -%}
- **{{class_name}}**: {{description}}
{% endfor %}

**Requirements:**
- Each region must have a unique integer `id` starting from 0
- `range` must be a single cell (e.g., "A1") or continuous range (e.g., "A1:D10")
- Invalid ranges: "A1:B2, C3:D4", "A1 and B2", "A1, B1, C1"
- **CRITICAL**: Your regions must cover ALL non-empty cells in the sheet - no gaps allowed
- **NO EMPTY REGIONS**: Every region must contain actual content (values, formulas, text, or formatting) - do not include ranges with only blank/empty cells
- Include ALL meaningful content - empty regions list only if sheet is truly empty
- Regions must NOT overlap
- Keep regions large - combine related adjacent content when possible

**Closing Operations (optional field):**
- `closing_operations`: A list of operations that should be performed at the END of building a region
- These are formatting or data entry operations that a user would naturally do after the main structure is complete
- Examples:
  - Highlighting data cells (FILL_COLOR) after the table structure is built
  - Making headers bold (FONT_BOLD) as a final touch
  - Adding a final calculated value (INPUT) after formulas are set up
- Each closing operation has:
  - `operation_type`: One of 'INPUT' (data/formula entry), 'FONT_NAME', 'FONT_SIZE', 'FONT_BOLD', 'FONT_ITALIC', 'FONT_UNDERLINE', 'FONT_COLOR', 'FILL_COLOR'
  - `range`: Excel range where this operation applies (must be within the region's range)
- List should be ordered in the sequence these final operations would be performed
- Can be empty `[]` if no operations should be deferred to the end

## Region Dependencies

**What to capture:**
- Dependencies reflect natural workflow order: Region X should come AFTER Region Y if it would feel unnatural or illogical to build X before Y
- This OVERRIDES spatial/visual order when there's a logical reason
- Examples:
  - A summary table depends on the data table it summarizes
  - Calculations depend on the input values they reference
  - A chart title might depend on the chart itself being present

**Format:**
- Dictionary where key is the dependent region ID, value is the ID it depends on
- `{2: 0, 3: 1}` means region 2 depends on 0, and region 3 depends on 1
- Only capture DIRECT dependencies (not transitive)
- Can be empty `{}` if no workflow dependencies exist (spatial order is sufficient)

**When to include:**
- Include when there's a clear logical reason to build one region before another
- Omit when regions are independent or spatial order is already natural

## Pasted Ranges

**What to identify:**
- Ranges of raw data that were pasted as values from external sources (CSV, database, clipboard, another file)
- These are usually spans of data that look random, incomprehensible, or are dense numeric/text patterns
- **CRITICAL**: These ranges contain NO Excel formulas - formulas are always entered one-by-one, not pasted
- These represent "uninteresting" bulk data that was imported/pasted as a whole chunk
- **NOT titles or text descriptions**: Titles, headers, and descriptive text are typically typed manually, not pasted

**Paste Nature - How the data was pasted:**
- **`full`**: The entire range was pasted in one operation as a complete rectangular block
- **`column_wise`**: Data was pasted column by column (each column pasted separately)
- **`row_wise`**: Data was pasted row by row (each row pasted separately)
- **`single_entry`**: Data was pasted one cell at a time (rare, but possible for scattered paste operations)

**Requirements:**
- `range`: Must be a valid Excel range string
- `paste_nature`: Must be one of 'full', 'column_wise', 'row_wise', or 'single_entry'
- **CRITICAL**: Each pasted range must be completely contained within a region (subset of a region's range)
- Only include ranges that clearly look pasted (regular patterns, dense data, no formulas)
- Can be empty `[]` if no pasted ranges are detected

**When to include:**
- Data appears to be bulk-imported or copy-pasted from external sources
- Contains raw values (numbers, text, dates) but NO formulas
- Has a regular, dense pattern suggesting automated paste rather than manual entry
- **Tabular raw-looking data**: Dense tables with numeric/text data in a regular grid pattern are strong indicators of pasted data (may be entire table or specific columns/rows)
- NOT for data entered cell-by-cell manually or areas with formulas
- NOT for titles, headers, or descriptive text (these are typed manually)

## Similarly Formatted Regions

**What to identify:**
- Groups of regions that share very similar visual formatting patterns
- Indicates the user likely copy-pasted formatting or duplicated a template structure
- Look for regions with matching: borders, fonts, colors, cell styles, merged cells patterns

**Format Paste Type - How the formatting was replicated:**
- **`paste_format`**: Only formatting was copied (like Excel's "Paste Format" option) - structure and values are different
- **`paste_full`**: Entire table was pasted with formatting, then values were cleared/replaced - structure matches exactly

**Requirements:**
- `similar_regions`: List of Excel range strings (at least 2) that share similar formatting
- `format_paste_type`: Must be one of 'paste_format' or 'paste_full'
- Only include regions where formatting similarity is strong and intentional (not coincidental)
- Can be empty `[]` if no similarly formatted region groups exist

**When to include:**
- Multiple regions have very similar or identical formatting patterns
- Visual inspection suggests deliberate copying of formatting or structure
- NOT for regions that just happen to use the same font or color coincidentally

# ⚠️ Critical Constraints

1. **Valid JSON**: Output must be valid, parseable JSON
2. **Required fields**: All required fields must be present
3. **Complete coverage**: Every cell with content must be covered by at least one region - no gaps
4. **No overlaps**: Region ranges must not overlap with each other
5. **No empty regions**: Do not include regions that contain only blank/empty cells - every region must have actual content
6. **Valid references**: All region IDs in `region_dependencies` must exist in the regions list
7. **Valid ranges**: All range strings must be valid Excel ranges (continuous rectangles only)
8. **Empty but present**: All four main fields (regions, region_dependencies, pasted_ranges, similarly_formatted_regions) must be present even if empty
9. **Closing operations subset**: All `closing_operations` ranges must be completely within their parent region's range
10. **Pasted ranges subset**: All `pasted_ranges` must be completely contained within at least one region's range - they cannot extend outside regions
11. **Operation types**: All `operation_type` values must be one of: 'INPUT', 'FONT_NAME', 'FONT_SIZE', 'FONT_BOLD', 'FONT_ITALIC', 'FONT_UNDERLINE', 'FONT_COLOR', 'FILL_COLOR'
12. **Paste nature types**: All `paste_nature` values must be one of: 'full', 'column_wise', 'row_wise', 'single_entry'
13. **Format paste types**: All `format_paste_type` values must be one of: 'paste_format' or 'paste_full'
14. **Minimum similar regions**: Each entry in `similarly_formatted_regions` must have at least 2 ranges in `similar_regions`

# 📋 Additional Context

{% if regions_info %}
**Detected contiguous regions with values:**
{{regions_info}}

*Use these detected regions as a starting point, but feel free to merge adjacent regions if they serve the same purpose or would be built together. The goal is logical grouping, not just following detected boundaries. These regions represent areas with actual content - do not add regions for blank areas.*
{% endif %}

{% if formula_ranges %}
**Cells containing formulas:**
{{formula_ranges}}

*These cells contain Excel formulas. Areas with formulas are typically NOT pasted from external sources - they indicate manual construction. However, formulas may exist outside pasted table regions.*
{% endif %}

{% if merged_cells_info %}
**Merged cells in this sheet:**
{{merged_cells_info}}

*Note: Merged cells should be completely contained within a region (not split across regions).*
{% endif %}

**Image bounds:** {{sheet_ranges}}

*Ensure your regions collectively cover all cells within these bounds that contain data. Ignore empty/blank areas.*

# 🎬 Your Task

Analyze the provided Excel sheet image and return a JSON object following the structure and guidelines above. Think carefully about:
- How a human would mentally break down this sheet when recreating it
- What areas would naturally be built as separate tasks
- Which areas logically depend on others for workflow purposes
- Which operations would naturally be saved for the end of building each region
- Which data ranges look like bulk paste operations from external sources
- How the data was pasted (full chunk, column-wise, row-wise, or single entries)
- Which regions share similar formatting suggesting copy-paste of formatting or templates
- How the formatting was replicated (format only, full paste with clearing, or empty template)
- **Ensuring every cell with content is included in at least one region**

Return ONLY the JSON object, no additional text or explanation.
""")


def get_structured_prompt() -> Template:
    """Get the structured region analysis prompt template."""
    return structured_region_analysis_prompt


def render_structured_prompt(
    sheet_ranges: str,
    regions_info: str = "",
    merged_cells_info: str = "",
    formula_ranges: str = "",
    **kwargs
) -> str:
    """
    Render the structured region analysis prompt with provided context.

    Args:
        sheet_ranges: String describing the sheet bounds (e.g., "Sheet1: A1:Z100")
        regions_info: String describing detected contiguous regions
        merged_cells_info: String listing merged cell ranges
        formula_ranges: String describing ranges containing formulas
        **kwargs: Additional template variables

    Returns:
        Rendered prompt string
    """
    return structured_region_analysis_prompt.render(
        sheet_ranges=sheet_ranges,
        regions_info=regions_info,
        merged_cells_info=merged_cells_info,
        formula_ranges=formula_ranges,
        region_classes=REGION_CLASSES,
        **kwargs
    )
