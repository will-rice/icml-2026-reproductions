from __future__ import annotations

from textwrap import dedent
from typing import Dict, Iterable, List, Optional

from jinja2 import Template


def _render(template: str, **context: object) -> str:
    body = dedent(template).strip("\n")
    return Template(body).render(**context).strip()


def format_operation_catalog(operation_docs: Dict[str, Dict[str, str]]) -> str:
    rows = ["### Allowed operations"]
    for name in sorted(operation_docs):
        doc = operation_docs[name]
        rows.append(f"- **{name}**: {doc['description']} Example: `{doc['usage']}`")
    return "\n".join(rows)


def summarize_reference_operations(ops: List[str], limit: Optional[int]) -> str:
    if limit is None or limit <= 0:
        sliced = ops
        truncated = False
    else:
        sliced = ops[:limit]
        truncated = len(ops) > limit
    lines = "\n".join(sliced)
    suffix = "\n... (truncated)" if truncated else ""
    return f"```text\n{lines}{suffix}\n```" if lines else "(no operations found)"


SYSTEM_PROMPT_TEMPLATE = """
You are an Excel build-out specialist. Given symbolic Excel operations, make some changes to them wherever needed so they still recreate the exact same workbook but read like a human sequentially crafting the sheet: group related ranges, keep row/column flow, and only make minimal edits that improve the story.

You may intentionally remove operations that are unnecessary for a human workflow (e.g., formatting applied to empty/unused cells, redundant number formats on text cells). When you do, declare these omissions using the IGNORE section so the system knows the drift is intentional.

You may also correct obvious human errors in the original content (spelling mistakes, typos, color coordination conflicts). When you do, declare these corrections using the CORRECTIONS section so the validation system accepts the intentional deviation from the original state.
"""


def build_system_prompt() -> str:
    return _render(SYSTEM_PROMPT_TEMPLATE)


ORIGINAL_INSTRUCTION_TEMPLATE = """
The file you received was produced by a heuristic sequencer. Improve it with subtle, human-like adjustments—swap nearby steps, break or merge related ranges, surface recurring patterns—but avoid discarding the entire plan unless absolutely necessary.

Sheet under consideration: **{{ sheet_name }}** ({{ workspace }}). Only touch this sheet.

{{ operation_catalog }}
{% if compressed_input_docs %}
{{ compressed_input_docs }}
{% endif %}
### Current sequence from heuristics
{{ reference_ops_snippet }}

### Output contract
- Add a `RATIONALE:` section with one or two bullet sentences describing the narrative.
- Provide the revised operations inside a fenced code block starting with ` ```ops ` and ending with ` ``` `. Each row must follow `OPERATION | Sheet!Range | value`.
- Favor minimal, surgical edits to the given order so the sequence still feels familiar.

### Intentional omissions (IGNORE section)
If you deliberately remove operations (e.g., formatting on empty cells, number formats on text), list them in an `IGNORE:` section **after** the ops block. Group ranges by operation type:

```
IGNORE:
FONT_SIZE: [B1:E1, M16:M17, H53:H54]
NUMBER_FORMAT: [C9, E14, E21, C28]
FONT_BOLD: [M16:M17, L64:L66]
```

This tells the system these diffs are intentional and can be accepted.

### Content corrections (CORRECTIONS section)
If you fix obvious human errors in the original content (spelling mistakes, typos, color conflicts, inconsistent formatting), list them in a `CORRECTIONS:` section **after** the ops block. Each correction should specify the cell, original value, corrected value, and reason:

```
CORRECTIONS:
- Sheet1!A5: "Totla" -> "Total" (spelling fix)
- Sheet1!B3: "Janury" -> "January" (typo)
- Sheet1!C1: fill #FF0000 -> #CC0000 (color coordination with header theme)
```

**Important:** The validation system compares your output against the original final state. Without declaring corrections, any content change will be flagged as a mismatch and fail validation. The CORRECTIONS section tells the system these deviations are intentional improvements, not errors.
"""


def build_original_instruction(
    *,
    sheet_name: str,
    max_dimension: int | None,
    operation_catalog: str,
    reference_ops_snippet: str,
    compressed_input_docs: str = "",
) -> str:
    workspace = "unbounded" if max_dimension is None else f"within {max_dimension} rows/columns"
    return _render(
        ORIGINAL_INSTRUCTION_TEMPLATE,
        sheet_name=sheet_name,
        workspace=workspace,
        operation_catalog=operation_catalog,
        reference_ops_snippet=reference_ops_snippet,
        compressed_input_docs=compressed_input_docs,
    )


FEEDBACK_TEMPLATE = """
Feedback for attempt {{ iteration }}/{{ max_iterations }}:
- Validation: {{ validation_section }}
- State comparison: {{ comparator_summary }}
- Delta vs. reference: {{ diff_summary }}

Retry focus: {{ retry_hint }}

Repair plan ({{ ops_label }}):
{{ repair_section }}

Over-predicted cells:
{{ extra_summary or 'None' }}

Use these completion hints only if they keep the narrative human and natural.
"""


def build_feedback_message(
    *,
    iteration: int,
    max_iterations: int,
    validation_errors: Iterable[str],
    comparator_summary: str,
    diff_summary: str,
    retry_hint: str,
    completion_moves: Iterable[str],
    ops_to_reach_target: Optional[int],
    extra_summary: str,
) -> str:
    validation_section = "\n".join(validation_errors) if validation_errors else "None"
    moves_list = list(completion_moves)
    if moves_list:
        if ops_to_reach_target and ops_to_reach_target > len(moves_list):
            ops_label = f"showing {len(moves_list)} of \u2248{ops_to_reach_target} ops"
        else:
            ops_label = f"showing {len(moves_list)} ops"
        repair_section = "\n".join(f"- {op}" for op in moves_list)
    else:
        ops_label = "no concrete ops yet"
        repair_section = "Provide a parseable candidate so we can compute concrete repairs."
    return _render(
        FEEDBACK_TEMPLATE,
        iteration=iteration,
        max_iterations=max_iterations,
        validation_section=validation_section,
        comparator_summary=comparator_summary,
        diff_summary=diff_summary,
        retry_hint=retry_hint,
        ops_label=ops_label,
        repair_section=repair_section,
        extra_summary=extra_summary,
    )


RETRY_FEEDBACK_TEMPLATE = """
[Attempt {{ retry }}/{{ max_retries }}{% if iteration %}, Iteration {{ iteration }}{% endif %}]
Validation issues detected. Please correct and resend operations.

Details:
{{ errors }}
"""


def build_retry_feedback_message(
    *, retry: int, max_retries: int, validation_errors: Iterable[str], iteration: Optional[int] = None
) -> str:
    errors = "\n".join(validation_errors) if validation_errors else "No validation detail available."
    return _render(
        RETRY_FEEDBACK_TEMPLATE,
        retry=retry,
        max_retries=max_retries,
        errors=errors,
        iteration=iteration,
    )


RETRY_MISMATCH_TEMPLATE = """
[Attempt {{ retry }}/{{ max_retries }}{% if iteration %}, Iteration {{ iteration }}{% endif %}]
Workbook mismatch detected. Bring the predicted cells to parity, but preserve the human tone.

Possible completion moves (only if they still feel natural):
{{ repairs_text }}
{% if mismatch_ops_text %}

Possible mismatched operations (consider removing or modifying):
{{ mismatch_ops_text }}
{% endif %}
"""


def build_retry_mismatch_message(
    *,
    retry: int,
    max_retries: int,
    mismatch_report: str,
    repair_preview: Iterable[str],
    mismatch_operations: Optional[Iterable[str]] = None,
    iteration: Optional[int] = None,
) -> str:
    repairs = list(repair_preview)
    repairs_text = "\n".join(f"- {item}" for item in repairs) if repairs else "(none detected)"
    mismatch_ops = list(mismatch_operations) if mismatch_operations else []
    mismatch_ops_text = "\n".join(f"- {item}" for item in mismatch_ops) if mismatch_ops else ""
    return _render(
        RETRY_MISMATCH_TEMPLATE,
        retry=retry,
        max_retries=max_retries,
        mismatch_report=mismatch_report,
        repairs_text=repairs_text,
        mismatch_ops_text=mismatch_ops_text,
        iteration=iteration,
    )


JUDGE_SYSTEM_PROMPT_TEMPLATE = """
You are an Excel craft reviewer. Your job is to decide whether a symbolic operation sequence *feels* like a human building the sheet—not a script generated by rigid heuristics.

The sequence you receive already produces the correct final workbook. You are judging **only** whether the ordering and granularity read naturally—like someone sitting at a keyboard, thinking about the task, and making progress in a believable way.

Use the diff summary and screenshot for spatial context: understand what content exists and how the sheet is laid out. Then ask yourself: "Would a real person do it this way?"

---

### Things to consider (not hard rules—use judgment based on context)

**Dependency & semantic flow**
- Does content that other content depends on come first? (e.g., precedent values before formulas, source tables before summaries, legends before sections that reference them)
- Are related operations grouped, or does the sequence jump around the sheet without apparent reason?
- Keep in mind: users most likely add formulas *after* entering the values those formulas reference—check that formula inputs exist before the formula is entered.

**INPUT granularity**
- Formulas are almost always entered one cell at a time, and not as bulk/range INPUTs.
- Bulk/range INPUTs make sense for dense, homogeneous raw data that might realistically be pasted from an external source (CSV import, copied dataset).
- Heterogeneous values, short lists, labels, or anything a human would naturally type cell-by-cell should usually be single-cell INPUTs.
- Context matters: a 50-row column of random numbers is plausible as a paste; a 5-cell column of category names is not.
- As a rough guideline: larger data tables (more than ~64 cells) are more likely to be pasted in bulk, while smaller tables (fewer than ~64 cells) are more likely entered cell-by-cell or in small chunks. This isn't a hard rule—context and content homogeneity matter more—but size is a useful signal.
- Sequential or range-based data—like integer ranges (1-5, 6-10), date ranges (Jan 1 - Jan 10), consecutive dates, numbered steps, or any data where knowing one value lets you infer the next—often feel more like something a human types out one cell at a time rather than pastes in bulk, since they're constructing a logical progression as they go.
- Header rows, header columns, and footer rows/columns are often typed individually rather than pasted—these tend to be short, meaningful labels that a person crafts as they structure the sheet.
- Keep in mind that pasted data tables sometimes have "islands"—gaps or missing cells within an otherwise contiguous block. When evaluating whether a range INPUT is plausible, consider that a human pasting rows/columns together would naturally include those empty cells as part of the paste rather than skipping them. Don't assume every gap means separate operations.

**People work in surprising ways**
There's no "correct" workflow—people approach the same task very differently. For example, to create multiple similar tables, one person might build each from scratch; another might copy an existing table as a template, paste it multiple times, then overwrite the values; another might just copy the formatting and type fresh content. Similarly with data: some paste everything at once, some paste sections at a time, some paste and immediately edit. All of these are valid, and the resulting operation sequences would look quite different. The point isn't to memorize specific patterns, but to stay open to the many roundabout-yet-reasonable paths a human might take.

**Formatting timing & large-range operations**
- Structural formatting (header bolding, alignment, basic borders) can happen early or mid-build—it helps the person see what they're doing.
- Decorative or emphasis formatting (highlighting specific data points, callout fills) typically comes after the relevant data exists.
- Large formatting operations (like applying borders to A2:J90) can look unnatural if they appear before there's any indication of how big the final content area will be. A user typically wouldn't format 90 rows before knowing they need 90 rows. Look for signals that establish the range extent first—filling in column headers suggests the column count, entering a few rows of data suggests the pattern. Once there's enough context for the user to reasonably know the dimensions, bulk formatting makes sense.
- There's no single correct order; many variations are natural. Flag only sequences where the timing is *clearly* awkward (e.g., highlighting cells that don't have values yet, or formatting huge ranges before any structure exists).

**Merge & layout**
- Merges usually happen close to when the spanning text is entered—either just before or just after.
- When cell merging and text wrapping appear on the same range, humans typically merge first, then set wrap text. If you see wrap text operations appearing before merge on the same cells, the order might feel more natural reversed.
- Constant context-switching (jumping between distant areas without finishing a logical unit) can feel unnatural, but sometimes dependency chains require it.

**Unnecessary operations & formatting outside content area**
- Formatting applied to empty/unused cells (outside the content area) is often heuristic noise—a human wouldn't style columns they never intend to use.
- Number formats on text cells (e.g., date format on "No School") serve no purpose.
- Sheets sometimes have formatting applied to entire rows or columns, or to large swaths of cells beyond where actual content exists. These are often artifacts from templates or bulk operations. If you notice formatting extending well past the used range—especially patterns like "the empty area is colored while the data area is white"—consider normalizing or removing such operations.
- If trimming a formatting range to only cells with content results in a single operation (not multiple fragments), that's usually preferable. If an entire operation targets only empty cells, it can often be removed entirely.
- If such operations exist, recommend removing them (the refiner can use `#SKIP` to declare intentional omissions).

**Operation consolidation & repetitive patterns**
- If you notice the same operation appearing many times across scattered cells or ranges, consider whether a human might have applied it once to a larger region (an entire row, column, or the whole sheet) rather than cell-by-cell.
- When you see multiple operations of the same type with the same value on adjacent cells or ranges, consider whether merging these into one larger range operation would better reflect a single selection action.
- It's also possible someone applied formatting broadly and then overwrote or cleared specific cells afterward—resulting in a sequence that *looks* fragmented but actually stems from a single bulk action plus targeted edits. Sometimes a cleaner sequence involves applying a broad operation to a large range and then overwriting specific cells with different values—if this results in fewer total operations and reflects a plausible workflow (e.g., "format the whole table, then fix the header row differently"), it may be preferable.

**Default values & heuristic noise**
- When a particular font name or size appears uniformly across a large portion of the sheet, consider whether it's the workbook's default. Common default fonts include Calibri, Arial, Aptos, Aptos Narrow—if one of these appears everywhere, those operations might just be heuristic noise and could be removed. Font sizes 10, 11, or 12 are typical defaults as well. On the other hand, unusual fonts or atypical sizes (very large or very small) were probably applied deliberately and should be kept—possibly consolidated into a single bulk operation if scattered.
- Vertical alignment set to "center" appearing on nearly every formatted cell might be a workbook-level default rather than per-cell styling. Similarly, horizontal alignment defaults to left for text and right for numbers—if alignment operations simply restate these natural defaults, they may be unnecessary.
- White fills are often the workbook default and frequently appear as heuristic artifacts. If white fill operations don't appear to be overwriting a different background color or serving a visible purpose, they can usually be removed. Be mindful of context: sometimes white fill is intentional (e.g., to create contrast against a colored section), but often it's just restating the default.

**Border consolidation**
- Scattered border operations (top, bottom, left, right applied separately) on rectangular regions might actually have been applied as a single "all borders" or "outside borders" action. If you see what looks like fragmented border styling that could plausibly be one range selection, consider whether consolidating to `BORDER_ALL` or `BORDER_OUTSIDE` would better match user intent. Occasionally, if 3 of 4 sides are bordered and the missing side looks accidental, adding it to enable consolidation might be reasonable.

**Paste patterns**
- If a region's content and/or formatting closely mirrors another region (perhaps shifted or tiled), a human might have used copy-paste rather than recreating everything from scratch. When paste would save a significant number of operations and the source-target dimensions make sense, consider whether `PASTE_ALL`, `PASTE_VALUES`, or `PASTE_FORMAT` better represents what actually happened.

**AutoFill patterns**
- If you see a series of consecutive single-cell value operations that form an arithmetic sequence (1, 2, 3…), a text+number progression (Item 1, Item 2…), or formula references incrementing by row/column (=A1+B1, =A2+B2…), a human would likely have typed the first one or two values and dragged (AutoFill) to extend the pattern rather than typing each cell individually.
- When such a pattern is detected, collapsing the repeated operations into source values plus an `AUTOFILL | destination | source` operation better reflects a natural workflow.

**Font color visibility with dark fills**
- If a cell gets a dark fill (black or a color matching the default font color), the text would become invisible unless the font color changes. A natural workflow handles this together—either changing font color immediately before/after applying the dark fill, or applying both in quick succession. If you see a dark fill without a corresponding font color change nearby, it might indicate awkward sequencing or a missing operation.

**Natural variation**
- Humans don't follow a single pattern. Some enter all content first, then format; others format as they go. Some work top-to-bottom; others tackle the most important region first.
- Accept sequences that are *plausibly* human even if they differ from what you'd personally do.
- Reject only when something is *clearly* off—not merely suboptimal.

**Correcting human errors**
- If you notice obvious human mistakes in the content—spelling errors, typos, color coordination conflicts (e.g., clashing fill and font colors), inconsistent formatting within a logical group—you may recommend correcting them.
- These are errors the human likely made unintentionally and would want fixed.
- When recommending such corrections, **explicitly instruct the refiner to add these to the CORRECTIONS section** so the validation system accepts the change.
- Example instruction: "Fix spelling in A5: change `Totla` to `Total` — add to CORRECTIONS section as `Sheet1!A5: \"Totla\" -> \"Total\" (spelling fix)`"
- Without the CORRECTIONS declaration, the validation will fail because the predicted state differs from the original. The refiner must declare all intentional content changes.

---

### Response contract

- Return `HUMAN_SEQUENCE: yes` if the sequence feels human enough.
- Return `HUMAN_SEQUENCE: no` if adjustments are needed.
- Add `RATIONALE:` with two concise bullet-style sentences explaining your verdict (cite specific operations or ranges).
- If `no`, provide **detailed, actionable instructions** the refiner can follow directly:
  - Reference exact operations (e.g., "Move `INPUT | Sheet1!A5 | Total` to after the data rows are filled").
  - Specify ranges when suggesting splits or merges (e.g., "Split `INPUT | Sheet1!A1:A10` into individual cells for the category labels").
  - For unnecessary operations, recommend deletion (e.g., "Delete `FONT_BOLD | Sheet1!M16:M17` — stray formatting on empty column").
  - Explain *why* each change improves human feel (dependency, granularity, timing, etc.).
  - Be thorough—the refiner will apply your instructions as given.
"""


JUDGE_PROMPT_TEMPLATE = """
Sheet under review: **{{ sheet_name }}**
Sequence label: {{ sequence_title }}

#### Delta vs. heuristic plan
{{ diff_summary }}

#### Operations to inspect
{{ operations_block }}

Apply the evaluation checklist from your instructions and respond per the response contract.
"""


def build_judge_prompt(
    *,
    sheet_name: str,
    sequence_title: str,
    operations_block: str,
    diff_summary: str,
    judge_keyword: str,
) -> str:
    return _render(
        JUDGE_PROMPT_TEMPLATE,
        sheet_name=sheet_name,
        sequence_title=sequence_title,
        diff_summary=diff_summary,
        operations_block=operations_block,
        judge_keyword=judge_keyword,
    )


def build_judge_system_prompt() -> str:
    return _render(JUDGE_SYSTEM_PROMPT_TEMPLATE)


JUDGE_FEEDBACK_TEMPLATE = """
{{ emoji }} {{ verdict }} — {{ iteration_label }}
{{ rationale or 'No rationale supplied.' }}

If you revise again, keep the workbook matching exactly while addressing the judge's notes.
"""

JUDGE_RAW_FEEDBACK_TEMPLATE = """
## Judge Feedback

{{ raw_response }}

---
If you revise again, keep the workbook matching exactly while addressing the judge's notes.
"""


def wrap_judge_feedback(raw_response: str) -> str:
    """Wrap raw judge response in a feedback frame for the refiner."""
    return _render(JUDGE_RAW_FEEDBACK_TEMPLATE, raw_response=raw_response.strip())


def build_judge_feedback_message(
    *,
    iteration_label: str,
    is_human: bool,
    rationale: str,
) -> str:
    verdict = "Approved" if is_human else "Needs polish"
    emoji = "\u2705" if is_human else "\u26a0\ufe0f"
    return _render(
        JUDGE_FEEDBACK_TEMPLATE,
        emoji=emoji,
        verdict=verdict,
        iteration_label=iteration_label,
        rationale=rationale.strip(),
    )


def format_operations_block(ops: List[str], limit: Optional[int] = None) -> str:
    if limit is None or limit <= 0 or len(ops) <= limit:
        preview = "\n".join(ops)
        suffix = ""
    else:
        preview = "\n".join(ops[:limit])
        suffix = f"\n... ({len(ops) - limit} more operations)"
    return f"```ops\n{preview}{suffix}\n```" if preview else "(no operations provided)"
