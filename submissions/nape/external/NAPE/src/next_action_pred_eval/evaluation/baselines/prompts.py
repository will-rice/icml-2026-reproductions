"""
Prompts for LLM-based Prediction
Contains prompt templates for the LLM baseline solver.

Templates use Jinja2 syntax.  Available variables:

Chat mode – system prompt (``DEFAULT_SYSTEM_TEMPLATE``):
    emit_intent : bool   – whether to add the "Intent:" instruction
    num_op_to_pred : int | None – max-op cap (omitted when None)

Chat mode – user prompt (``DEFAULT_USER_TEMPLATE``):
    sheet_name : str | None  – active worksheet name
    previous_actions : str   – pre-formatted numbered action list
    num_actions : int        – number of actions in the list

Completion mode (``DEFAULT_COMPLETION_TEMPLATE``):
    All of the above.  A short, minimalistic prompt suitable for
    raw-completion endpoints.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import jinja2

logger = logging.getLogger(__name__)

TRUNCATION_MARKER = "..."

# ---------------------------------------------------------------------------
# Default Jinja2 templates
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_TEMPLATE = """\
{%- set single = (num_op_to_pred == 1) -%}
You are an autocomplete copilot for spreadsheet authors. {% if single %}Given the user's recent editing history, predict the single most likely next operation they will perform.{% else %}Continue the user's current workflow without inventing new, unrelated workstreams.{% endif %}

Decision rubric:
1. Infer the immediate intent from the latest few steps (table build, formatting sweep, etc.).{% if not single %} Finish that chunk before starting anything new.{% endif %}
2. Keep ranges tight and contiguous unless the history explicitly shows a jump to a different area.
3. Only suggest {% if single %}an operation that clearly advances{% else %}operations that clearly advance{% endif %} the current goal. If you cannot justify {% if single %}it{% else %}an action{% endif %} with the visible history, omit it.
{%- if not single %}
4. Never repeat large historical blocks verbatim\u2014mirror the pattern, not the entire plan.
{%- endif %}
{{ '4' if single else '5' }}. Stop as soon as the workflow looks complete or uncertain.

Output contract:
- {% if single %}Emit exactly one well-justified, high-confidence operation{% else %}Emit only well-justified, high-confidence steps{% endif %} in the format: OPERATION | RANGE | VALUE
- Ranges do not include sheet names (the active sheet is provided separately).
- Prefer formulas with relative references that match the user's existing convention.
{%- if emit_intent %}
- Begin with a single intent line (max 30 words) prefixed with "Intent:" summarizing the user's current goal. Then list operations.
{%- endif %}
{%- if num_op_to_pred %}
- Do not exceed {{ num_op_to_pred }} total operations. Returning fewer (including zero) is acceptable when intent is ambiguous.
{%- endif %}

Available operations (use EXACT names, no wildcards):
  Data: INPUT, PASTE_FROM, AUTOFILL
  Formatting: NUMBER_FORMAT, FILL_COLOR
  Font: FONT_BOLD, FONT_ITALIC, FONT_SIZE, FONT_COLOR, FONT_UNDERLINE, FONT_NAME
  Alignment: ALIGN_HORIZONTAL, ALIGN_VERTICAL
  Borders: BORDER_LEFT, BORDER_RIGHT, BORDER_TOP, BORDER_BOTTOM, BORDER_OUTSIDE, BORDER_ALL, BORDER_INSIDE_HORIZONTAL, BORDER_INSIDE_VERTICAL
  Other: MERGE, UNMERGE, WRAP_TEXT, TEXT_ORIENTATION

Value formats by operation type:
  INPUT: JSON-encoded \u2014 strings ("Hello"), numbers (123), booleans (true/false),
         formulas ("=SUM(A1:B1)"), 2D arrays ([["a","b"],["c","d"]]), or clear
  FILL_COLOR / FONT_COLOR: Hex color code (e.g., #FF0000) or clear
  FONT_BOLD / FONT_ITALIC / WRAP_TEXT: true or false
  FONT_SIZE: Numeric (e.g., 12, 14.5)
  FONT_UNDERLINE: none, single, double, singleAccounting, doubleAccounting
  FONT_NAME: Font family name (e.g., Calibri, Arial)
  ALIGN_HORIZONTAL: left, center, right, justify, general
  ALIGN_VERTICAL: top, center, bottom, justify
  NUMBER_FORMAT: Excel format code (e.g., #,##0.00, 0.00%, mm/dd/yyyy, General)
  TEXT_ORIENTATION: Integer degrees (-90 to 90)
  MERGE / UNMERGE: true or false
  BORDER_*: Weight, Style, Color (e.g., Thin, Continuous, #000000) or clear
    Weights: Hairline, Thin, Medium, Thick
    Styles: Continuous, Dash, Dot, DashDot, DashDotDot, Double, SlantDashDot
  PASTE_FROM: see Rules for PASTE_FROM below
  AUTOFILL: see Rules for AUTOFILL below

Rules for INPUT:
- Single cell: INPUT | A1 | "Hello World"
- Multiple cells: INPUT | A1:C2 | [["a","b","c"],["d","e","f"]]
  The 2D array dimensions MUST exactly match the range (rows x cols). Do NOT abbreviate or truncate.

Rules for PASTE_FROM:
- Format: PASTE_FROM | destination | source_range | mode
  Modes: all, values, formats, formulas
- Example: PASTE_FROM | A10:C12 | A1:C3 | all

Rules for AUTOFILL:
- Format: AUTOFILL | destination_range | source_range
  The destination must fully contain the source and extend on exactly one axis.
  Direction is inferred: same columns = vertical, same rows = horizontal.
- Pattern detection: numbers form arithmetic series, text cycles, text+numbers increment.
  Single numbers copy.
- Example: AUTOFILL | A1:A10 | A1:A3

Clearing and resetting:
- Use "clear" to erase content (INPUT), fill (FILL_COLOR), or borders (BORDER_*).
- To reset formatting, set the Excel default value directly (e.g., FONT_BOLD \u2192 false, NUMBER_FORMAT \u2192 General).

Note on history context:
- Some operations in the history may show "..." as a placeholder inside large arrays or long values to save context space. For 2D arrays this shows the corner cells with "..." in the gaps. You MUST NOT use this marker in your predictions. Always provide the full, exact values.

{% if emit_intent -%}
Example output:
  Intent: Building a formatted data table with headers, borders, and formulas.
{%- else -%}
Example syntax:
{%- endif %}
  INPUT | A1 | "Hello World"
  INPUT | A1:B2 | [["a","b"],["c","d"]]
  INPUT | C1 | "=SUM(A1:B1)"
  INPUT | A1:A5 | clear
  FILL_COLOR | A1:B5 | #FF0000
  FILL_COLOR | C1 | clear
  BORDER_ALL | A1:C3 | Thin, Continuous, #000000
  FONT_BOLD | A1 | true
  MERGE | A1:C1 | true
  UNMERGE | A1:C1 | false
  PASTE_FROM | A10:C12 | A1:C3 | all
  AUTOFILL | A1:A10 | A1:A3"""

DEFAULT_USER_TEMPLATE = """\
{% if sheet_name -%}
Active sheet: {{ sheet_name }}

{% endif -%}
Recent operation history:
{{ previous_actions }}"""

DEFAULT_COMPLETION_TEMPLATE = """\
Complete the sequence of actions to build the following spreadsheet by identifying and extending key patterns.
{% if emit_stop_instruction %}Write STOP as soon as you're uncertain of the next step or if the workflow looks complete.
{% endif %}
{{ previous_actions }}
"""

# ---------------------------------------------------------------------------
# Backward-compatible aliases (kept for any external code importing them)
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_TEMPLATE
DEFAULT_USER_PROMPT_TEMPLATE = DEFAULT_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Template loading & rendering
# ---------------------------------------------------------------------------

def _compile_template(source: str) -> jinja2.Template:
    """Compile a Jinja2 template from a raw string.

    Uses ``jinja2.Environment(undefined=jinja2.Undefined)`` so that
    missing variables render as empty strings rather than raising.
    """
    env = jinja2.Environment(undefined=jinja2.Undefined)
    return env.from_string(source)


def load_prompt_template(
    template_file: Optional[str] = None,
    template_inline: Optional[str] = None,
    default_template: str = "",
) -> str:
    """Resolve a prompt template source.

    Priority: *template_file* → *template_inline* → *default_template*.

    Args:
        template_file: Absolute or pre-resolved path to a ``.jinja2`` /
            ``.txt`` file.  ``None`` to skip.
        template_inline: Inline template string.  ``None`` to skip.
        default_template: Built-in default used when both file and inline
            are ``None``.

    Returns:
        The raw template string (not yet compiled).
    """
    if template_file:
        path = Path(template_file)
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt template file not found: {path}"
            )
        logger.debug("Loading prompt template from %s", path)
        return path.read_text(encoding="utf-8")
    if template_inline:
        return template_inline
    return default_template


def render_prompt_template(
    template_source: str,
    context: dict,
) -> str:
    """Compile and render a Jinja2 template string with *context*.

    Args:
        template_source: Raw Jinja2 template string.
        context: Variables to inject into the template.

    Returns:
        Rendered prompt string.
    """
    tmpl = _compile_template(template_source)
    return tmpl.render(**context)


# ---------------------------------------------------------------------------
# High-level prompt builder
# ---------------------------------------------------------------------------

def _format_actions(actions: List[str]) -> str:
    """Format a list of symbolic actions as a newline-separated string."""
    if actions:
        return "\n".join(actions)
    return "(No prior steps \u2014 begin with foundational setup.)"


def create_prediction_prompt(
    previous_actions: List[str],
    max_context: Optional[int] = 50,
    sheet_name: Optional[str] = None,
    num_op_to_pred: Optional[int] = None,
    emit_intent: bool = False,
    is_completion: bool = False,
    system_prompt_file: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt_file: Optional[str] = None,
    user_prompt: Optional[str] = None,
    completion_prompt_file: Optional[str] = None,
    completion_prompt: Optional[str] = None,
    # Deprecated — kept for backward compat; mapped to inline overrides
    custom_system_prompt: Optional[str] = None,
    custom_user_template: Optional[str] = None,
    emit_stop_instruction: bool = False,
) -> Tuple[Optional[str], str]:
    """
    Create a prediction prompt from previous actions.

    Supports two modes controlled by *is_completion*:

    ``False`` (default — **chat mode**)
        Returns ``(system_prompt_str, user_prompt_str)`` — two separate
        messages sent to the LLM.

    ``True`` (**completion mode**)
        Returns ``(None, completion_prompt_str)`` — a single prompt string
        suitable for raw-completion endpoints.

    Template resolution (per slot):
        file path → inline string → deprecated legacy param → built-in default.

    All templates are Jinja2.  Available variables:

    *System*: ``emit_intent`` (bool), ``num_op_to_pred`` (int|None).
    *User*: ``sheet_name`` (str|None), ``previous_actions`` (str),
    ``num_actions`` (int).
    *Completion*: all of the above.

    Args:
        previous_actions: List of symbolic operation strings.
        max_context: Maximum number of operations to include.
        sheet_name: Active worksheet name.
        num_op_to_pred: Upper bound on predicted operations.
        emit_intent: Ask LLM to emit an intent line.
        is_completion: If True, return a single completion prompt instead
            of split system/user messages.
        system_prompt_file: Path to system prompt Jinja2 file.
        system_prompt: Inline system prompt Jinja2 string.
        user_prompt_file: Path to user prompt Jinja2 file.
        user_prompt: Inline user prompt Jinja2 string.
        completion_prompt_file: Path to completion prompt Jinja2 file.
        completion_prompt: Inline completion prompt Jinja2 string.
        custom_system_prompt: *Deprecated* — alias for *system_prompt*.
        custom_user_template: *Deprecated* — alias for *user_prompt*.

    Returns:
        ``(system_str, user_str)`` when *is_completion* is False, or
        ``(None, completion_str)`` when *is_completion* is True.
    """
    # Map deprecated params as fallback for inline overrides
    effective_system_inline = system_prompt or custom_system_prompt
    effective_user_inline = user_prompt or custom_user_template

    # Truncate context
    if max_context and len(previous_actions) > max_context:
        actions = previous_actions[-max_context:]
    else:
        actions = previous_actions

    # Pre-format actions
    actions_str = _format_actions(actions)

    # Build shared template context
    ctx = {
        "emit_intent": emit_intent,
        "num_op_to_pred": num_op_to_pred,
        "sheet_name": sheet_name,
        "previous_actions": actions_str,
        "num_actions": len(actions),
        "emit_stop_instruction": emit_stop_instruction,
    }

    if is_completion:
        tmpl_source = load_prompt_template(
            template_file=completion_prompt_file,
            template_inline=completion_prompt,
            default_template=DEFAULT_COMPLETION_TEMPLATE,
        )
        rendered = render_prompt_template(tmpl_source, ctx)
        return None, rendered

    # --- chat mode (default) ---
    sys_source = load_prompt_template(
        template_file=system_prompt_file,
        template_inline=effective_system_inline,
        default_template=DEFAULT_SYSTEM_TEMPLATE,
    )
    usr_source = load_prompt_template(
        template_file=user_prompt_file,
        template_inline=effective_user_inline,
        default_template=DEFAULT_USER_TEMPLATE,
    )

    system_str = render_prompt_template(sys_source, ctx)
    user_str = render_prompt_template(usr_source, ctx)
    return system_str, user_str


def _truncate_2d_corners(
    parsed: list,
    corner_cells_dim: int,
) -> list:
    """Build a corners-preview of a 2D array.

    Shows ``corner_cells_dim`` rows/cols from each of the four corners
    with ``"..."`` filling the gaps, similar to how pandas displays large
    DataFrames.
    """
    num_rows = len(parsed)
    num_cols = max((len(r) for r in parsed), default=0)

    r = min(corner_cells_dim, num_rows)
    c = min(corner_cells_dim, num_cols)

    need_row_gap = num_rows > 2 * r
    need_col_gap = num_cols > 2 * c

    def _shorten_row(row: list) -> list:
        if need_col_gap:
            return row[:c] + ["..."] + row[-c:]
        return list(row)

    top_rows = [_shorten_row(parsed[i]) for i in range(r)]
    bot_rows = [_shorten_row(parsed[i]) for i in range(num_rows - r, num_rows)]

    if need_row_gap:
        # Ellipsis row with same width as the shortened rows
        sample_width = len(top_rows[0]) if top_rows else (2 * c + 1 if need_col_gap else num_cols)
        gap_row = ["..."] * sample_width
        return top_rows + [gap_row] + bot_rows
    else:
        # Few enough rows — just show them all (columns may still be truncated)
        return [_shorten_row(parsed[i]) for i in range(num_rows)]


def shorten_symbolic_values(
    symbolic_ops: List[str],
    max_value_length: int = 128,
    max_cells_2d: Optional[int] = None,
    corner_cells_dim: int = 3,
) -> List[str]:
    """
    Shorten long values in symbolic operations to reduce context size.

    Large 2D arrays are replaced with a corners-preview that shows
    ``corner_cells_dim`` rows and columns from each of the four corners,
    with ``"..."`` filling the gaps.  Long string values are truncated.

    Args:
        symbolic_ops: List of symbolic operation strings.
        max_value_length: Maximum character length for individual values.
        max_cells_2d: Maximum total cells for 2D arrays before truncation.
            ``None`` (default/auto) = ``corner_cells_dim ** 2 * 4``.
        corner_cells_dim: Number of rows/cols to keep from each corner.

    Returns:
        List of symbolic operations with shortened values.
    """
    import json

    if max_cells_2d is None:
        max_cells_2d = corner_cells_dim ** 2 * 4

    # Operation types where the value is a cell reference, not data content.
    # These must never be truncated — truncation corrupts the range reference.
    _REF_VALUE_OPS = frozenset({"AUTOFILL", "PASTE_FROM"})

    shortened = []
    for op in symbolic_ops:
        parts = op.split(" | ")
        if len(parts) >= 3:
            op_type = parts[0]
            cell_ref = parts[1]
            value = parts[2]
            remaining = parts[3:] if len(parts) > 3 else []

            # Skip truncation for ops whose value is a cell reference
            if op_type not in _REF_VALUE_OPS:
                # Try to parse as JSON
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
                        # 2D array — check total cells
                        total_cells = sum(len(row) for row in parsed)
                        if total_cells > max_cells_2d:
                            value = json.dumps(
                                _truncate_2d_corners(parsed, corner_cells_dim)
                            )
                    elif isinstance(parsed, str) and len(parsed) > max_value_length:
                        if not parsed.startswith("="):
                            value = json.dumps(parsed[:max_value_length] + "...")
                except (json.JSONDecodeError, TypeError):
                    if len(value) > max_value_length:
                        raw = value.strip().lstrip('"')
                        if not raw.startswith("="):
                            value = value[:max_value_length] + "..."

            # Rebuild operation
            if remaining:
                shortened.append(" | ".join([op_type, cell_ref, value] + remaining))
            else:
                shortened.append(" | ".join([op_type, cell_ref, value]))
        else:
            shortened.append(op)

    return shortened


__all__ = [
    "DEFAULT_SYSTEM_TEMPLATE",
    "DEFAULT_USER_TEMPLATE",
    "DEFAULT_COMPLETION_TEMPLATE",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_PROMPT_TEMPLATE",
    "TRUNCATION_MARKER",
    "create_prediction_prompt",
    "load_prompt_template",
    "render_prompt_template",
    "shorten_symbolic_values",
]
