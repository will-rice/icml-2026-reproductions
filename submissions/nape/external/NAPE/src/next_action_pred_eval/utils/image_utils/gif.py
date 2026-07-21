"""
Pure PIL/Pillow-based GIF creation utilities.

Provides functions for creating animated GIFs from image sequences and
drawing annotated frame headers. No xlwings or openpyxl dependency.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Union

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_gif_from_images(
    image_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    duration: Union[int, List[int]] = 500,
    loop: int = 0,
) -> Path:
    """
    Create an animated GIF from a list of image files, anchored at the top-left corner.

    Every frame is padded to the maximum width/height found across all source
    images so that the GIF does not "jump" between frames of different sizes.

    Args:
        image_paths: Ordered list of paths to source image files (PNG, JPEG, etc.).
        output_path: Destination path for the generated GIF.
        duration: Duration per frame in milliseconds.  May be a single int
                  (applied to every frame) or a list with one entry per frame.
        loop: Number of loops (0 = infinite).

    Returns:
        The resolved *output_path* as a ``Path`` object.

    Raises:
        ValueError: If *image_paths* is empty.
    """
    if not image_paths:
        raise ValueError("No image paths provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # First pass: determine maximum canvas size (reads headers only).
    max_width = 0
    max_height = 0
    for path in image_paths:
        with Image.open(path) as img:
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)

    # Second pass: create uniformly-sized frames, closing originals immediately
    # to keep peak memory at roughly 2x one frame instead of N frames.
    processed_frames: List[Image.Image] = []
    for path in image_paths:
        img = Image.open(path)
        canvas = Image.new("RGB", (max_width, max_height), "white")
        canvas.paste(img, (0, 0))
        img.close()
        processed_frames.append(canvas)

    # Write the animated GIF.
    processed_frames[0].save(
        output_path,
        save_all=True,
        append_images=processed_frames[1:],
        duration=duration,
        loop=loop,
    )

    # Release frame memory.
    for frame in processed_frames:
        frame.close()

    logger.info("GIF saved to %s (%d frames)", output_path, len(processed_frames))
    return output_path


# ---------------------------------------------------------------------------
# Frame header drawing
# ---------------------------------------------------------------------------


def draw_frame_header(
    frame_img: Image.Image,
    text_lines: Sequence[str],
    header_height: int = 80,
    background_color: tuple = (245, 245, 245),
    text_color: tuple = (50, 50, 50),
) -> Image.Image:
    """
    Draw a simple text header banner above *frame_img*.

    Args:
        frame_img: The source frame image.
        text_lines: Lines of text to render inside the header area.
        header_height: Pixel height of the header bar.
        background_color: RGB tuple for the header background.
        text_color: RGB tuple for the header text.

    Returns:
        A new ``Image`` with the header prepended above the original frame.
    """
    img_width = frame_img.width
    new_height = frame_img.height + header_height

    new_img = Image.new("RGB", (img_width, new_height), background_color)
    new_img.paste(frame_img, (0, header_height))

    draw = ImageDraw.Draw(new_img)
    font = _load_font(size=20)

    y_offset = 8
    for line in text_lines:
        draw.text((12, y_offset), line, fill=text_color, font=font)
        y_offset += 24

    return new_img


def draw_trajectory_headers(
    frame_img: Image.Image,
    step_num: int,
    total_steps: int,
    source: str,
    accepted: Optional[bool] = None,
    user_count: int = 0,
    model_count: int = 0,
    rejected_count: int = 0,
    header_height: int = 80,
    show_accepted_rejected: bool = False,
    operation_names: Optional[List[str]] = None,
) -> Image.Image:
    """
    Draw left + right sidebars on a trajectory frame image.

    Left sidebar: step counter, progress bar, operation names.
    Right sidebar: source attribution (USER / MODEL) with cumulative stats.

    This is a direct port of ``_draw_headers`` from the reference repo's
    ``gif_utils.py``.

    Args:
        frame_img: The spreadsheet screenshot.
        step_num: Current 1-based step number.
        total_steps: Total number of steps in the trajectory.
        source: ``"USER"`` or ``"MODEL"``.
        accepted: For MODEL frames -- ``True`` if accepted, ``False`` if rejected.
        user_count: Cumulative user step count so far.
        model_count: Cumulative accepted-model step count so far.
        rejected_count: Cumulative rejected-model step count so far.
        header_height: (unused, kept for API compatibility).
        show_accepted_rejected: If ``True``, show ACCEPTED / REJECTED labels.
        operation_names: Optional list of operation type names to display.

    Returns:
        A new ``Image`` with sidebars added.
    """
    img_width = frame_img.width
    img_height = frame_img.height

    sidebar_width = 280
    new_width = img_width + (sidebar_width * 2)
    new_img = Image.new("RGB", (new_width, img_height), (245, 245, 245))

    # Paste original in centre.
    new_img.paste(frame_img, (sidebar_width, 0))

    draw = ImageDraw.Draw(new_img)

    title_font = _load_font(size=32)
    stat_font = _load_font(size=28)
    small_font = _load_font(size=20)
    emoji_font = _load_font(size=42, prefer_emoji=True)

    left_center_x = sidebar_width // 2
    v_center = img_height // 2

    # --- Left sidebar: step counter + progress bar + operation names ---
    draw.text(
        (left_center_x, v_center - 70),
        f"Step {step_num}",
        fill=(50, 50, 50),
        font=title_font,
        anchor="mm",
    )
    draw.text(
        (left_center_x, v_center - 35),
        f"of {total_steps}",
        fill=(120, 120, 120),
        font=stat_font,
        anchor="mm",
    )

    # Progress bar
    bar_width = sidebar_width - 50
    bar_height = 12
    bar_x = (sidebar_width - bar_width) // 2
    bar_y = v_center + 10

    draw.rectangle(
        [bar_x, bar_y, bar_x + bar_width, bar_y + bar_height],
        fill=(210, 210, 210),
        outline=(180, 180, 180),
    )
    progress = (step_num / total_steps) * bar_width
    if progress > 0:
        draw.rectangle(
            [bar_x, bar_y, bar_x + progress, bar_y + bar_height],
            fill=(66, 135, 245),
        )

    # Operation names (below progress bar)
    if operation_names:
        ops_y = bar_y + bar_height + 25
        display_names = operation_names[:6]
        if len(operation_names) > 6:
            display_names.append("...")
        ops_text = ",\n".join(display_names)
        draw.multiline_text(
            (bar_x, ops_y), ops_text, fill=(140, 140, 140), font=small_font, align="left"
        )

    # --- Right sidebar: source attribution ---
    right_center_x = img_width + sidebar_width + (sidebar_width // 2)

    icon_y = v_center - 70
    source_text_y = v_center - 25
    status_text_y = v_center + 10
    stats_start_y = v_center + 50
    line_height = 28

    # Determine colours / labels
    if source == "USER":
        source_emoji = "\U0001F464"  # bust-in-silhouette
        source_label = "USER"
        source_color = (66, 135, 245)
        status_text = None
    else:
        source_label = "MODEL"
        if show_accepted_rejected:
            if accepted:
                source_emoji = "\u2713"
                status_text = "ACCEPTED"
                source_color = (46, 125, 50)
            else:
                source_emoji = "\u2717"
                status_text = "REJECTED"
                source_color = (198, 40, 40)
        else:
            source_emoji = "\u2713"
            status_text = None
            source_color = (46, 125, 50)

    # Emoji / icon
    try:
        draw.text(
            (right_center_x, icon_y),
            source_emoji,
            fill=source_color,
            font=emoji_font,
            anchor="mm",
        )
    except Exception:
        pass  # Emoji rendering may fail on some systems

    # Source label
    draw.text(
        (right_center_x, source_text_y),
        source_label,
        fill=source_color,
        font=title_font,
        anchor="mm",
    )

    # Status text (detailed view only)
    if status_text:
        draw.text(
            (right_center_x, status_text_y),
            status_text,
            fill=source_color,
            font=stat_font,
            anchor="mm",
        )

    # Cumulative stats
    draw.text(
        (right_center_x, stats_start_y),
        f"User: {user_count}",
        fill=(66, 135, 245),
        font=stat_font,
        anchor="mm",
    )

    if show_accepted_rejected:
        draw.text(
            (right_center_x, stats_start_y + line_height),
            f"Accepted: {model_count}",
            fill=(46, 125, 50),
            font=stat_font,
            anchor="mm",
        )
        draw.text(
            (right_center_x, stats_start_y + line_height * 2),
            f"Rejected: {rejected_count}",
            fill=(198, 40, 40),
            font=stat_font,
            anchor="mm",
        )
    else:
        draw.text(
            (right_center_x, stats_start_y + line_height),
            f"Model: {model_count}",
            fill=(46, 125, 50),
            font=stat_font,
            anchor="mm",
        )

    return new_img


# ---------------------------------------------------------------------------
# Operation-name extraction
# ---------------------------------------------------------------------------


def extract_operation_names(operations: Sequence) -> List[str]:
    """
    Extract unique operation-type names from a list of operations.

    Supported formats:

    * **Pipe-delimited string** -- ``"INPUT | STA!A1 | value"`` -> ``"INPUT"``
    * **Function-call string** -- ``"SetValue(Sheet1!A1, ...)"`` -> ``"SetValue"``
    * **Dict with 'type' key** -- ``{"type": "SetValue", ...}`` -> ``"SetValue"``

    Returns:
        A sorted, de-duplicated list of operation-type names.
    """
    names: set = set()

    for op in operations:
        if isinstance(op, str):
            if " | " in op:
                op_name = op.split(" | ")[0].strip()
                if op_name:
                    names.add(op_name)
            else:
                match = re.match(r"^([A-Za-z_]+)\(", op)
                if match:
                    names.add(match.group(1))
        elif isinstance(op, dict):
            if "type" in op:
                names.add(op["type"])
            elif "operation_type" in op:
                names.add(op["operation_type"])

    return sorted(names)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FONT_CACHE: dict = {}


def _load_font(size: int = 20, prefer_emoji: bool = False) -> ImageFont.FreeTypeFont:
    """
    Load a TrueType font at the requested size, falling back to the default
    bitmap font if none of the known paths are available.
    """
    cache_key = (size, prefer_emoji)
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]

    if prefer_emoji:
        candidates = ["seguiemj.ttf", "Segoe UI Emoji.ttf", "NotoColorEmoji.ttf"]
    else:
        candidates = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]

    for name in candidates:
        try:
            font = ImageFont.truetype(name, size)
            _FONT_CACHE[cache_key] = font
            return font
        except (OSError, IOError):
            continue

    font = ImageFont.load_default()
    _FONT_CACHE[cache_key] = font
    return font
